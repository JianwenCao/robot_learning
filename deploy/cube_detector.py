"""Off-the-shelf cube segmenter for the real-robot wrist image.

The default real-robot mask in :mod:`deploy.deploy_real` is an HSV gate +
largest-CC pick. That works when the cube is the only saturated thing in
view, but breaks the policy when a colored object (tool handle, cable,
sticker, hand) enters the wrist FOV — the gate fires on the distractor, the
largest-CC pick latches onto the wrong blob, and the policy reads a mask
that has nothing to do with the cube.

This module replaces the mask channel with a text-prompted open-vocabulary
segmenter so the upstream image distribution is closer to sim's
``semantic_segmentation`` filtered to ``class:block``. It runs at deploy
time only; sim training is unchanged.

Design choices
--------------
* **Florence-2-base** (~230M params, ~1 GB on disk). Single-model,
  single-forward-pass design (no DINO+SAM cascade) keeps integration small.
  Runs in ~2–5 s/frame on a modern laptop CPU — slow but acceptable for the
  "non-real-time, success matters" deploy budget. Policy is reactive, so a
  reduced control rate is fine; the slew cap inside ``deploy_real`` already
  bounds per-step joint motion.
* **Task**: ``<REFERRING_EXPRESSION_SEGMENTATION>``. Returns polygons in
  original image coordinates; we rasterize to a binary mask and run the same
  ``keep_largest_component`` cleanup as the HSV path so the policy sees the
  same one-blob shape it was trained on.
* **Detector is a Protocol**, so swapping in GroundedSAM / SAM-3 / etc.
  later is a single class drop-in; the rest of ``deploy_real`` keeps its
  ``mask(rgb) -> (H, W) float`` boundary.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    """Wrist-image cube segmenter.

    Returns a single-channel float32 binary mask in ``[0, 1]``, same H×W as
    the input RGB. Downsampling to the policy's 72×128 input happens in
    :func:`deploy.deploy_real._build_image` so each detector implementation
    can pick whatever input resolution it wants for the model.

    Implementations that take a text prompt (Florence, CLIPSeg, …) should
    also implement :meth:`set_prompt` so the Eval-3 deploy loop can re-key
    the detector per sub-goal without reloading the model.
    """

    def mask(self, rgb_hwc_uint8: np.ndarray) -> np.ndarray: ...

    def set_prompt(self, prompt: str) -> None: ...


def keep_largest_component(mask01: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Collapse a binary mask to its single largest connected component.

    Matches the post-processing in ``deploy_real._hsv_mask`` so detector-
    swap doesn't change the mask's topological structure. Sim's
    ``semantic_segmentation`` filtered to one class is always one blob, so
    the policy never saw multi-blob masks during training.
    """
    import cv2
    bin8 = (mask01 > 0).astype(np.uint8)
    if bin8.sum() == 0:
        return np.zeros_like(mask01, dtype=np.float32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask01, dtype=np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if stats[best, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask01, dtype=np.float32)
    return (labels == best).astype(np.float32)


class FlorenceDetector:
    """Florence-2 referring-expression segmentation, text-prompted.

    One forward pass per ``mask(rgb)`` call. CPU latency on a modern laptop:
    ~2–5 s/frame at ``num_beams=1`` (greedy). Bumping ``num_beams`` improves
    quality on ambiguous prompts but multiplies decode cost.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        prompt: str = "small wooden cube",
        device: str = "cpu",
        num_beams: int = 1,
        max_new_tokens: int = 512,
        min_area: int = 6,
    ):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(
                "Florence-2 detector requires `transformers`. Install via "
                "deploy/setup_inference_pc.sh (the script installs the model deps)."
            ) from e
        import torch

        self._torch = torch
        # ``trust_remote_code`` because Florence-2 ships its modeling +
        # processing code in the model repo (no first-class transformers
        # integration as of model release).
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float32
            )
            .eval()
            .to(device)
        )
        self.device = device
        self.prompt = prompt
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.min_area = min_area
        self._task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        self._full_prompt = self._task + prompt
        print(
            f"[florence] loaded {model_id} (device={device}, "
            f"prompt={prompt!r}, num_beams={num_beams})"
        )

    def set_prompt(self, prompt: str) -> None:
        """Re-key the detector to a new text prompt.

        Cheap (just string concat) — Florence reads ``self._full_prompt``
        each :meth:`mask` call. Used by the Eval-3 deploy loop to switch
        the target color between sub-goals without reloading the model.
        """
        self.prompt = prompt
        self._full_prompt = self._task + prompt
        print(f"[florence] prompt → {prompt!r}")

    def mask(self, rgb_hwc_uint8: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        H, W = rgb_hwc_uint8.shape[:2]
        pil = Image.fromarray(rgb_hwc_uint8)
        inputs = self.processor(
            text=self._full_prompt, images=pil, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=self._task, image_size=(W, H)
        )
        result = parsed.get(self._task, {}) or {}
        polygons = result.get("polygons", []) or []

        # Rasterize: ``polygons`` is list[instance] of list[polygon] of
        # [x1, y1, x2, y2, ...] flat coords. Multiple instances are unioned
        # before the largest-CC pick, so the detector returning two halves
        # of the cube doesn't get filtered out.
        out = np.zeros((H, W), dtype=np.uint8)
        for inst in polygons:
            for poly in inst:
                pts = (
                    np.asarray(poly, dtype=np.float32)
                    .reshape(-1, 2)
                    .astype(np.int32)
                )
                if len(pts) >= 3:
                    cv2.fillPoly(out, [pts], 255)

        m01 = (out > 0).astype(np.float32)
        return keep_largest_component(m01, min_area=self.min_area)


def build_detector(
    source: str, prompt: str = "small wooden cube", device: str = "cpu"
) -> Detector | None:
    """Factory used by :func:`deploy.deploy_real.run`.

    Returns ``None`` for ``source='hsv'`` so the caller falls back to its
    built-in ``_hsv_mask`` (the original code path). Any other string is
    looked up against the registered model classes.
    """
    if source == "hsv":
        return None
    if source == "florence":
        return FlorenceDetector(prompt=prompt, device=device)
    raise ValueError(
        f"unknown --mask-source: {source!r}. Valid: hsv, florence."
    )
