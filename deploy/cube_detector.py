"""AprilTag cube-localisation detector for the real-robot deploy loop.

The state-only + AprilTag deploy path uses :class:`AprilTagDetector` to
return ``((cube_x, cube_y), valid)`` in the robot base frame given an
RGB frame and the current ``T_base_ee``. The script in
:mod:`deploy.deploy_real` calls it once per step and feeds the result
into the policy's 27-D state vector.

See ``docs/EVAL1_PLAN.md`` for the full pipeline spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_DEPLOY_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEPLOY_DIR.parent


def _load_camera_intrinsics(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, dist)`` from ``camera_intrinsics.yaml``.

    Mirrors ``deploy.driver._load_intrinsics`` (kept inline there to
    avoid an import cycle during dry-run smoke tests). The YAML embeds
    ``!!python/object/apply`` numpy pickles in ``projection_matrix`` which
    fail to load on newer numpy; we only read ``camera_matrix`` and
    ``distortion_coefficients`` so we install a permissive Loader that
    yields ``None`` for any unknown Python-tagged node.
    """
    import yaml

    p = Path(path) if path is not None else _PROJECT_ROOT / "camera_intrinsics.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"camera_intrinsics.yaml not found at {p}. "
            "Run wrist-cam calibration first (see deploy/README.md)."
        )

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:", lambda l, t, n: None
    )
    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/object/new:", lambda l, t, n: None
    )
    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/name:", lambda l, t, n: None
    )
    with open(p, "r") as f:
        data = yaml.load(f, Loader=_Loader)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
    return K, dist


def _load_hand_eye(path: str | Path | None = None) -> np.ndarray:
    """Return ``T_ee_cam`` (4×4) from ``deploy/hand_eye.yaml``.

    The YAML is written by ``deploy/calibrate_hand_eye.py``. Schema:

    .. code-block:: yaml

        T_ee_cam:
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0.0, 0.0, 0.0, 1.0]
    """
    import yaml

    p = Path(path) if path is not None else _DEPLOY_DIR / "hand_eye.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"hand_eye.yaml not found at {p}. Run "
            "`python -m deploy.calibrate_hand_eye` first (see deploy/README.md §Step 4)."
        )
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    T = np.array(data["T_ee_cam"], dtype=np.float64).reshape(4, 4)
    return T


class AprilTagDetector:
    """AprilTag pose detector — returns cube (x, y) in robot base frame.

    Loads ``pupil-apriltags`` at construction (CPU detector, ~2 ms/frame
    on the inference PC), the camera intrinsics from
    ``camera_intrinsics.yaml``, and the hand-eye transform ``T_ee_cam``
    from ``deploy/hand_eye.yaml``.

    Two API levels:

    * :meth:`detect` returns *all* visible tags with their full ``T_cam_tag``
      poses — useful for the hand-eye calibration script and for debugging.
    * :meth:`pose` is the convenience the deploy loop uses:
      ``pose(rgb, T_base_ee) -> ((x, y), valid)``. It filters to
      ``self.target_id`` and composes the chain
      ``T_base_tag = T_base_ee · T_ee_cam · T_cam_tag``.

    Eval-2/3 switch the target between sub-goals via :meth:`set_target_id`.
    """

    def __init__(
        self,
        family: str = "tagStandard41h12",
        tag_size_m: float = 0.015,
        target_id: int = 0,
        intrinsics_yaml: str | Path | None = None,
        hand_eye_yaml: str | Path | None = None,
    ):
        try:
            from pupil_apriltags import Detector as _AprilDetector
        except ImportError as e:
            raise RuntimeError(
                "AprilTagDetector requires `pupil-apriltags`. Install via "
                "deploy/setup_inference_pc.sh."
            ) from e

        self._detector = _AprilDetector(families=family)
        self.family = family
        self.tag_size_m = tag_size_m
        self.target_id = int(target_id)

        K, dist = _load_camera_intrinsics(intrinsics_yaml)
        self.K = K
        self.dist = dist
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        self.T_ee_cam = _load_hand_eye(hand_eye_yaml)

        print(
            f"[apriltag] family={family}, tag_size={tag_size_m*1000:.0f} mm, "
            f"target_id={target_id}"
        )

    def set_target_id(self, tag_id: int) -> None:
        """Switch the active target tag (Eval-2/3 per-sub-goal switch)."""
        self.target_id = int(tag_id)
        print(f"[apriltag] target_id → {self.target_id}")

    def detect(self, rgb_hwc_uint8: np.ndarray) -> list[dict]:
        """Return list of detections with ``T_cam_tag`` (4×4), corners, etc.

        Caller is responsible for *undistorting* the RGB first
        (``cv2.undistort`` with the intrinsics from ``camera_intrinsics.yaml``)
        so the PnP solution from ``estimate_tag_pose=True`` matches the
        ideal pinhole the camera_params imply.
        """
        import cv2

        gray = cv2.cvtColor(rgb_hwc_uint8, cv2.COLOR_RGB2GRAY)
        results = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=self.tag_size_m,
        )
        out: list[dict] = []
        for r in results:
            T_cam_tag = np.eye(4, dtype=np.float64)
            T_cam_tag[:3, :3] = np.asarray(r.pose_R, dtype=np.float64)
            T_cam_tag[:3, 3] = np.asarray(r.pose_t, dtype=np.float64).flatten()
            out.append(
                {
                    "tag_id": int(r.tag_id),
                    "T_cam_tag": T_cam_tag,
                    "corners": np.asarray(r.corners, dtype=np.float64),
                    "center": np.asarray(r.center, dtype=np.float64),
                    "decision_margin": float(r.decision_margin),
                }
            )
        return out

    def pose(
        self,
        rgb_hwc_uint8: np.ndarray,
        T_base_ee: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Return ``((x, y), valid)`` for the active target tag in base frame.

        ``T_base_ee`` is the current end-effector pose in base frame (from
        URDF FK on the live joint positions). ``valid`` is ``False`` when
        no detection matches ``self.target_id`` this frame — the deploy
        loop's job is to hold the last value in that case (matches the
        sim-side dropout / freeze behavior in
        :func:`mdp.observations.cube_pos_xy_noisy`).
        """
        dets = self.detect(rgb_hwc_uint8)
        target = next((d for d in dets if d["tag_id"] == self.target_id), None)
        if target is None:
            return np.zeros(2, dtype=np.float32), False
        T_base_tag = T_base_ee @ self.T_ee_cam @ target["T_cam_tag"]
        return T_base_tag[:2, 3].astype(np.float32), True
