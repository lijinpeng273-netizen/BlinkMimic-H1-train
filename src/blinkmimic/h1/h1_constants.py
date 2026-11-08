"""H1 asset, PD controls and named NPZ motions imported from the successful H1 run.

Run tools/prepare_h1.py once. Training has no dependency on the SuperMimic package.
"""

from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from blinkmimic.dr import DomainRandCfg
from blinkmimic.robot import MotionArrays, RobotSpec

REPO = Path(__file__).resolve().parents[3]
MJCF_PATH = REPO / "data/h1_description/h1_training.xml"
MOTION_DIR = REPO / "data/npz/h1"
H1_JOINT_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link_ball_hand",
    "right_shoulder_roll_link",
    "right_elbow_link_ball_hand",
)
DEFAULT_POS = [0.0, 0.0, -0.4, 0.8, -0.4] * 2 + [0.0] * 9
STIFFNESS = (
    [150.0, 150.0, 150.0, 200.0, 40.0] * 2 + [300.0] + [100.0, 100.0, 100.0, 50.0] * 2
)
DAMPING = [2.0, 2.0, 2.0, 4.0, 2.0] * 2 + [3.0] + [2.0] * 8
EFFORT_LIMIT = (
    [200.0, 200.0, 200.0, 300.0, 40.0] * 2 + [200.0] + [40.0, 40.0, 18.0, 18.0] * 2
)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    model = spec.compile()
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if (model.nq, model.nv, model.nu) != (26, 25, 0) or names != H1_JOINT_NAMES:
        raise ValueError(
            "Expected the prepared H1 asset: free pelvis, 19 ordered hinges, no XML actuators"
        )
    return spec


def get_robot_cfg() -> EntityCfg:
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            armature=0.1,
            viscous_damping=1.0,
        )
        for name, kp, kd, effort in zip(
            H1_JOINT_NAMES, STIFFNESS, DAMPING, EFFORT_LIMIT
        )
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.98),
            joint_pos=dict(zip(H1_JOINT_NAMES, DEFAULT_POS)),
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=0.9
        ),
    )


def load_h1_npz(path: str, expected_asset_hash: str | None = None) -> MotionArrays:
    from blinkmimic.checkpoint import asset_sha256

    expected = expected_asset_hash or asset_sha256(str(MJCF_PATH))
    with np.load(path, allow_pickle=False) as d:
        required = {
            "robot",
            "asset_sha256",
            "joint_names",
            "body_names",
            "fps",
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
        }
        if not required.issubset(d.files):
            raise ValueError(
                f"{path}: missing named H1 metadata/arrays: {sorted(required - set(d.files))}"
            )
        if d["robot"].item() != "h1" or d["asset_sha256"].item() != expected:
            raise ValueError(
                f"{path}: robot/asset identity does not match the H1 training asset"
            )
        names = [str(n) for n in d["joint_names"]]
        bodies = [str(n) for n in d["body_names"]]
        if (
            len(names) != 19
            or len(set(names)) != 19
            or set(names) != set(H1_JOINT_NAMES)
        ):
            raise ValueError(f"{path}: expected each of the 19 H1 joints exactly once")
        if len(bodies) != len(set(bodies)) or "pelvis" not in bodies:
            raise ValueError(f"{path}: body_names must be unique and contain pelvis")
        fps = np.asarray(d["fps"]).reshape(-1)
        if fps.size != 1 or not np.isclose(fps[0], 50.0):
            raise ValueError(f"{path}: expected 50 Hz H1 motion")
        j, p, quat = d["joint_pos"], d["body_pos_w"], d["body_quat_w"]
        T = len(j)
        for key, arr, shape in (
            ("joint_pos", j, (T, 19)),
            ("body_pos_w", p, (T, len(bodies), 3)),
            ("body_quat_w", quat, (T, len(bodies), 4)),
        ):
            if T < 2 or arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"{path}: invalid {key} shape or values")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=2e-3):
            raise ValueError(f"{path}: expected unit wxyz quaternions")
        root = bodies.index("pelvis")
        return (
            p[:, root].copy(),
            quat[:, root].copy(),
            j[:, [names.index(n) for n in H1_JOINT_NAMES]].copy(),
            50.0,
        )


def _mirror(name: str) -> tuple[str, float]:
    partner = (
        name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
    )
    sign = -1.0 if name == "torso_joint" or "_yaw_" in name or "_roll_" in name else 1.0
    return partner, sign


SPEC = RobotSpec(
    name="h1",
    mjcf_path=str(MJCF_PATH),
    joint_names=H1_JOINT_NAMES,
    tracked_bodies=TRACKED_BODIES,
    default_joint_pos=dict(zip(H1_JOINT_NAMES, DEFAULT_POS)),
    action_scale={n: 0.25 for n in H1_JOINT_NAMES},
    com_body="torso_link",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    motion_dir=str(MOTION_DIR),
    motion_glob="*.npz",
    load_motion=load_h1_npz,
    anchor_body="torso_link",
    reference_position=True,
    foot_bodies=("left_ankle_link", "right_ankle_link"),
    foot_collision_geoms=("left_ankle_link_collision", "right_ankle_link_collision"),
    contact_height=0.015,
    flat_foot_height=0.0,
    joint_mirror={n: _mirror(n) for n in H1_JOINT_NAMES},
    arm_joints=H1_JOINT_NAMES[11:],
    training_rewards={
        "w_arm_joint_pos": 1.0,
        "std_arm_joint_pos": 0.5,
        "w_arm_target_limit": -0.2,
        "w_planar_drift": -0.5,
        "planar_drift_scale": 1.0,
        "w_heading": 0.5,
    },
    motion_manifest=str(REPO / "data/h1_manifest.json"),
    dr=DomainRandCfg(friction_geoms=r"^(left|right)_ankle_link_collision$"),
)
