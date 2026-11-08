"""Portable asset/control/observation identity; reject incompatible checkpoints before loading."""

import hashlib
import json
import re
from dataclasses import asdict
from functools import cache
from pathlib import Path


@cache
def asset_sha256(xml_path: str) -> str:
    path = Path(xml_path)
    digest = hashlib.sha256(path.read_bytes())
    for mesh in sorted((path.parent / "meshes").rglob("*")):
        if mesh.is_file():
            digest.update(str(mesh.relative_to(path.parent)).encode())
            digest.update(mesh.read_bytes())
    return digest.hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robot_identity(robot) -> dict:
    acts = robot.robot_cfg().articulation.actuators
    controls = []
    for name in robot.joint_names:
        act = next(
            a for a in acts if any(re.fullmatch(p, name) for p in a.target_names_expr)
        )
        controls.append(
            {
                k: getattr(act, k, None)
                for k in (
                    "stiffness",
                    "damping",
                    "effort_limit",
                    "armature",
                    "viscous_damping",
                    "delay_min_lag",
                    "delay_max_lag",
                )
            }
        )
    value = {
        "schema": 1,
        "robot": robot.name,
        "asset_sha256": asset_sha256(robot.mjcf_path),
        "joint_names": robot.joint_names,
        "tracked_bodies": robot.tracked_bodies,
        "root_body": robot.root_body,
        "anchor_body": robot.anchor_name,
        "reference_position": robot.reference_position,
        "default_q": robot.default_joint_pos_list(),
        "action_scale": robot.action_scale_list(),
        "controls": controls,
        "sensing": asdict(robot.sensing),
        "foot_bodies": robot.foot_bodies,
        "foot_collision_geoms": robot.foot_collision_geoms,
        "contact_height": robot.contact_height,
        "flat_foot_height": robot.flat_foot_height,
        "control_dt": 0.02,
        "physics_dt": 0.005,
    }
    return json.loads(json.dumps(value))


def validate_checkpoint(
    ck: dict, robot, motion=None, resume: bool = False, current_args: dict | None = None
) -> None:
    args = ck.get("args", {})
    if args.get("robot", "g1") != robot.name:
        raise ValueError(
            "Checkpoint belongs to another robot; train a new policy for H1"
        )
    identity = ck.get("robot_identity")
    if identity is None:
        if robot.name != "g1":
            raise ValueError(
                "H1 checkpoints require asset/control/observation identity metadata"
            )
    elif identity != robot_identity(robot):
        raise ValueError(
            "Checkpoint asset, controls or observation definition differs from this robot"
        )
    if (
        resume
        and (robot.name != "g1" or "motion_identity" in ck)
        and ck.get("motion_identity") != motion.identity
    ):
        raise ValueError(
            "Resume requires the checkpoint's exact motion set and file contents"
        )
    if resume and robot.name == "h1" and current_args is not None:
        for name in (
            "wm_structured",
            "pi_ref_skip",
            "hist",
            "goal_frames",
            "wm_hidden",
            "enc_hidden",
            "dec_hidden",
            "wm_opt",
            "pi_opt",
        ):
            if args.get(name) != current_args.get(name):
                raise ValueError(
                    f"Cannot resume with a different {name}; use the checkpoint's model/optimizer configuration"
                )
