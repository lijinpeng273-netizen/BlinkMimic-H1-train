"""Differentiable forward kinematics of the robot in torch, built from its MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import RobotSpec
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    def __init__(self, robot: RobotSpec, device: str):
        self.robot = robot
        m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.device = device
        tracked = robot.tracked_bodies
        jname_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.anchor_idx = tracked.index(robot.anchor_name)
        if bodies[0] != robot.root_body:
            raise ValueError("The physical root must be the first moving MJCF body")
        self.feet = [tracked.index(n) for n in robot.foot_bodies]
        points = []
        if robot.foot_collision_geoms and len(robot.foot_collision_geoms) != len(self.feet):
            raise ValueError("Each foot needs one collision geom")
        for i, body in enumerate(robot.foot_bodies):
            if robot.foot_collision_geoms:
                geom = m.geom(robot.foot_collision_geoms[i]).id
                if m.body(m.geom_bodyid[geom]).name != body or m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError("Foot collision geometry must be a mesh attached to its foot body")
                mesh = m.geom_dataid[geom]
                start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, m.geom_quat[geom])
                vertices = m.mesh_vert[start:start + count] @ rot.reshape(3, 3).T + m.geom_pos[geom]
                lo, hi = vertices.min(0), vertices.max(0)
                points.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
            else:
                points.append(np.zeros((8, 3)))  # preserve G1's link-origin contact convention
        self.foot_points = torch.tensor(np.array(points).reshape(len(points), 8, 3), dtype=torch.float32, device=device)
        self.num_bodies = len(bodies)
        # joint limits (in joint_names order)
        lo, hi = np.zeros(robot.num_joints), np.zeros(robot.num_joints)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def foot_kinematics(self, bp, bR, bv, bw):
        """Lowest bounding-box corner height and its link-origin-corrected velocity.

        H1 contact/penetration are geometric proxies using eight mesh bounds corners,
        avoiding thousands of mesh vertices in every differentiable rollout step.
        G1 keeps the published ankle-origin convention.
        """
        offset = mv3(bR[..., self.feet, :, :].unsqueeze(-3), self.foot_points)
        positions = bp[..., self.feet, :].unsqueeze(-2) + offset
        velocities = bv[..., self.feet, :].unsqueeze(-2) + torch.cross(bw[..., self.feet, :].unsqueeze(-2).expand_as(offset), offset, dim=-1)
        z, index = positions[..., 2].min(-1)
        velocity = velocities.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
        return z, velocity

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,J) in joint_names order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = mm3(Rp, self.off_R[i])
            Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], q[..., j]))
            axis_w = mv3(R_off, self.axes[i])
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
