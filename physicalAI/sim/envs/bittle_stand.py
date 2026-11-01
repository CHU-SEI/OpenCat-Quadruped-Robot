"""Bittle standing / push-recovery task for Brax (MJX backend).

Action: 8 offsets in [-1, 1], scaled to ACTION_SCALE rad around the kbalance posture, sent as
position targets (same meaning as the firmware 'L' command, joint order lfs lfk rfs rfk lbs lbk rbs rbk).
Control rate: 20 Hz (25 physics steps of 2 ms). Observation uses only what the real robot can measure:
IMU (gravity direction, angular velocity), joint targets as read back from the firmware, previous action.
"""
import jax
import mujoco
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp

BALANCE = 0.5236  # 30 deg, firmware kbalance for all 8 joints
ACTION_SCALE = 0.35  # rad (about 20 deg)
N_FRAMES = 25  # 25 x 2 ms = 50 ms


class BittleStand(PipelineEnv):
    def __init__(self, xml_path, push_prob=0.02, push_force=1.0, timestep=0.002, **kwargs):
        sys = mjcf.load(xml_path)
        sys = sys.tree_replace({
            "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
            "opt.iterations": 4,
            "opt.ls_iterations": 8,
            "opt.timestep": timestep,
        })
        kwargs["n_frames"] = kwargs.get("n_frames", int(round(0.05 / timestep)))  # 50 ms per control step
        super().__init__(sys=sys, backend="mjx", **kwargs)
        self._default_q = jp.array(sys.init_q)  # keyframe 0 is not used: set below
        self._default_joints = jp.full(8, BALANCE)
        self._init_q = jp.concatenate([jp.array([0.0, 0.0, 0.0886, 1.0, 0.0, 0.0, 0.0]), self._default_joints])
        self._lo = jp.array(sys.mj_model.jnt_range[1:, 0])
        self._hi = jp.array(sys.mj_model.jnt_range[1:, 1])
        self._push_prob = push_prob
        self._push_force = push_force
        self._nbody = sys.mj_model.nbody

    # --- helpers -------------------------------------------------------------
    def _obs(self, ps, prev_action):
        rot = ps.x.rot[0]
        gravity = math.inv_rotate(jp.array([0.0, 0.0, -1.0]), rot)
        gyro = math.inv_rotate(ps.xd.ang[0], rot)
        joints = ps.q[7:] - self._default_joints
        jvel = ps.qd[6:] * 0.1
        return jp.concatenate([gravity, gyro, joints, jvel, prev_action])

    # --- env API ---------------------------------------------------------------
    def reset(self, rng):
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        q = self._init_q.at[7:].add(jax.random.uniform(k1, (8,), minval=-0.15, maxval=0.15))
        # small random tilt and yaw
        axis_noise = jax.random.uniform(k2, (3,), minval=-0.08, maxval=0.08)
        quat = math.normalize(jp.array([1.0, axis_noise[0], axis_noise[1], axis_noise[2]]))[0]
        q = q.at[3:7].set(quat)
        qd = jax.random.normal(k3, (14,)) * 0.02
        ps = self.pipeline_init(q, qd)
        prev = jp.zeros(8)
        obs = self._obs(ps, prev)
        zero = jp.zeros(())
        metrics = {"upright": zero, "height": zero, "push": zero, "push_fx": zero, "push_fy": zero}
        return State(ps, obs, zero, zero, metrics, {"rng": rng, "prev_action": prev})

    def step(self, state, action):
        rng, k1, k2, k3 = jax.random.split(state.info["rng"], 4)
        action = jp.clip(action, -1.0, 1.0)
        target = jp.clip(self._default_joints + ACTION_SCALE * action, self._lo, self._hi)

        # random horizontal push on the torso for this control step
        pushed = jax.random.uniform(k1) < self._push_prob
        direction = jax.random.uniform(k2, (2,), minval=-1.0, maxval=1.0)
        force = jp.where(pushed, self._push_force * direction, jp.zeros(2))
        xfrc = jp.zeros((self._nbody, 6)).at[1, :2].set(force)
        ps0 = state.pipeline_state.tree_replace({"xfrc_applied": xfrc})

        ps = self.pipeline_step(ps0, target)

        rot = ps.x.rot[0]
        up = math.rotate(jp.array([0.0, 0.0, 1.0]), rot)
        upright = up[2]
        height = ps.x.pos[0, 2]
        smooth = jp.sum(jp.square(action - state.info["prev_action"]))
        effort = jp.sum(jp.square(ps.qd[6:])) * 1e-3

        reward = (
            1.0 * upright
            - 20.0 * jp.square(height - 0.0886)
            - 0.05 * smooth
            - effort
            - 0.1 * jp.sum(jp.square(jp.clip(ps.q[7:] - self._default_joints, -2, 2))) * 0.1
        )
        done = jp.where((upright < 0.5) | (height < 0.03), 1.0, 0.0)

        obs = self._obs(ps, action)
        state.metrics.update(
            upright=upright, height=height, push=pushed.astype(jp.float32), push_fx=force[0], push_fy=force[1]
        )
        info = {**state.info, "rng": rng, "prev_action": action}  # keep the keys the Brax wrappers add
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, info=info)

    @property
    def action_size(self):
        return 8

    @property
    def observation_size(self):
        return 30
