import sys
import time

import jax
from jax import numpy as jp

sys.path.insert(0, sys.argv[1])
from bittle_stand import BittleStand

env = BittleStand(
    sys.argv[2],
    timestep=float(sys.argv[3]) if len(sys.argv) > 3 else 0.002,
    push_force=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
)
print("obs", env.observation_size, "action", env.action_size, "dt", env.dt)

reset = jax.jit(env.reset)
step = jax.jit(env.step)
s = reset(jax.random.PRNGKey(0))
print("reset obs finite:", bool(jp.isfinite(s.obs).all()), "z", float(s.pipeline_state.x.pos[0, 2]))

tot = 0.0
for i in range(100):  # 5 s, zero action = hold the balance posture
    s = step(s, jp.zeros(8))
    tot += float(s.reward)
    if float(s.done):
        print("terminated at step", i)
        break
print("5 s hold: return %.2f, upright %.3f, height %.4f, done %s" % (tot, float(s.metrics["upright"]), float(s.metrics["height"]), bool(s.done)))

# batch speed
N = 2048
rs = jax.jit(jax.vmap(env.reset))
st = jax.jit(jax.vmap(env.step))
b = rs(jax.random.split(jax.random.PRNGKey(1), N))
a = jp.zeros((N, 8))
b = st(b, a)
jax.block_until_ready(b.obs)
t0 = time.perf_counter()
alive = jp.ones(N)
for _ in range(100):  # 5 s with random pushes, zero action
    b = st(b, a)
    alive = alive * (1 - b.done)
jax.block_until_ready(b.obs)
dt = time.perf_counter() - t0
print("%d envs: %.1f ms per control step -> %.0f env steps/s (%d physics steps per control step)" % (N, dt / 100 * 1000, N * 100 / dt, env._n_frames))
print("fraction never fallen in 5 s with pushes and zero action:", float(alive.mean()), "| all finite:", bool(jp.isfinite(b.obs).all()))
