"""Train the Bittle stand / push-recovery task with Brax PPO on the GPU, then compare with a no-op baseline.

usage: python train_stand.py <num_timesteps> [push_force_N]
"""
import functools
import os
import sys
import time

import jax
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax import numpy as jp



def _device_put_replicated(tree, devices):
    """Brax 0.14.2 calls jax.device_put_replicated, which JAX 0.12 removed. Stack one copy per device, as it did."""
    stacked = jax.tree.map(lambda a: jp.stack([jp.asarray(a)] * len(devices)), tree)
    return jax.device_put(stacked, devices[0]) if len(devices) == 1 else jax.device_put(stacked)


jax.device_put_replicated = _device_put_replicated

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "envs"))
from bittle_stand import BittleStand  # noqa: E402

XML = os.path.join(HERE, "..", "models", "bittle.xml")
NUM_TIMESTEPS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000
PUSH = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
EPISODE_LEN = 100  # 5 s at 20 Hz

env = BittleStand(XML, timestep=0.005, push_force=PUSH)
t0 = time.time()


def progress(step, metrics):
    print("%9d steps | eval reward %.2f | eval length %.1f | %.0f s" % (
        step, float(metrics["eval/episode_reward"]), float(metrics["eval/avg_episode_length"]), time.time() - t0), flush=True)


make_inference_fn, params, _ = ppo.train(
    environment=env,
    num_timesteps=NUM_TIMESTEPS,
    num_evals=8,
    episode_length=EPISODE_LEN,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=8,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=256,
    network_factory=functools.partial(
        ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 64), value_hidden_layer_sizes=(128, 128, 64)
    ),
    seed=0,
    progress_fn=progress,
)
os.makedirs(os.path.join(HERE, "..", "checkpoints"), exist_ok=True)
ckpt = os.path.join(HERE, "..", "checkpoints", "stand_ppo_params")
model.save_params(ckpt, params)
print("saved", os.path.abspath(ckpt), "| train time %.0f s" % (time.time() - t0))

# --- evaluation: same pushes, same start states ---
N = 1024
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
infer = jax.jit(make_inference_fn(params, deterministic=True))


def run(policy):
    s = reset(jax.random.split(jax.random.PRNGKey(123), N))
    alive = jp.ones(N)
    ret = jp.zeros(N)
    key = jax.random.PRNGKey(7)
    for _ in range(EPISODE_LEN):
        key, k = jax.random.split(key)
        a = policy(s.obs, k)
        s = step(s, a)
        ret = ret + s.reward * alive
        alive = alive * (1 - s.done)
    return float(alive.mean()), float(ret.mean())


noop = lambda obs, k: jp.zeros((N, 8))
trained = lambda obs, k: infer(obs, k)[0]
print("push force %.1f N, %d episodes of %d s" % (PUSH, N, EPISODE_LEN * 0.05))
print("no-op (hold the balance posture): never fell %.3f | mean return %.1f" % run(noop))
print("trained PPO policy:               never fell %.3f | mean return %.1f" % run(trained))
