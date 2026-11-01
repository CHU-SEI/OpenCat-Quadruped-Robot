"""Before / after video for the stand + push-recovery task.

Left: hold the balance posture (no policy). Right: the trained PPO policy. Same start, same pushes.
Only the joint trajectories are stored from the GPU rollout; the pictures are drawn afterwards with MuJoCo (CPU, EGL).

usage: python render_compare.py <account-name-for-the-title-card> [push_force_N]
"""
import functools
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import jax  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from brax.io import model  # noqa: E402
from brax.training import types  # noqa: E402,F401
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from jax import numpy as jp  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "envs"))
from bittle_stand import BittleStand  # noqa: E402

ACCOUNT = sys.argv[1] if len(sys.argv) > 1 else "(account name)"
PUSH = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
XML = os.path.join(HERE, "..", "models", "bittle.xml")
CKPT = os.path.join(HERE, "..", "checkpoints", "stand_ppo_params")
OUT = os.path.join(HERE, "..", "videos")
os.makedirs(OUT, exist_ok=True)
EPISODE_LEN, N = 100, 1024
SLOW = 0.5  # playback speed of the episodes (0.5 = half speed)
FPS = int(round(20 * SLOW))
W, H = 640, 480
STRIP = 36  # caption strip under the two panels
HOLD_PUSH = FPS  # keep the "pushed" text for one second of video after the push
FONT = "/mnt/c/Windows/Fonts/meiryo.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


# ---------------------------------------------------------------- rollouts (GPU)
env = BittleStand(XML, timestep=0.005, push_force=PUSH)
network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=(128, 128, 64), value_hidden_layer_sizes=(128, 128, 64),
)
infer = jax.jit(ppo_networks.make_inference_fn(network)(model.load_params(CKPT), deterministic=True))
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))


def rollout(policy):
    s = reset(jax.random.split(jax.random.PRNGKey(123), N))
    q = [np.asarray(s.pipeline_state.q)]
    alive = np.ones((EPISODE_LEN + 1, N), bool)
    push = np.zeros((EPISODE_LEN, N, 2), np.float32)
    key = jax.random.PRNGKey(7)
    for t in range(EPISODE_LEN):
        key, k = jax.random.split(key)
        s = step(s, policy(s.obs, k))
        q.append(np.asarray(s.pipeline_state.q))
        alive[t + 1] = alive[t] & (np.asarray(s.done) == 0)
        push[t, :, 0] = np.asarray(s.metrics["push_fx"])
        push[t, :, 1] = np.asarray(s.metrics["push_fy"])
    return np.stack(q), alive, push


q_no, alive_no, push_no = rollout(lambda obs, k: jp.zeros((N, 8)))
q_pp, alive_pp, push_pp = rollout(lambda obs, k: infer(obs, k)[0])
assert np.allclose(push_no, push_pp), "pushes must be identical for a fair comparison"
never_no, never_pp = alive_no[-1].mean(), alive_pp[-1].mean()
print("never fell: no-op %.3f | PPO %.3f" % (never_no, never_pp))

# episodes where no-op falls and PPO stays up: earliest falls first
fall_step = np.where(alive_no.all(0), 10**6, np.argmin(alive_no, axis=0))
good = np.where(alive_pp[-1] & ~alive_no[-1])[0]
order = good[np.argsort(fall_step[good])]
chosen = order[:: max(1, len(order) // 4)][:4]
print("chosen episodes", chosen, "no-op falls at steps", fall_step[chosen])

# ---------------------------------------------------------------- drawing (CPU)
xml = open(XML).read()
xml = xml.replace(
    '<geom name="floor" type="plane" size="1 1 .1" friction="1 0.01 0.001"/>',
    '<geom name="floor" type="plane" size="1 1 .1" material="grid" friction="1 0.01 0.001"/>',
).replace(
    "<worldbody>",
    '<asset><texture name="grid" type="2d" builtin="checker" rgb1=".27 .32 .38" rgb2=".2 .24 .29" width="300" height="300"/>'
    '<material name="grid" texture="grid" texrepeat="10 10" reflectance="0.15"/>'
    '<texture type="skybox" builtin="gradient" rgb1=".55 .68 .8" rgb2=".1 .12 .16" width="64" height="64"/></asset>'
    '<visual><headlight ambient=".45 .45 .45" diffuse=".8 .8 .8"/><global offwidth="640" offheight="480"/></visual><worldbody>',
    1,
)
m = mujoco.MjModel.from_xml_string(xml)
for g in range(m.ngeom):
    name = m.geom(g).name
    if name == "torso":
        m.geom_rgba[g] = [0.95, 0.72, 0.12, 1]
    elif name.endswith("s"):
        m.geom_rgba[g] = [0.95, 0.72, 0.12, 1]
    elif name.endswith("k"):
        m.geom_rgba[g] = [0.1, 0.1, 0.12, 1]
renderer = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth, cam.elevation, cam.distance = 140, -22, 0.42


def draw_panel(d, q, push, label, fell, t, shown_push=None):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam.lookat[:] = d.qpos[:3] * np.array([1, 1, 0]) + np.array([0, 0, 0.04])
    renderer.update_scene(d, camera=cam)
    scene = renderer.scene
    if np.linalg.norm(push) > 1e-6:  # arrow showing the push on the torso
        start = d.qpos[:3] + np.array([0, 0, 0.02])
        end = start + np.array([push[0], push[1], 0]) * 0.012
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1, 0.15, 0.1, 1], dtype=np.float32))
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.004, start, end)
        scene.ngeom += 1
    img = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 44], fill=(15, 20, 25))
    dr.text((14, 6), label, font=font(26), fill=(255, 255, 255))
    if fell:
        dr.rectangle([0, H - 52, W, H], fill=(150, 20, 20))
        dr.text((14, H - 46), "転倒", font=font(30), fill=(255, 255, 255))
    elif shown_push is not None:  # the text stays up longer than the arrow
        dr.text((W - 220, H - 46), "押された！ %.1f N" % np.linalg.norm(shown_push), font=font(26), fill=(255, 120, 90))
    return img


def card(lines, secs, bg=(18, 28, 32)):
    img = Image.new("RGB", (2 * W, H + STRIP), bg)
    dr = ImageDraw.Draw(img)
    y = 40
    for text, size, color in lines:
        dr.text((60, y), text, font=font(size), fill=color)
        y += int(size * 1.7)
    return [np.asarray(img)] * int(secs * FPS)


WHITE, GREY, ACC = (255, 255, 255), (190, 200, 205), (120, 220, 200)
frames = []
frames += card([
    ("Bittle 四足ロボットの「押されても倒れない」制御", 44, WHITE),
    ("強化学習（PPO）× MuJoCo MJX（GPU 並列シミュレーション）", 28, ACC),
    ("", 20, WHITE),
    ("Physical AI 応用1講座 最終課題", 26, GREY),
    ("PAI最終課題_%s" % ACCOUNT, 26, GREY),
    ("※ 映像はシミュレーションです。再生速度は %.1f 倍速（スローモーション）" % SLOW, 22, GREY),
], 6)
frames += card([
    ("やったこと", 38, ACC),
    ("・実機 Bittle（NyBoard V1.0, 290 g）の寸法・質量・関節範囲をモデル化", 26, WHITE),
    ("・実機の通信を計測し、制御周期を 20 Hz に設定（往復 約 44 ms）", 26, WHITE),
    ("・観測は実機で取れる値だけ：IMU の姿勢・角速度、関節角、前回の行動", 26, WHITE),
    ("・2048 台を GPU で並列に動かし、PPO で 2000 万ステップ（約 7 分）学習", 26, WHITE),
    ("・学習中に、胴体へランダムな水平の力（各方向 最大 %.0f N、体重の約 2 倍）を 50 ms 加える" % PUSH, 26, WHITE),
], 10)

for ep, i in enumerate(chosen):
    last_push, last_t = None, -10**6
    for t in range(EPISODE_LEN + 1):
        p = push_no[t - 1, i] if t >= 1 else np.zeros(2)
        if np.linalg.norm(p) > 1e-6:
            last_push, last_t = p, t
        shown = last_push if t - last_t < HOLD_PUSH else None
        a = draw_panel(mujoco.MjData(m), q_no[t, i], p, "学習前（何もしない）", not alive_no[t, i], t, shown)
        b = draw_panel(mujoco.MjData(m), q_pp[t, i], p, "学習後（PPO 方策）", not alive_pp[t, i], t, shown)
        canvas = Image.new("RGB", (2 * W, H + STRIP), (15, 20, 25))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (W, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((W - 90, 6), "%.2f s" % (t * 0.05), font=font(24), fill=(255, 230, 120))
        dr.text((14, H + 6), "例 %d / %d　再生 %.1f 倍速　学習前が転んだ試行から選んだ例です（全体の割合は最後に表示）" % (ep + 1, len(chosen), SLOW),
                font=font(20), fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    frames += [frames[-1]] * FPS  # hold the last picture for one second

frames += card([
    ("結果（%d 回の試行、各 5 秒、同じ押し方）" % N, 36, ACC),
    ("5 秒間、転ばなかった割合", 28, GREY),
    ("   学習前（何もしない）：%.0f %%" % (100 * never_no), 40, (255, 140, 120)),
    ("   学習後（PPO 方策）  ：%.0f %%" % (100 * never_pp), 40, (140, 240, 170)),
    ("", 16, WHITE),
    ("今後：実機 Bittle への展開（シミュレーションと実機の差の評価）、転倒復帰、荷物運搬への適応", 22, GREY),
], 9)

path = os.path.join(OUT, "stand_before_after.mp4")
imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
print("wrote", os.path.abspath(path), "| %d frames | %.1f s" % (len(frames), len(frames) / FPS))
