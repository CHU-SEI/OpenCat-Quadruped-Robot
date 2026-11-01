# Bittle × Physical AI

Petoi Bittle（NyBoard V1.0）の「押されても倒れない」制御を、強化学習（PPO）で学習するプロジェクトです。
Physical AI 応用1講座の最終課題として作成しました。

- 物理シミュレーション: MuJoCo / MJX（GPU 並列）
- 学習: Brax の PPO（2048 環境並列、2000 万ステップ、約 7 分）
- 実機: Petoi Bittle。ファームウェアは OpenCat（MIT License）を一部変更して使用

## 結果（シミュレーション）

胴体にランダムな水平の力（各方向 最大 6 N、50 ms、体重の約 2 倍）を加えた 1024 回の試行（各 5 秒）で、転ばなかった割合は次のとおりです。

| | 転ばなかった割合 |
|---|---|
| 何もしない（`kbalance` の姿勢を保つだけ） | 32.5 % |
| 学習した PPO 方策 | 60.2 % |

（`scripts/train_stand.py` の評価。比較動画 `render_compare.py` も、同じ乱数で評価しています。）

方策は、まだ実機では試していません。サーボの特性（トルク・遅れ）は仮の値で、シミュレーションと実機の差は評価できていません。

## 構成

```
physicalAI/
├── measure_latency.py   PC と Bittle の通信周期・IMU 読み出し周期の計測
├── joint_check.py       関節番号 8〜15 がどの脚か確認する（L コマンドで 1 関節ずつ動かす）
├── joint_check.md       確認結果の表
└── sim/
    ├── models/bittle.xml        MuJoCo モデル（実機 290 g、関節範囲はファームウェアの angleLimit）
    ├── envs/bittle_stand.py     Brax 環境（立位保持・押されても倒れない）
    ├── scripts/train_stand.py   学習と評価
    ├── scripts/render_compare.py 比較動画の作成
    ├── scripts/env_smoke.py     環境の動作確認・速度測定
    └── checkpoints/             学習済みの方策（Brax 形式）
```

## 実機の計測（PC ↔ NyBoard、USB、115200 baud）

| 項目 | 平均 |
|---|---|
| IMU の読み出し（`v`） | 16.7 ms |
| 関節の読み出し（`j`） | 23.3 ms |
| 関節指令の往復（`L`） | 27.8 ms |
| 閉ループ（`L` + `v`） | 44 ms（約 23 Hz） |

この結果から、制御周期を 20 Hz（50 ms）にしました。
ファームウェアは、バッファにある全バイトを 1 つのコマンドとして読み込むため、**コマンドは必ず echo を待ってから次を送る**必要があります（`L` の直後に `v` を送ると、`v` が失われます）。

## ファームウェアの変更（OpenCat）

`L` コマンド（16 関節を一括指定）の補間をなくし、目標角度をそのまま書き込むようにしました。

- `src/reaction.h`: `T_LISTED_BIN` の処理で、`DIRECT_LISTED_BIN` が定義されているときは `transform(..., 1, 0)` を呼ぶ
- `OpenCat.ino`: `#define DIRECT_LISTED_BIN`

閉ループの周期は、動かしたとき 63 ms（16 Hz）から 44 ms（23 Hz）になりました。

## モデルと実機の対応

- ファームウェアの関節番号: 8＝左前、9＝右前、10＝右後、11＝左後の肩、12〜15＝同じ順の膝（実機で確認）
- 関節角（rad）＝ファームウェアの角度（度）× π/180（8〜15 すべて同じ）。正の角度で、ももとすねが後ろへ振れる
- `kcalib`（全関節 0°）: ももが真下、すねが前向きに水平
- 質量は合計 0.290 kg（バッテリー込みの実測）。胴体・もも・すねへの配分は仮の値
- 脚の長さ: 肩の軸から膝の軸まで 46.6 mm、膝の軸から足先まで 51.5 mm + 半径 5 mm

## 再現手順

動作確認した環境: Windows 11 + WSL2（Ubuntu 26.04）、Python 3.14、RTX 4070 Ti（12 GB）。

```bash
cd physicalAI/sim
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -r ../requirements.txt

python scripts/env_smoke.py envs models/bittle.xml 0.005 6   # 環境の確認
python scripts/train_stand.py 20000000                       # 学習（約 7 分）
python scripts/render_compare.py <account-name>              # 比較動画 → videos/
```

実機の計測（Windows 側）:

```powershell
pip install pyserial
python physicalAI\measure_latency.py COM4
python physicalAI\joint_check.py COM4
```

## 既知の制限

- Brax 0.14.2 は JAX 0.12 と組み合わせると、そのままでは動きません（`jax.device_put_replicated` が削除されたため）。`train_stand.py` の冒頭で、同じ働きの関数を差し込んで回避しています。Brax は、現在は積極的に保守されていません。
- サーボのトルク・剛性・遅れ、質量の配分は仮の値です。サーボには位置のフィードバックがないため、実機での同定は、動画や IMU からの推定になります。
- 物理の刻みは 5 ms です（2 ms より粗いが、約 4 倍速く、押す力の試験で安定していました）。

## ライセンス・謝辞

- OpenCat（Petoi）: MIT License。このリポジトリの `physicalAI/` 以外のファームウェアは、OpenCat のものです。
- コードの作成には、AI アシスタント（Claude Code）を使用しました。

