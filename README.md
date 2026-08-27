# Contrast Lab

PyTorchで教師あり対照学習を条件統制して比較するための実験基盤です。中核のViT、投影ヘッド、損失、拡張、GradCacheは自前実装し、設定検証にPydantic、Schedule-Free optimizer、dashboard APIにHonoとBunを使います。

## 比較の前提

既定値はCIFAR-100、ViT-Tiny/4相当（dim 192、12 blocks、3 heads）、LayerNorm、GELU、PyTorch SDPAです。global source batch、view数、augmentation、optimizer、precisionを固定し、`objective`だけを差し替えられます。

実装済みの損失は以下です。

- Cross-Entropy
- NT-Xent（同一画像の別viewをpositiveとする）
- Supervised Contrastive Loss
- SINCERE
- Sigmoid-SupCon（SigLIP型のpairwise sigmoidを、同一クラスをpositiveとする多positive設定へ拡張）
- Cross-Entropy + SupCon

Sigmoid-SupConは画像・テキストの二塔SigLIPそのものではありません。ほかの画像単塔損失とモデル条件を揃えるため、SigLIPのpairwise sigmoid設計を教師ありpositive maskへ適用しています。複数positiveへの拡張では、各anchorのvalid pair loss和をpositive数で割ってからanchor平均します。これは[公式SigLIP実装のpositive数による正規化方針](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py#L287-L308)を複数positiveへ一般化したものです。SINCEREも論文の式に従い、positive pair全体ではなく各anchorのpositive平均を等しく平均します。

## Dataset

CIFAR-100は[`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100)から取得します。`configs/base.toml`ではrepository revisionをcommit SHAへ固定しており、別配布元への自動fallbackは行いません。HubのParquetを初回だけuint8 tensorへ変換し、`data/processed`にcacheします。

## Setup

Python 3.12とBunを使います。

```bash
uv sync
cd web
bun install
bun run build
cd ..
```

設定の解決結果だけを確認できます。

```bash
uv run contrast validate -c configs/objectives/sigmoid_supcon.toml
```

短い配線確認と本学習の例です。

```bash
uv run contrast train -c configs/experiments/smoke.toml
uv run contrast train -c configs/objectives/sigmoid_supcon.toml --set run.seed=1
```

本番前に、5損失をseed 0で各1,000 optimizer stepだけ動かすpilotを実行します。lossの有限性、gradient clipping率、throughput、Sigmoidのscale/biasをDashboardで確認します。

```bash
uv run contrast sweep configs/sweeps/core_losses_pilot.toml --dry-run
uv run contrast sweep configs/sweeps/core_losses_pilot.toml
```

pilot確認後、損失5種 × seed 3種の本sweepを実行します。sweepは起動前に全組合せを設定検証し、seedごとに5損失を順番に実行します。結果は既存runと分離した`runs/cifar100-core-v3/`へ保存されます。各runは120 epochで、400 epochの初回sweepで観測された80〜120 epoch付近の性能ピークを含みます。

```bash
uv run contrast sweep configs/sweeps/core_losses.toml --dry-run
uv run contrast sweep configs/sweeps/core_losses.toml
```

## PrecisionとGradCache

既定はFP32 parameter、BF16 autocast、FP32 loss、TF32許可です。TF32自体は主に速度向上の設定で、activation memory削減はBF16 autocastが担います。同じGPU・ソフトウェア条件でseed、data order、viewごとのaugmentationを固定します。より強い決定性が必要なら`reproducibility.mode = "strict"`を指定できます。

GradCacheはlogical batch全体の表現から一度だけ損失を計算し、chunkごとにforwardを再実行します。optimizer stepはlogical batchにつき1回です。`tests/test_grad_cache.py`でDirectとの勾配一致を検証します。現時点のGradCacheはFP32/BF16を対象とし、FP16 GradScalerは明示的に拒否します。今回のViT-Tiny・logical batch 256 sourceはRTX 4070 Ti SUPER上でDirect実行が約2.8 GiBに収まり、GradCache 128 source chunkより約21%高速だったため、本sweepの既定は`step_strategy = "direct"`です。SupConの短時間計測ではbatch 512/1024へ増やしてもsource throughputは向上せず、1024は約11.8 GiBを使用しました。pair行列とbatch内positive/negative数も変わるため、core sweepはbatch 256を維持します。より大きなbatch/modelでOOMする場合は`"grad_cache"`へ切り替え、`batch.grad_cache_chunk_size_per_rank`を調整します。

Schedule-Freeでは学習時と評価時のparameter viewが異なるため、評価とcheckpoint保存を必ずoptimizerのeval mode内で行います。optimizerの`weight_decay_policy = "standard"`はLinear/Conv等の行列weightだけをdecayし、bias、Norm、class token、position embedding、Sigmoid lossのscalar parameterを除外します。旧single-group optimizer checkpointはresume時に新しいgroupへ移行します。非有限lossまたはgradientはそのstepで即座に例外にします。

`optimization/lr`はSchedule-Freeの基準`lr`ではなく、linear warmupを反映してoptimizerが公開する`scheduled_lr`を記録します。通常のAdamWではscheduler適用後の`lr`を記録します。

本sweepはraw/EMA評価と途中checkpointを20 epochごとに実行します。`eval/backbone_knn_top1`が改善した評価時点は`best.pt`へ原子的に上書き保存し、最終epochは`final.pt`へ保存します。gradient clipは10.0とし、初期の大きな勾配を抑えつつ定常時の常時clipを避けます。clip前後のglobal norm、clip係数、model/objective別norm、CUDA allocated/reserved memoryを記録します。

## EMA

モデルparameterのEMA（Exponential Moving Average）を任意で保持できます。更新式は
`ema = decay * ema + (1 - decay) * model`で、最初の更新だけは学習中モデルをそのままcopyします。EMAはSchedule-Free optimizerが持つ評価用parameter viewとは独立したshadow modelです。Schedule-Free使用時の`raw`評価は従来どおりoptimizerのeval viewを使い、`ema`評価はoptimizer step後の学習中モデルから更新したEMAを使います。そのため、両者を同じrunで比較できます。

共通設定は次のとおりです。`start_step`と`update_every_steps`はoptimizer step単位で、decay scheduleの経過も更新回数ではなくoptimizer stepで数えます。`buffer_mode = "copy"`はbufferを更新ごとにcopyし、`"ema"`は浮動小数・複素数bufferにもEMAを適用します（整数bufferは常にcopy）。ViTのLayerNormはrunning statisticsを持ちませんが、将来BatchNormを比較するときにも同じ設定を利用できます。

```toml
[ema]
enabled = true
start_step = 0
update_every_steps = 1
buffer_mode = "copy" # "copy" | "ema"
evaluation_weights = "both" # "raw" | "ema" | "both"
```

decay scheduleは次の4種類から一つを選びます。`linear`と`cosine`は`start_decay`から`end_decay`まで`schedule_steps`で補間し、それ以降は`end_decay`を維持します。

```toml
[ema.decay]
kind = "constant"
decay = 0.999
```

```toml
[ema.decay]
kind = "linear"
start_decay = 0.9
end_decay = 0.9999
schedule_steps = 10000
```

```toml
[ema.decay]
kind = "cosine"
start_decay = 0.9
end_decay = 0.9999
schedule_steps = 10000
```

```toml
[ema.decay]
kind = "inverse_power"
min_decay = 0.0
max_decay = 0.9999
inv_gamma = 1.0
power = 0.6666666666666666
```

`evaluation_weights`は通常モデルだけを評価する`"raw"`、EMAだけを評価する`"ema"`、両方を評価する`"both"`から選びます。通常モデルの指標は`eval/*`と`test/*`、EMA指標は`eval_ema/*`と`test_ema/*`へ分けて記録されます。checkpointには通常モデルとEMA shadow、update回数、最新decayが保存され、resume後もschedule stateを復元します。`contrast evaluate --checkpoint ...`もcheckpointからEMAを復元し、保存済みconfigの`evaluation_weights`に従ってoffline評価します。EMAを有効にするとshadow model一つ分のparameter memoryが追加で必要です。

## 評価プロトコル

`run.seed`はモデル初期化・data order・augmentationを制御し、train/validation分割は独立した`data.split_seed`で固定します。これにより、seed sweepで評価画像そのものが変わる交絡を避けます。

CIFAR-100はstratified split後もtrainが各class 450枚で均衡しているため、train samplerはreplacementなしのepoch permutationを使います。`WeightedRandomSampler`は同一画像の重複・epoch内未使用画像を生み、batch内のclass均衡も保証しないため既定にはしません。class-balanced batchはpositive/negative構成そのものを変えるので、必要なら別sampler ablationとして扱います。

学習中は20 epochごとに次の2種類のk-NNを記録します。どちらも比較前にL2 normalizeします。

- `eval/backbone_knn_top1`: encoder feature上のk-NN。全objectiveで同じ意味を持つ主要指標
- `eval/projector_knn_top1`: projection head出力上のk-NN。損失が直接最適化する空間の診断指標

同じ固定validation表現について、backbone/projectorごとにcollapseと次元冗長性の診断も記録します。`std_mean`はL2 normalize後の次元別標準偏差の平均、`isotropy`はそれを単位球上の最大値`1/sqrt(d)`で正規化した値です。`effective_rank_ratio`は中心化共分散のentropy rankを表現次元で割った値で、`offdiag_correlation_rms`は分散を持つ次元間の非対角相関、`mean_pairwise_cosine`は異なるサンプル全組の平均cosineです。前3者が0へ、またはpairwise cosineが1へ近づく挙動はcollapseを示します。EMA版は`eval_ema/*`に分けて保存します。

最終epochではencoderをeval modeで凍結し、augmentationなしのtrain featureを一度だけ抽出します。その固定feature上で共通の`nn.Linear`をSGD + cosine decayで学習し、`eval/linear_probe_top1`を記録します。encoderやprojectorへ勾配は流れず、probeのseed・epoch・batch size・optimizer条件は`[evaluation.linear_probe]`で全run共通です。core sweepでは`test_at_end = false`として、checkpoint選択中にtest splitを参照しません。

`eval/joint_classifier_top1`は補助診断です。CE/CE+SupConでは学習されますが、対照損失単独ではclassifier headがobjectiveに含まれないためchance accuracy付近になるのが正常です。手法間の主要比較にはbackbone k-NNとfrozen linear probeを使います。
既存checkpointにも同じ評価を後付けできます。checkpointが元runの`checkpoints/`内にあればrun directoryは自動推定され、結果はそのrunの`metrics.jsonl`へ追記されます。移動したcheckpointには`--run-dir`を指定します。GPUを学習runと共有するため、同じGPUでの学習中ではなく停止後または完了後に実行してください。

```bash
uv run contrast evaluate \
  --checkpoint runs/cifar100-core-v3/<run>/checkpoints/best.pt \
  --queries test
```

`--queries`は`config`（保存済み設定に従う）、`eval`、`test`、`both`を選べます。3 seedのvalidation結果からcheckpointと手法を確定した後に、選択済み`best.pt`へ`--queries test`を一度だけ実行します。

## Dashboard

学習メトリクスは各runの`metrics.jsonl`が正本です。Web側のHono APIがrunファイルを直接読むため、Python serverは不要です。UI開発時はViteからAPIと画面を同時に起動します。

```bash
cd web
bun run dev
```

[http://127.0.0.1:5173](http://127.0.0.1:5173) で確認できます。本番相当ではReact UIをbuildし、Bun/HonoからAPIと静的ファイルを同じポートで配信します。

```bash
bun run build
bun run serve
```

[http://127.0.0.1:8000](http://127.0.0.1:8000) で最大6 runを選び、loss、通常モデルとEMAのbackbone/projector k-NN・frozen linear probe、EMA decay・update回数、gradient、Sigmoidパラメータなどを同時に比較できます。既存runの旧指標もlegacy cardに残ります。

既定ではrepository直下の`runs/`を読みます。変更する場合は`bun run serve --runs-dir /path/to/runs`を指定し、Viteでは`CONTRAST_RUNS_DIR=/path/to/runs bun run dev`を使います。

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
cd web
bun run format:check
bun run lint
bun run typecheck
bun test
bun run build
```

## Distributed extension

`RuntimeContext`、global batch sampler、rank情報はtorchrun/DDPを見据えた境界です。ただし、対照損失に必要な全rank表現のgradient-aware gatherは未実装なので、現バージョンはworld size 1を明示的に要求します。誤ったlocal-negative実験を静かに実行しないための制約です。
