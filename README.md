# Contrast Lab

PyTorchで教師あり・自己教師あり表現学習を条件統制して比較するための実験基盤です。中核のViT、投影ヘッド、損失、拡張、GradCacheは自前実装し、設定検証にPydantic、Schedule-Free optimizer、実験追跡にWeights & Biases（W&B）を使います。

## 比較の前提

既定値はCIFAR-100、ViT-Tiny/4相当（dim 192、12 blocks、3 heads）、LayerNorm、GELU、PyTorch SDPAです。global source batch、view数、augmentation、optimizer、precisionを固定し、`objective`だけを差し替えられます。

実装済みの手法は以下です。

- 分類・proxy: Cross-Entropy、Normalized Softmax、CosFace、ArcFace、Proxy Anchor
- 教師ありpair: SupCon、SINCERE、Sigmoid-SupCon、Circle Loss、Batch-hard Triplet、Multi-Similarity
- 混合: Cross-Entropy + SupCon
- 自己教師あり: NT-Xent、Barlow Twins、BYOL、MoCo

BYOLはonline predictorとcosine schedule付きmomentum target、MoCoはmomentum key encoderとcheckpoint可能なFIFO queueを持ちます。Barlow Twinsはraw projector出力のcross-correlationを使います。比較対象のViTと2層projectorは全手法で固定し、原論文固有のBatchNorm付きheadへは置換しません。これによりoptimizer、augmentation、parameter budgetの差を抑えますが、原論文architectureの完全再現ではなく、各手法の中核的なobjective/state機構を共通architecture上で比較する実験です。自己教師あり手法はlabelを使わないため、教師あり手法とは別familyとして集計します。

Sigmoid-SupConは画像・テキストの二塔SigLIPそのものではありません。ほかの画像単塔損失とモデル条件を揃えるため、SigLIPのpairwise sigmoid設計を教師ありpositive maskへ適用しています。複数positiveへの拡張では、各anchorのvalid pair loss和をpositive数で割ってからanchor平均します。これは[公式SigLIP実装のpositive数による正規化方針](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py#L287-L308)を複数positiveへ一般化したものです。SINCEREも論文の式に従い、positive pair全体ではなく各anchorのpositive平均を等しく平均します。

## Dataset

CIFAR-100は[`uoft-cs/cifar100`](https://huggingface.co/datasets/uoft-cs/cifar100)から取得します。`configs/base.toml`ではrepository revisionをcommit SHAへ固定しており、別配布元への自動fallbackは行いません。HubのParquetを初回だけuint8 tensorへ変換し、`data/processed`にcacheします。

## Setup

Python 3.12を使います。依存関係を同期したら、初回だけW&Bへログインします。

```bash
uv sync
uv run wandb login
```

追跡先は設定ファイルの`[tracking]`で指定します。`entity`は個人アカウントの既定entityを使う場合は省略できます。

```toml
[tracking]
project = "contrast-lab"
# entity = "your-team"
mode = "online" # "online" | "offline" | "disabled"
```

`online`は学習中にW&Bへ送信し、`offline`はSDKのローカルデータへ記録して後から`wandb sync`できるようにします。テストや追跡不要の実行では`disabled`を指定できます。

設定の解決結果だけを確認できます。

```bash
uv run contrast validate -c configs/objectives/sigmoid_supcon.toml
```

短い配線確認と本学習の例です。

```bash
uv run contrast train -c configs/experiments/smoke.toml
uv run contrast train -c configs/objectives/sigmoid_supcon.toml --set run.seed=1
```

全16手法の比較は`run_train.sh`から実行します。既定はseed 0・各1,000 optimizer stepのpilotです。lossの有限性、gradient clipping率、throughput、表現collapse、proxy、BYOL target decay、MoCo queueをW&Bで確認します。

```bash
./run_train.sh pilot --dry-run
./run_train.sh pilot
```

pilot終了後は、test splitを参照せずに全16個の`final.pt`へvalidation k-NNと
表現collapse診断を追加できます。raw/EMAの両方を評価し、短時間診断では
linear probeを省きます。結果は各runのW&B履歴へ追記されます。

```bash
./run_eval.sh all-methods-pilot
```

pilotでclip前global normが10を超えた手法だけを対象に、閾値100で6 epoch
（1,050 step）を再確認する監査sweepも用意しています。最終epochに
validation k-NNとcollapse診断をraw/EMAの両方で実行し、linear probeは省きます。

```bash
./run_train.sh clip-audit --dry-run
./run_train.sh clip-audit
```

pilot確認後、16手法 × 3 seedの本sweepを実行します。sweepは起動前に48組すべてを設定検証し、seedごとに16手法を順番に実行します。結果は`runs/cifar100-all-methods-v1/`へ保存されます。各runは120 epochで、validationだけを使って手法ごとのcheckpointを選びます。

```bash
./run_train.sh full --dry-run
./run_train.sh full
```

各組合せにはdry-runに表示される1始まりのindexが付きます。長いsweepを分割または中断位置から再開する場合は範囲を指定できます。

```bash
./run_train.sh full --start-index 17 --end-index 32
./run_train.sh full --start-index 33
```

従来の5損失だけを再実行する`configs/sweeps/core_losses*.toml`も残しています。

## PrecisionとGradCache

既定はFP32 parameter、BF16 autocast、FP32 loss、TF32許可です。TF32自体は主に速度向上の設定で、activation memory削減はBF16 autocastが担います。同じGPU・ソフトウェア条件でseed、data order、viewごとのaugmentationを固定します。より強い決定性が必要なら`reproducibility.mode = "strict"`を指定できます。

GradCacheはlogical batch全体の表現から一度だけ損失を計算し、chunkごとにforwardを再実行します。optimizer stepはlogical batchにつき1回です。通常のpair/proxy lossに加え、Barlow Twinsのraw projector出力、BYOLのdetached target、MoCoのkeyとqueue一回更新についてもDirectとの勾配・state一致をテストします。現時点のGradCacheはFP32/BF16を対象とし、FP16 GradScalerは明示的に拒否します。今回のViT-Tiny・logical batch 256 sourceはRTX 4070 Ti SUPER上でDirect実行が約2.8 GiBに収まり、GradCache 128 source chunkより約21%高速だったため、本sweepの既定は`step_strategy = "direct"`です。SupConの短時間計測ではbatch 512/1024へ増やしてもsource throughputは向上せず、1024は約11.8 GiBを使用しました。pair行列とbatch内positive/negative数も変わるため、全手法でbatch 256を維持します。より大きなbatch/modelでOOMする場合は`"grad_cache"`へ切り替え、`batch.grad_cache_chunk_size_per_rank`を調整します。

Schedule-Freeでは学習時と評価時のparameter viewが異なるため、評価とcheckpoint保存を必ずoptimizerのeval mode内で行います。optimizerの`weight_decay_policy = "standard"`はLinear/Conv等の行列weightだけをdecayし、bias、Norm、class token、position embedding、Sigmoid lossのscalar parameterを除外します。旧single-group optimizer checkpointはresume時に新しいgroupへ移行します。非有限lossまたはgradientはそのstepで即座に例外にします。

`optimization/lr`はSchedule-Freeの基準`lr`ではなく、linear warmupを反映してoptimizerが公開する`scheduled_lr`を記録します。通常のAdamWではscheduler適用後の`lr`を記録します。

本sweepはraw/EMA評価と途中checkpointを20 epochごとに実行します。`eval/backbone_knn_top1`が改善した評価時点は`best.pt`へ原子的に上書き保存し、最終epochは`final.pt`へ保存します。gradient clipは100.0とします。pilotで10.0を超えた8手法を6 epoch再実行したところ、100.0でも全runが有限のまま完走し、定常時のgradientを通しながらCircle/Barlow Twinsの初期スパイクだけをclipできたためです。clip前後のglobal norm、clip係数、model/objective別norm、CUDA allocated/reserved memoryを記録します。

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

BYOL/MoCoの内部target encoderはこの評価用EMAとは別です。内部targetとMoCo queueはobjective stateとしてcheckpoint/resumeされ、`target/*`、`byol/*`、`moco/*`へ記録されます。評価用EMAはこれまでどおりonline backboneだけを追跡します。

## 評価プロトコル

`run.seed`はモデル初期化・data order・augmentationを制御し、train/validation分割は独立した`data.split_seed`で固定します。これにより、seed sweepで評価画像そのものが変わる交絡を避けます。

CIFAR-100はstratified split後もtrainが各class 450枚で均衡しているため、train samplerはreplacementなしのepoch permutationを使います。`WeightedRandomSampler`は同一画像の重複・epoch内未使用画像を生み、batch内のclass均衡も保証しないため既定にはしません。class-balanced batchはpositive/negative構成そのものを変えるので、必要なら別sampler ablationとして扱います。

学習中は20 epochごとに次の2種類のk-NNを記録します。どちらも比較前にL2 normalizeします。

- `eval/backbone_knn_top1`: encoder feature上のk-NN。全objectiveで同じ意味を持つ主要指標
- `eval/projector_knn_top1`: projection head出力上のk-NN。損失が直接最適化する空間の診断指標

同じ固定validation表現について、backbone/projectorごとにcollapseと次元冗長性の診断も記録します。`std_mean`はL2 normalize後の次元別標準偏差の平均、`isotropy`はそれを単位球上の最大値`1/sqrt(d)`で正規化した値です。`effective_rank_ratio`は中心化共分散のentropy rankを表現次元で割った値で、`offdiag_correlation_rms`は分散を持つ次元間の非対角相関、`mean_pairwise_cosine`は異なるサンプル全組の平均cosineです。前3者が0へ、またはpairwise cosineが1へ近づく挙動はcollapseを示します。EMA版は`eval_ema/*`に分けて保存します。

最終epochではencoderをeval modeで凍結し、augmentationなしのtrain featureを一度だけ抽出します。その固定feature上で共通の`nn.Linear`をSGD + cosine decayで学習し、`eval/linear_probe_top1`を記録します。encoderやprojectorへ勾配は流れず、probeのseed・epoch・batch size・optimizer条件は`[evaluation.linear_probe]`で全run共通です。core sweepでは`test_at_end = false`として、checkpoint選択中にtest splitを参照しません。

`eval/joint_classifier_top1`は補助診断です。CE/CE+SupConでは学習されますが、対照損失、自己教師あり、objective-owned proxyを使う手法ではmodelのclassifier headがobjectiveに含まれないためchance accuracy付近になるのが正常です。proxy手法の学習中accuracyは`proxy/top1`へ記録します。手法間の主要比較にはbackbone k-NNとfrozen linear probeを使います。
既存checkpointにも同じ評価を後付けできます。checkpointが元runの`checkpoints/`内にあればrun directoryは自動推定され、`wandb.json`に保存されたrun IDを使って同じW&B runへ結果を追記します。移動したcheckpointには`--run-dir`を指定します。GPUを学習runと共有するため、同じGPUでの学習中ではなく停止後または完了後に実行してください。

```bash
uv run contrast evaluate \
  --checkpoint runs/cifar100-core-v3/<run>/checkpoints/best.pt \
  --queries test
```

`--queries`は`config`（保存済み設定に従う）、`eval`、`test`、`both`を選べます。3 seedのvalidation結果からcheckpointと手法を確定した後に、選択済み`best.pt`へ`--queries test`を一度だけ実行します。

全16手法sweepでは、事前に定めた主要指標である3 seed平均のbest validation
backbone k-NNによりProxy Anchorを選択しました。各seedで選択済みの`best.pt`
だけをtest splitで一度評価するコマンドは次のとおりです。

```bash
./run_eval.sh all-methods-winner-test
```

## W&Bへの記録と過去データの移行

今後の学習メトリクスはW&Bへ直接記録します。通常の学習コマンドを実行すれば、設定、タグ、学習・評価メトリクスが`tracking.project`のrunに送信されます。run表示名は`実験名/条件名-seed-N`（例: `cifar100-core-v3/sigmoid-supcon-seed-1`）になります。

```bash
uv run contrast train -c configs/objectives/sigmoid_supcon.toml --set run.seed=1
```

既存の`runs/`にある`config.json`、`environment.json`、`metrics.jsonl`は一括移行できます。最初にdry-runで対象を確認し、その後に送信します。`--entity`は必要な場合だけ追加してください。

```bash
uv run contrast wandb-import --runs-dir runs --project contrast-lab --dry-run
uv run contrast wandb-import --runs-dir runs --project contrast-lab
```

移行では各ローカルrunとW&B runを一対一で対応させ、完了markerによって再実行時の二重送信を防ぎます。同じoptimizer stepにある学習・評価・終了イベントも別々の履歴行として保持し、`step`をグラフの横軸に使います。`environment.json`は再現性に必要な項目だけを送信し、hostname、PID、実行コマンドは送信しません。

モデルとoptimizerのcheckpointは引き続き各runの`checkpoints/`へ原子的に保存されます。W&Bへcheckpointを自動アップロードしないため、resumeとoffline評価にはローカルファイルを残してください。W&B SDKが作る`wandb/`のローカルデータやcacheも、同期完了を確認するまでは削除しないでください。新しいrunでは`metrics.jsonl`を生成しませんが、移行元のファイルは削除しません。

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

## Distributed extension

`RuntimeContext`、global batch sampler、rank情報はtorchrun/DDPを見据えた境界です。ただし、対照損失に必要な全rank表現のgradient-aware gatherは未実装なので、現バージョンはworld size 1を明示的に要求します。誤ったlocal-negative実験を静かに実行しないための制約です。
