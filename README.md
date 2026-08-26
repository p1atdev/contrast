# Contrast Lab

PyTorchで教師あり対照学習を条件統制して比較するための実験基盤です。中核のViT、投影ヘッド、損失、拡張、GradCacheは自前実装し、設定検証にPydantic、Schedule-Free optimizer、管理APIにFastAPIを使います。

## 比較の前提

既定値はCIFAR-100、ViT-Tiny/4相当（dim 192、12 blocks、3 heads）、LayerNorm、GELU、eager attentionです。global source batch、view数、augmentation、optimizer、precisionを固定し、`objective`だけを差し替えられます。

実装済みの損失は以下です。

- Cross-Entropy
- NT-Xent（同一画像の別viewをpositiveとする）
- Supervised Contrastive Loss
- SINCERE
- Sigmoid-SupCon（SigLIP型のpairwise sigmoidを、同一クラスをpositiveとする多positive設定へ拡張）
- Cross-Entropy + SupCon

Sigmoid-SupConは画像・テキストの二塔SigLIPそのものではありません。ほかの画像単塔損失とモデル条件を揃えるため、SigLIPのpairwise sigmoid設計を教師ありpositive maskへ適用しています。

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

損失5種 × seed 3種の逐次sweepは次のコマンドです。最初に`--dry-run`で展開結果を確認できます。

```bash
uv run contrast sweep configs/sweeps/core_losses.toml --dry-run
uv run contrast sweep configs/sweeps/core_losses.toml
```

## PrecisionとGradCache

既定はFP32 parameter、BF16 autocast、FP32 loss、TF32許可です。TF32自体は主に速度向上の設定で、activation memory削減はBF16 autocastが担います。同じGPU・ソフトウェア条件でseed、data order、viewごとのaugmentationを固定します。より強い決定性が必要なら`reproducibility.mode = "strict"`を指定できます。

GradCacheはlogical batch全体の表現から一度だけ損失を計算し、chunkごとにforwardを再実行します。optimizer stepはlogical batchにつき1回です。`tests/test_grad_cache.py`でDirectとの勾配一致を検証します。現時点のGradCacheはFP32/BF16を対象とし、FP16 GradScalerは明示的に拒否します。

Schedule-Freeでは学習時と評価時のparameter viewが異なるため、評価とcheckpoint保存を必ずoptimizerのeval mode内で行います。

## 評価プロトコル

`run.seed`はモデル初期化・data order・augmentationを制御し、train/validation分割は独立した`data.split_seed`で固定します。これにより、seed sweepで評価画像そのものが変わる交絡を避けます。

学習中は10 epochごとに次の2種類のk-NNを記録します。どちらも比較前にL2 normalizeします。

- `eval/backbone_knn_top1`: encoder feature上のk-NN。全objectiveで同じ意味を持つ主要指標
- `eval/projector_knn_top1`: projection head出力上のk-NN。損失が直接最適化する空間の診断指標

最終epochではencoderをeval modeで凍結し、augmentationなしのtrain featureを一度だけ抽出します。その固定feature上で共通の`nn.Linear`をSGD + cosine decayで学習し、`eval/linear_probe_top1`を記録します。encoderやprojectorへ勾配は流れず、probeのseed・epoch・batch size・optimizer条件は`[evaluation.linear_probe]`で全run共通です。`test_at_end = true`なら同じprobeで`test/linear_probe_top1`も最終時だけ計測します。

`eval/joint_classifier_top1`は補助診断です。CE/CE+SupConでは学習されますが、対照損失単独ではclassifier headがobjectiveに含まれないためchance accuracy付近になるのが正常です。手法間の主要比較にはbackbone k-NNとfrozen linear probeを使います。
既存checkpointにも同じ評価を後付けできます。checkpointが元runの`checkpoints/`内にあればrun directoryは自動推定され、結果はそのrunの`metrics.jsonl`へ追記されます。移動したcheckpointには`--run-dir`を指定します。GPUを学習runと共有するため、同じGPUでの学習中ではなく停止後または完了後に実行してください。

```bash
uv run contrast evaluate --checkpoint runs/cifar100-core/<run>/checkpoints/final.pt
```

## Dashboard

学習メトリクスは各runの`metrics.jsonl`が正本です。React UIをbuildした後、APIとUIを同じプロセスから起動します。

```bash
uv run contrast serve --runs-dir runs
```

[http://127.0.0.1:8000](http://127.0.0.1:8000) で最大6 runを選び、loss、backbone/projector k-NN、frozen linear probe、gradient、Sigmoidパラメータなどを同時に比較できます。既存runの旧指標もlegacy cardに残ります。UI開発時は別端末で次を実行します。

```bash
cd web
bun run dev
```

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
