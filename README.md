# Contrast Lab

PyTorchで教師あり対照学習を条件統制して比較するための実験基盤です。中核のViT、投影ヘッド、損失、拡張、GradCacheは自前実装し、設定検証にPydantic、Schedule-Free optimizer、管理APIにFastAPIを使います。

## 比較の前提

既定値は CIFAR-100、ViT-Tiny/4相当（dim 192、12 blocks、3 heads）、LayerNorm、GELU、eager attentionです。global source batch、view数、augmentation、optimizer、precisionを固定し、objectiveだけを差し替えられます。

実装済みの損失は以下です。

- Cross-Entropy
- NT-Xent（同一画像の別viewをpositiveとする）
- Supervised Contrastive Loss
- SINCERE
- Sigmoid-SupCon（SigLIP型のpairwise sigmoidを、同一クラスをpositiveとする多positive設定へ拡張）
- Cross-Entropy + SupCon

Sigmoid-SupConは画像・テキストの二塔SigLIPそのものではありません。ほかの画像単塔損失とモデル条件を揃えるため、SigLIPのpairwise sigmoid設計を教師ありpositive maskへ適用しています。

## Setup

Python 3.12とBunを使います。

    uv sync
    cd web
    bun install
    bun run build
    cd ..

設定の解決結果だけを確認できます。

    uv run contrast validate -c configs/objectives/sigmoid_supcon.toml

短い配線確認と本学習の例です。

    uv run contrast train -c configs/experiments/smoke.toml
    uv run contrast train -c configs/objectives/sigmoid_supcon.toml --set run.seed=1

損失5種 × seed 3種の逐次sweepは次のコマンドです。最初にdry-runで展開結果を確認できます。

    uv run contrast sweep configs/sweeps/core_losses.toml --dry-run
    uv run contrast sweep configs/sweeps/core_losses.toml

## PrecisionとGradCache

既定はFP32 parameter、BF16 autocast、FP32 loss、TF32許可です。TF32自体は主に速度向上の設定で、activation memory削減はBF16 autocastが担います。同じGPU・ソフトウェア条件でseed、data order、viewごとのaugmentationを固定します。より強い決定性が必要ならreproducibility.modeをstrictに指定できます。

GradCacheはlogical batch全体の表現から一度だけ損失を計算し、chunkごとにforwardを再実行します。optimizer stepはlogical batchにつき1回です。tests/test_grad_cache.pyでDirectとの勾配一致を検証します。現時点のGradCacheはFP32/BF16を対象とし、FP16 GradScalerは明示的に拒否します。

Schedule-Freeでは学習時と評価時のparameter viewが異なるため、評価とcheckpoint保存を必ずoptimizerのeval mode内で行います。

## Dashboard

学習メトリクスは各runのmetrics.jsonlが正本です。React UIをbuildした後、APIとUIを同じプロセスから起動します。

    uv run contrast serve --runs-dir runs

http://127.0.0.1:8000 で最大6 runを選び、stepを揃えたloss、k-NN、classifier accuracy、learning rateと主要なresolved configを比較できます。UI開発時は別端末でcd web && bun run devを実行します。

## Quality checks

    uv run ruff format --check .
    uv run ruff check .
    uv run pytest
    cd web
    bun run format:check
    bun run lint
    bun test
    bun run build

## Distributed extension

RuntimeContext、global batch sampler、rank情報はtorchrun/DDPを見据えた境界です。ただし、対照損失に必要な全rank表現のgradient-aware gatherは未実装なので、現バージョンはworld size 1を明示的に要求します。誤ったlocal-negative実験を静かに実行しないための制約です。
