# TabICL for Clorch

An unofficial Clojure implementation of the TabICLv1 and TabICLv2 neural
models, built directly on [Clorch](https://github.com/antlobach/clorch).

The v2 implementation strict-loads the official classifier and regressor
checkpoints. It includes the target-aware column encoder, non-interleaved RoPE,
classification and quantile-regression heads, ensemble preprocessing, and
single-dataset fine-tuning.

## Upstream references

- [Official TabICL implementation](https://github.com/soda-inria/tabicl)
- [TabICL models and checkpoints on Hugging Face](https://huggingface.co/jingang/TabICL)
- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564), ICML 2025
- [Clorch](https://github.com/antlobach/clorch)

TabICL is the work of Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. This port is not affiliated with or endorsed by the upstream authors.

## Scope

The v1 classifier remains checkpoint-compatible. TabICLv2 adds strict
checkpoint conversion/loading for both official v2 models. On an RTX 2000 Ada,
the loaded Clojure model matched the official PyTorch checkpoint for a fixed
CUDA input with maximum absolute errors of `5.37e-6` (classifier) and
`4.77e-6` (regressor).

TabICLv2 preprocessing is represented as fitted
[Noj](https://github.com/scicloj/noj)/Metamorph pipelines. Standard scaling,
Yeo-Johnson normalization, soft outlier handling, feature/class shuffling, and
context-only fitting are reused consistently by training, validation, and
inference.

## Use

```clojure
(require '[clorch.nn :as nn]
         '[clorch.torch :as t]
         '[tabicl.core :as tabicl])

(def model (tabicl/tabicl tabicl/paper-config))

(t/with-torch
  (let [x (t/randn [1 12 8])
        y-train (t/tensor [[0 1 0 1 0 1 0 1]] {:dtype :int64})]
    (t/size (nn/forward model {:x x :y-train y-train}))))
;; => [1 4 10]
```

For repeated predictions against the same training table, use `prepare-cache` once and pass its result to `forward-with-cache`.

## TabICLv2 fine-tuning

Convert an official checkpoint once:

```bash
python scripts/convert_v2_checkpoint.py \
  tabicl-classifier-v2-20260212.ckpt \
  tabicl-classifier-v2.safetensors
```

The two executable examples use the exact synthetic datasets and train,
validation, and test splits from the upstream tutorials:

```bash
clojure -M -m tabicl.examples.finetune-classifier \
  tabicl-classifier-v2.safetensors

clojure -M -m tabicl.examples.finetune-regressor \
  tabicl-regressor-v2.safetensors
```

Both use AdamW, FP16 autocast and gradient scaling, gradient clipping,
cosine decay with warmup, fresh context/query meta-batches, the official
cross-entropy or pinball objective, validation-driven early stopping, and
best-state restoration.

## Development

Start the long-lived nREPL:

```bash
CLOJURE_DISABLE_RLWRAP=1 clojure -M:dev
```

The `:dev` alias uses a local `../clorch` checkout and the macOS 13-compatible PyTorch preset used for this port's verification. Normal consumers use the pinned Clorch Git revision.

## License

BSD 3-Clause. See [LICENSE](LICENSE). The upstream attribution is retained there.
