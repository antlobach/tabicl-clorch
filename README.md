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

## TabICLv2 quick start

### Download and convert the official weights

The JVM loader reads SafeTensors rather than Python pickle. Download the
official checkpoints and convert each one once:

```bash
python -m pip install torch safetensors huggingface_hub

hf download jingang/TabICL tabicl-classifier-v2-20260212.ckpt \
  --local-dir checkpoints
hf download jingang/TabICL tabicl-regressor-v2-20260212.ckpt \
  --local-dir checkpoints

python scripts/convert_v2_checkpoint.py \
  checkpoints/tabicl-classifier-v2-20260212.ckpt \
  checkpoints/tabicl-classifier-v2.safetensors
python scripts/convert_v2_checkpoint.py \
  checkpoints/tabicl-regressor-v2-20260212.ckpt \
  checkpoints/tabicl-regressor-v2.safetensors
```

Each conversion writes a `.safetensors` file and a same-named `.json`
configuration file. Keep them together. Loading is strict: missing, unexpected,
or shape-mismatched model parameters raise an exception.

### Classification

`finetune/predict` accepts ordinary JVM row vectors. It fits a Noj/Metamorph
ensemble on the supplied context rows and returns one probability vector per
query row:

```clojure
(require '[clorch.nn :as nn]
         '[tabicl.v2.finetune :as finetune]
         '[tabicl.v2.model :as v2])

(def device :cuda)
(def classifier
  (nn/to (v2/load-converted-model
          "checkpoints/tabicl-classifier-v2.safetensors")
         device))

(def x-context [[0.1 1.2] [0.8 -0.4] [-1.1 0.3] [1.4 0.9]])
(def y-context [0 1 0 1])
(def x-query [[0.2 0.7] [-0.9 0.1]])

(finetune/predict classifier
                  x-context y-context x-query
                  true
                  {:device device :n-estimators 2})
;; => [[p(class-0) p(class-1)]
;;     [p(class-0) p(class-1)]]
```

Resolve the application-level device once and keep the loaded model outside
repeated prediction or training scopes.

### Regression

Load the regressor checkpoint and pass `false` for the classification flag.
The result contains the ensemble-averaged means of the 999 predicted
quantiles, transformed back to the target scale:

```clojure
(def regressor
  (nn/to (v2/load-converted-model
          "checkpoints/tabicl-regressor-v2.safetensors")
         device))

(finetune/predict regressor
                  [[-1.0] [0.0] [1.0] [2.0]]
                  [-0.8 0.0 0.9 0.4]
                  [[0.5] [1.5]]
                  false
                  {:device device :n-estimators 2})
;; => [predicted-mean-0 predicted-mean-1]
```

### Use TabICL as a Metamorph model

Requiring `tabicl.noj` registers `:tabicl/classifier` and
`:tabicl/regressor` with `metamorph.ml`. They work with the ordinary
`ml/model` pipeline step; fitted pipeline state retains the pretrained model
and context table for subsequent transforms:

```clojure
(require '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml :as ml]
         '[tabicl.noj]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset.metamorph :as ds-mm])

(def training-data
  (tc/dataset {:x1 [0.1 0.8 -1.1 1.4]
               :x2 [1.2 -0.4 0.3 0.9]
               :label [0 1 0 1]}))

(def pipeline
  (mm/pipeline
   (ds-mm/set-inference-target :label)
   {:metamorph/id :tabicl}
   (ml/model {:model-type :tabicl/classifier
              :checkpoint "checkpoints/tabicl-classifier-v2.safetensors"
              :device :cuda
              :n-estimators 2})))

(def fitted
  (pipeline {:metamorph/data training-data
             :metamorph/mode :fit}))

(def predictions
  (:metamorph/data
   (pipeline (assoc fitted
                    :metamorph/data
                    (tc/dataset {:x1 [0.2 -0.9]
                                 :x2 [0.7 0.1]
                                 ;; The target column can be present during
                                 ;; evaluation and is ignored for prediction.
                                 :label [0 0]})
                    :metamorph/mode :transform))))
```

`predictions` contains the predicted `:label` column, marked as
`:prediction`, plus one probability-distribution column per observed class.
Use `:model-type :tabicl/regressor` with the converted regressor checkpoint
for regression pipelines. `:checkpoint` may be replaced with a preloaded
`:model` to avoid repeated checkpoint loading. A preloaded model is mutable
native state; do not use the same instance concurrently from multiple
pipelines.

## TabICLv1 low-level model

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

For repeated predictions against the same training table, use `prepare-cache`
once and pass its result to `forward-with-cache`.

## TabICLv2 fine-tuning

The two executable examples load the converted pretrained weights before
fine-tuning. They use the exact synthetic datasets and train, validation, and
test splits from the upstream tutorials:

```bash
clojure -M -m tabicl.examples.finetune-classifier \
  checkpoints/tabicl-classifier-v2.safetensors

clojure -M -m tabicl.examples.finetune-regressor \
  checkpoints/tabicl-regressor-v2.safetensors
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

The `:dev` alias uses a local `../clorch` checkout. Normal consumers use the
pinned Clorch Git revision.

## License

BSD 3-Clause. See [LICENSE](LICENSE). The upstream attribution is retained there.
