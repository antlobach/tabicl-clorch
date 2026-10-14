# TabICL for Clorch

An unofficial Clojure port of the TabICLv1 neural model, implemented with [Clorch](https://github.com/antlobach/clorch).

This repository translates the original PyTorch architecture rather than wrapping Python. It covers the checkpoint-compatible TabICLv1 neural network: column embedding with induced attention, row interaction with interleaved rotary embeddings, label-conditioned in-context learning, mixed-width feature masks, and cached inference.

## Upstream references

- [Official TabICL implementation](https://github.com/soda-inria/tabicl)
- [TabICL models and checkpoints on Hugging Face](https://huggingface.co/jingang/TabICL)
- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data](https://arxiv.org/abs/2502.05564), ICML 2025
- [Clorch](https://github.com/antlobach/clorch)

TabICL is the work of Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. This port is not affiliated with or endorsed by the upstream authors.

## Scope

The port reproduces the original TabICLv1 classifier model and can load its state dictionary after it has been stored as an F32 SafeTensors archive with the original PyTorch parameter names. Against the official `tabicl-classifier-v1-20250208.ckpt`, its CPU logits match the PyTorch reference within `2e-5` maximum absolute error.

The upstream estimator's scikit-learn preprocessing ensemble, automatic support for more than ten classes, and memory-offloading scheduler are outside this neural-model port.

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

## Development

Start the long-lived nREPL:

```bash
CLOJURE_DISABLE_RLWRAP=1 clojure -M:dev
```

The `:dev` alias uses a local `../clorch` checkout and the macOS 13-compatible PyTorch preset used for this port's verification. Normal consumers use the pinned Clorch Git revision.

## License

BSD 3-Clause. See [LICENSE](LICENSE). The upstream attribution is retained there.
