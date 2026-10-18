# TabICL API roadmap

The current low-level `tabicl.v2.model` and `tabicl.v2.finetune` namespaces prove the model architecture, checkpoint conversion, inference, and fine-tuning paths. The next pull request should add a production-facing estimator API without removing those lower-level building blocks.

## Goal

Make the common path as small as Python TabICL while preserving idiomatic Clojure data and direct Noj/Metamorph interoperability.

```clojure
(def classifier (tabicl/classifier))
(def fitted (tabicl/fit classifier x-train y-train))
(tabicl/predict fitted x-test)
```

The same estimator should plug into `metamorph.ml`:

```clojure
(ml/model {:model-type :tabicl/classifier})
```

## High-level inference estimators

Add public `tabicl/classifier` and `tabicl/regressor` constructors. Constructors return immutable estimator configuration; `fit` returns fitted state containing the loaded model, preprocessing state, context rows, target metadata, and class mapping.

Support these Python-equivalent options, using idiomatic kebab-case keys:

- `:n-estimators`
- `:norm-methods`
- `:feature-shuffle-method`
- `:class-shuffle-method`
- `:outlier-threshold`
- `:softmax-temperature`
- `:average-logits?`
- `:support-many-classes?`
- `:batch-size`
- `:kv-cache?`
- `:model-path`
- `:auto-download?`
- `:checkpoint-version`
- `:device`
- `:use-amp`
- `:use-fa3`
- `:offload-mode`
- `:disk-offload-dir`
- `:random-state`
- `:n-jobs`
- `:verbose?`
- `:inference-config`

Do not silently accept inert options. Each option must either affect execution or fail with a clear unsupported-capability error. Device auto-selection must only choose backends that Clorch actually supports; today that means CUDA when available, otherwise CPU. AMP, Flash Attention 3, KV caching, many-class decomposition, and offloading require reusable Clorch primitives before they can be exposed as working TabICL options.

Checkpoint handling should accept a local path, cache downloads by checkpoint version, verify the downloaded artifact, and default to automatic download when no path is supplied. Loading and device placement happen once per fitted estimator, not once per prediction.

## High-level fine-tuning estimators

Add `tabicl/finetuned-classifier` and `tabicl/finetuned-regressor` constructors with:

- `:epochs`
- `:learning-rate`
- `:n-estimators-finetune`
- `:n-estimators-validation`
- `:n-estimators-inference`
- `:early-stopping?`
- `:patience`
- `:eval-metric`
- `:random-state`
- `:verbose?`

`fit` must optionally accept validation features and targets plus an output directory. Classifier metrics: `:roc-auc`, `:log-loss`, and `:accuracy`. Regressor metrics: `:mse`, `:mae`, and `:r2`. Early stopping must restore the best parameters. Saved checkpoints must use the same schema consumed by the zero-shot estimators.

## Noj and Metamorph contract

The high-level API and `metamorph.ml` registrations must call the same implementation rather than maintain parallel inference paths.

Required behavior:

- Accept `tech.v3.dataset`/Tablecloth datasets directly.
- Preserve feature order, target name, target metadata, and classification labels.
- Return a prediction column plus correctly tagged class-probability columns for classifiers.
- Return a correctly tagged prediction column for regressors.
- Keep fitted model and preprocessing state in the Metamorph fit context and reuse it during transform.
- Support preloaded models for explicit lifecycle control, while documenting that mutable native model instances cannot be used concurrently without synchronization.
- Close owned native resources deterministically; never close caller-owned preloaded models.
- Make seeded ensemble transformations reproducible.
- Validate missing values, unsupported dtypes, feature-schema drift, unseen labels, excessive class count, and incompatible classifier/regressor checkpoints at the API boundary.

## Reusable Clorch work

Implement model-independent capabilities in Clorch first, then consume them from TabICL:

1. Checkpoint download/cache utilities and artifact verification.
2. Mixed-precision inference primitives and capability checks.
3. Batched ensemble execution without avoidable tensor copies.
4. Reusable key/value projection caches with explicit native-memory ownership.
5. Device capability selection and clear unsupported-backend errors.
6. CPU/disk parameter offloading only if it can be expressed independently of TabICL.
7. Thread-count configuration and deterministic RNG control.

TabICL should retain model-specific ensemble permutations, normalization choices, many-class decomposition, checkpoint naming, and estimator defaults.

## Acceptance checks

- One-line default classifier and regressor construction with automatic checkpoint acquisition.
- Numerical comparison against the matching Python checkpoint and deterministic configuration.
- Fit/predict and fine-tune/predict smoke tests on CPU and a JVM 25 CUDA RunPod.
- Noj/Metamorph fit and transform examples for classification and regression.
- Repeated prediction demonstrates that checkpoint and context preprocessing are not rebuilt.
- Resource-lifecycle test proves owned model cleanup and caller-owned model preservation.
- Public README documents the high-level path first and lower-level APIs as advanced usage.
