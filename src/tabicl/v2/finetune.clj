(ns tabicl.v2.finetune
  (:require [clorch.amp :as amp]
            [clorch.autograd :as autograd]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.optim :as optim]
            [clorch.torch :as t]
            [tabicl.v2.preprocessing :as prep])
  (:import [java.util ArrayList Collections Random]))

(set! *warn-on-reflection* true)

(def default-options
  {:epochs 30
   :learning-rate 1.0e-5
   :weight-decay 0.01
   :grad-clip 1.0
   :amp? true
   :warmup-proportion 0.1
   :n-estimators 2
   :query-ratio 0.2
   :max-data-size 10000
   :min-chunk-size 50
   :patience 5
   :early-stopping? true
   :random-state 0
   :norm-methods [:power :none]
   :outlier-threshold 4.0
   :device :cuda})

(defn- clip [value lower upper]
  (max lower (min upper value)))

(defn cosine-warmup-factor
  "Matches the cosine-with-warmup factor used by the official fine-tuner."
  [step warmup-steps total-steps]
  (if (< step warmup-steps)
    (/ (double step) (double (max 1 warmup-steps)))
    (let [progress (/ (double (- step warmup-steps))
                      (double (max 1 (- total-steps warmup-steps))))]
      (if (>= progress 1.0)
        0.0
        (* 0.5 (+ 1.0 (Math/cos (* Math/PI progress))))))))

(defn- shuffled [values seed]
  (let [result (ArrayList. ^java.util.Collection (vec values))]
    (Collections/shuffle result (Random. (long seed)))
    (vec result)))

(defn- split-indices [labels query-size classification? seed]
  (let [n (count labels)
        query-size (max 1 (min query-size (dec n)))]
    (if-not classification?
      (let [indices (shuffled (range n) seed)]
        [(subvec indices query-size) (subvec indices 0 query-size)])
      (let [groups (vals (group-by #(nth labels %) (range n)))
            selected (mapcat (fn [group]
                               (let [count-for-class (max 1 (long (Math/round
                                                                   (* query-size
                                                                      (/ (count group)
                                                                         (double n))))))]
                                 (take (min count-for-class (dec (count group)))
                                       (shuffled group (+ seed (long (nth labels (first group))))))))
                             groups)
            query (vec (take query-size selected))
            query-set (set query)
            context (vec (remove query-set (range n)))]
        [context query]))))

(defn meta-batch
  "Builds one fresh context/query ensemble meta-batch from JVM feature rows."
  [x y classification? epoch options]
  (let [{:keys [n-estimators query-ratio random-state split-seed
                norm-methods outlier-threshold]}
        (merge default-options options)
        query-size (max 1 (long (* (count y) query-ratio)))
        [context-indices query-indices]
        (split-indices y (if classification?
                           (max query-size (count (set y)))
                           query-size)
                       classification?
                       (or split-seed (+ random-state epoch)))
        x-context (mapv x context-indices)
        x-query (mapv x query-indices)
        raw-y-context (mapv y context-indices)
        raw-y-query (mapv y query-indices)
        [y-context y-query y-mean y-std]
        (if classification?
          [raw-y-context raw-y-query nil nil]
          (let [center (/ (reduce + 0.0 raw-y-context) (double (count raw-y-context)))
                scale (max 1.0e-8
                           (Math/sqrt
                            (/ (reduce (fn [sum value]
                                         (let [delta (- (double value) center)]
                                           (+ sum (* delta delta))))
                                       0.0 raw-y-context)
                               (double (count raw-y-context)))))]
            [(mapv #(/ (- (double %) center) scale) raw-y-context)
             (mapv #(/ (- (double %) center) scale) raw-y-query)
             center scale]))
        configurations (prep/fit-ensemble x-context y-context
                                          :classification? classification?
                                          :n-estimators n-estimators
                                          :norm-methods norm-methods
                                          :outlier-threshold outlier-threshold)
        variants (mapv (fn [configuration]
                         (let [context (prep/transform-variant configuration x-context y-context)
                               query (prep/transform-variant configuration x-query y-query)]
                           {:x (into (:x context) (:x query))
                            :y-train (:y context)
                            :y-query (:y query)}))
                       configurations)]
    {:x (mapv :x variants)
     :y-train (mapv :y-train variants)
     :y-query (mapv :y-query variants)
     :train-size (count context-indices)
     :y-mean y-mean
     :y-std y-std}))

(defn count-chunks
  "Returns the number of fine-tuning chunks yielded per epoch."
  [sample-count max-data-size min-chunk-size]
  (cond
    (not (pos? sample-count)) 0
    (<= sample-count max-data-size) 1
    :else (let [full (quot sample-count max-data-size)
                remainder (mod sample-count max-data-size)]
            (+ full (if (>= remainder min-chunk-size) 1 0)))))

(defn epoch-meta-batches
  "Builds freshly shuffled chunks and context/query splits for one epoch."
  [x y classification? epoch options]
  (let [{:keys [max-data-size min-chunk-size random-state]}
        (merge default-options options)
        indices (shuffled (range (count y)) (+ random-state epoch))
        chunks (if (<= (count y) max-data-size)
                 [indices]
                 (->> indices
                      (partition-all max-data-size)
                      (filter #(>= (count %) min-chunk-size))))
        indexed (map-indexed vector chunks)]
    (mapv (fn [[chunk-index indices]]
            (meta-batch (mapv x indices)
                        (mapv y indices)
                        classification?
                        epoch
                        (assoc options
                               :split-seed (+ random-state epoch
                                              (* chunk-index 7919)))))
          indexed)))

(defn classification-loss
  "Cross-entropy over active classes and all query/ensemble rows."
  [tabicl batch device]
  (let [x (t/tensor (:x batch) {:dtype :float32 :device device})
        y-train (t/tensor (:y-train batch) {:dtype :float32 :device device})
        y-query (t/tensor (:y-query batch) {:dtype :int64 :device device})
        logits (nn/forward tabicl {:x x :y-train y-train})
        class-count (inc (long (apply max (mapcat identity (:y-train batch)))))]
    (F/cross-entropy (t/reshape (t/ix logits :_ :_ [0 class-count]) [-1 class-count])
                     (t/reshape y-query [-1]))))

(defn pinball-loss
  "Official TabICLv2 quantile objective over ensemble members, queries, and quantiles."
  [tabicl batch device]
  (let [x (t/tensor (:x batch) {:dtype :float32 :device device})
        y-train (t/tensor (:y-train batch) {:dtype :float32 :device device})
        target (t/tensor (:y-query batch) {:dtype :float32 :device device})
        quantiles (nn/forward tabicl {:x x :y-train y-train})
        quantile-count (last (t/size quantiles))
        alpha (t/tensor (mapv #(/ (double %) (inc quantile-count))
                              (range 1 (inc quantile-count)))
                        {:dtype :float32 :device device})
        difference (t/sub (t/unsqueeze target -1) quantiles)]
    (t/mean (t/maximum (t/mul alpha difference)
                       (t/mul (t/sub alpha 1.0) difference)))))

(defn- tensor-matrix [tensor]
  (let [columns (last (t/size tensor))
        values (mapv t/item-float (t/tseq (t/reshape tensor [-1])))]
    (mapv vec (partition columns values))))

(defn predict
  "Runs ensemble prediction using preprocessing fitted only on the supplied context."
  [tabicl x-context y-context x-query classification? options]
  (let [{:keys [device n-estimators norm-methods outlier-threshold temperature]
         :or {temperature 0.9}}
        (merge default-options options)
        [normalized-y center scale]
        (if classification?
          [y-context nil nil]
          (let [center (/ (reduce + 0.0 y-context) (double (count y-context)))
                scale (max 1.0e-8
                           (Math/sqrt
                            (/ (reduce (fn [sum value]
                                         (let [delta (- (double value) center)]
                                           (+ sum (* delta delta))))
                                       0.0 y-context)
                               (double (count y-context)))))]
            [(mapv #(/ (- (double %) center) scale) y-context) center scale]))
        configurations (prep/fit-ensemble x-context normalized-y
                                          :classification? classification?
                                          :n-estimators n-estimators
                                          :norm-methods norm-methods
                                          :outlier-threshold outlier-threshold)
        variants (mapv (fn [configuration]
                         (let [context (prep/transform-variant configuration x-context normalized-y)
                               query (prep/transform-variant configuration x-query (repeat (count x-query) 0))]
                           {:x (into (:x context) (:x query))
                            :y (:y context)}))
                       configurations)]
    (nn/train tabicl false)
    (try
      (t/with-torch
        (autograd/no-grad
         (let [x (t/tensor (mapv :x variants) {:dtype :float32 :device device})
               y (t/tensor (mapv :y variants) {:dtype :float32 :device device})
               output (nn/forward tabicl {:x x :y-train y})]
           (if classification?
             (let [class-count (count (set y-context))
                   probabilities (t/softmax
                                  (t/div (t/mean (t/ix output :_ :_ [0 class-count]) 0)
                                         temperature)
                                  -1)]
               (tensor-matrix probabilities))
             (let [means (t/mean (t/mean output -1) 0)]
               (mapv #(+ center (* scale (t/item-float %))) (t/tseq means)))))))
      (finally
        (nn/train tabicl true)))))

(defn classification-metrics [probabilities targets]
  (let [positive (mapv second probabilities)
        pairs (sort-by first (map vector positive targets))
        negatives (count (filter zero? targets))
        positives (count (remove zero? targets))
        rank-sum (reduce + 0.0
                         (keep-indexed (fn [index [_ target]]
                                         (when (= 1 (long target)) (inc index)))
                                       pairs))
        auc (/ (- rank-sum (/ (* positives (inc positives)) 2.0))
               (* positives negatives))
        epsilon 1.0e-15
        log-loss (- (/ (reduce + 0.0
                               (map (fn [probability target]
                                      (Math/log (clip (nth probability (long target))
                                                      epsilon (- 1.0 epsilon))))
                                    probabilities targets))
                       (count targets)))
        accuracy (/ (count (filter true?
                                   (map (fn [probability target]
                                          (= (long target)
                                             (if (> (second probability) (first probability)) 1 0)))
                                        probabilities targets)))
                    (double (count targets)))]
    {:primary auc :roc-auc auc :log-loss log-loss :accuracy accuracy}))

(defn regression-metrics [predictions targets]
  (let [errors (mapv - predictions targets)
        mse (/ (reduce + 0.0 (map #(* % %) errors)) (count errors))
        mae (/ (reduce + 0.0 (map #(Math/abs (double %)) errors)) (count errors))
        center (/ (reduce + 0.0 targets) (count targets))
        total (reduce + 0.0 (map #(let [delta (- % center)] (* delta delta)) targets))]
    {:primary mse :mse mse :mae mae
     :r2 (- 1.0 (/ (* mse (count targets)) total))}))

(defn validate
  "Evaluates current model weights with the same fitted-context inference path."
  [tabicl x-train y-train x-validation y-validation classification? options]
  (let [predictions (predict tabicl x-train y-train x-validation classification? options)]
    (if classification?
      (classification-metrics predictions y-validation)
      (regression-metrics predictions y-validation))))

(defn fine-tune!
  "Fine-tunes a loaded TabICLv2 model and restores the best validation state.

  The model and optimizer live outside per-batch `with-torch` scopes. Each epoch
  rebuilds the official context/query ensemble meta-batch."
  [tabicl x-train y-train x-validation y-validation classification? & [options]]
  (let [{:keys [epochs learning-rate weight-decay grad-clip amp? warmup-proportion
                patience early-stopping? device max-data-size min-chunk-size]
         :as options} (merge default-options options)
        parameters (nn/parameters tabicl :trainable-only true)
        optimizer (optim/adamw parameters :lr learning-rate :weight-decay weight-decay)
        scaler (amp/grad-scaler {:enabled? (and amp? (= device :cuda))})
        chunks-per-epoch (max 1 (count-chunks (count y-train)
                                              max-data-size
                                              min-chunk-size))
        total-steps (* epochs chunks-per-epoch)
        warmup-steps (long (* total-steps warmup-proportion))
        baseline (validate tabicl x-train y-train x-validation y-validation classification? options)
        best-state (atom (nn/clone-flat-state-dict tabicl))
        best-metric (atom (:primary baseline))
        patience-counter (atom 0)
        global-step (atom 0)
        history (atom [])
        maximize? classification?]
    (nn/train tabicl true)
    (loop [epoch 0]
      (when (and (< epoch epochs)
                 (or (not early-stopping?) (< @patience-counter patience)))
        (let [batches (epoch-meta-batches x-train y-train classification? epoch options)
              losses
              (mapv
               (fn [batch]
                 (let [factor (cosine-warmup-factor @global-step warmup-steps total-steps)]
                   (optim/set-learning-rate! optimizer (* learning-rate factor))
                   (optim/zero-grad optimizer)
                   (let [loss-value
                         (t/with-torch
                           (let [loss (amp/call-with-autocast
                                       {:device device
                                        :dtype :float16
                                        :enabled? (and amp? (= device :cuda))}
                                       #(if classification?
                                          (classification-loss tabicl batch device)
                                          (pinball-loss tabicl batch device)))]
                             (amp/backward! scaler loss)
                             (when (and grad-clip
                                        (pos? grad-clip)
                                        (amp/unscale! scaler optimizer))
                               (nn/clip-grad-norm! parameters grad-clip))
                             (amp/step! scaler optimizer)
                             (t/item-float loss)))]
                     (swap! global-step inc)
                     loss-value)))
               batches)
              mean-loss (/ (reduce + 0.0 losses) (double (count losses)))
              metrics (validate tabicl x-train y-train x-validation y-validation
                                classification? options)
              improved? (if maximize?
                          (> (:primary metrics) @best-metric)
                          (< (:primary metrics) @best-metric))]
          (swap! history conj (assoc metrics
                                     :epoch (inc epoch)
                                     :train-loss mean-loss
                                     :learning-rate (first (optim/learning-rates optimizer))))
          (if improved?
            (do
              (t/release! @best-state)
              (reset! best-state (nn/clone-flat-state-dict tabicl))
              (reset! best-metric (:primary metrics))
              (reset! patience-counter 0))
            (swap! patience-counter inc))
          (recur (inc epoch)))))
    (nn/load-flat-state-dict! tabicl @best-state)
    (t/release! @best-state)
    {:model tabicl
     :baseline baseline
     :best-metric @best-metric
     :epochs-run (count @history)
     :history @history}))
