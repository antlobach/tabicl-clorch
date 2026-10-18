(ns tabicl.core
  "A from-scratch expression of the TabICLv1 architecture from Qu et al., ICML 2025.

  This matches the official Hugging Face TabICLv1 checkpoint architecture and can
  load its converted weights. Estimator-level preprocessing and ensembling remain
  separate from this model."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clorch.nn :as nn]
            [clorch.nn.init :as init]
            [clorch.safetensors :as safetensors]
            [clorch.torch :as t]))

(set! *warn-on-reflection* true)

(defn- initialized-transformer-block
  [model-dim heads feedforward-dim dropout-probability]
  (let [block (nn/transformer-encoder-layer
               model-dim heads
               :feedforward-dim feedforward-dim
               :dropout dropout-probability
               :activation :gelu
               :norm-first? true)
        parameters (nn/named-parameters block)]
    (doseq [projection (t/chunk
                        (get parameters "attention.in-projection.weight")
                        3
                        0)]
      (init/xavier-uniform! projection (/ 1.0 (Math/sqrt 2.0))))
    (init/zeros! (get parameters "attention.in-projection.bias"))
    (init/zeros! (get parameters "attention.out-projection.weight"))
    (init/zeros! (get parameters "attention.out-projection.bias"))
    (init/zeros! (get parameters "feedforward[3].weight"))
    (init/zeros! (get parameters "feedforward[3].bias"))
    block))

(defn- rope-frequencies [head-dim sequence-length]
  (let [[cosines sines]
        (t/precompute-rope-freqs
         head-dim sequence-length :theta 100000.0 :interleaved? true)]
    {:cos cosines :sin sines :interleaved? true}))

(defn- initialized-label-weight [max-classes model-dim]
  (let [weight (t/empty [model-dim max-classes])
        bound (/ 1.0 (Math/sqrt (double max-classes)))]
    (init/uniform! weight (- bound) bound)
    (nn/parameter (t/clone (t/transpose weight 0 1)))))

(defn- initialized-label-bias [max-classes model-dim]
  (let [bias (t/empty [model-dim])
        bound (/ 1.0 (Math/sqrt (double max-classes)))]
    (init/uniform! bias (- bound) bound)
    (nn/parameter bias)))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel InducedAttentionBlock [model-dim heads feedforward-dim inducing-count dropout-probability]
  [inducing-points (let [parameter (nn/parameter (t/empty [inducing-count model-dim]))]
                     (init/trunc-normal! (:tensor parameter) :std 0.02)
                     parameter)
   inducing-attention (initialized-transformer-block
                       model-dim heads feedforward-dim dropout-probability)
   input-attention (initialized-transformer-block
                    model-dim heads feedforward-dim dropout-probability)
   num-inducing inducing-count
   dim model-dim]
  (forward [input]
           (let [{:keys [x train-size]} input
                 batch-size (first (t/size x))
                 inducing (t/expand (t/unsqueeze (:tensor inducing-points) 0)
                                    [batch-size num-inducing dim])
                 context (if train-size (t/ix x :_ [0 train-size] :_) x)
                 hidden (nn/forward inducing-attention
                                    {:x inducing :key context :value context})]
             (nn/forward input-attention {:x x :key hidden :value hidden}))))

(defn- run-blocks [blocks input]
  (reduce (fn [x block]
            (nn/forward block (assoc input :x x)))
          (:x input)
          blocks))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel ColumnEmbedding [model-dim heads feedforward-dim inducing-count block-count dropout-probability]
  [input-projection (nn/linear 1 model-dim)
   blocks (vec (repeatedly block-count
                           #(InducedAttentionBlock model-dim heads feedforward-dim
                                                   inducing-count dropout-probability)))
   weight-projection (nn/linear model-dim model-dim)
   bias-projection (nn/linear model-dim model-dim)
   weight-norm (nn/layernorm model-dim)
   bias-norm (nn/layernorm model-dim)
   dim model-dim]
  (forward [input]
           (let [{:keys [x train-size]} input
                 [batch-size sample-count feature-count] (t/size x)
                 columns (t/reshape (t/unsqueeze (t/transpose x 1 2) -1)
                                    [(* batch-size feature-count) sample-count 1])
                 encoded (run-blocks blocks {:x (nn/forward input-projection columns)
                                             :train-size train-size})
                 weights (nn/forward weight-norm (nn/forward weight-projection encoded))
                 biases (nn/forward bias-norm (nn/forward bias-projection encoded))
                 embedded (t/add (t/mul columns weights) biases)]
             (t/permute (t/reshape embedded
                                   [batch-size feature-count sample-count dim])
                        [0 2 1 3]))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel RowInteraction [model-dim heads feedforward-dim class-token-count block-count dropout-probability]
  [class-tokens (let [parameter (nn/parameter (t/empty [class-token-count model-dim]))]
                  (init/trunc-normal! (:tensor parameter) :std 0.02)
                  parameter)
   blocks (vec (repeatedly block-count
                           #(initialized-transformer-block
                             model-dim heads feedforward-dim dropout-probability)))
   output-norm (nn/layernorm model-dim)
   cls-count class-token-count
   dim model-dim
   head-count heads]
  (forward [input]
           (let [{:keys [x feature-counts]} (if (map? input) input {:x input})
                 [batch-size sample-count feature-count _] (t/size x)
                 flattened-batch (* batch-size sample-count)
                 features (t/reshape x [flattened-batch feature-count dim])
                 cls (t/expand (t/unsqueeze (:tensor class-tokens) 0)
                               [flattened-batch cls-count dim])
                 tokens (t/cat [cls features] 1)
                 attention-mask (when feature-counts
                                  (let [indices (t/reshape (t/arange (+ cls-count feature-count))
                                                           [1 1 1 (+ cls-count feature-count)])
                                        limits (t/reshape (t/add feature-counts cls-count)
                                                          [batch-size 1 1 1])]
                                    (t/reshape
                                     (t/expand (t/lt indices limits)
                                               [batch-size sample-count 1
                                                (+ cls-count feature-count)])
                                     [flattened-batch 1 1
                                      (+ cls-count feature-count)])))
                 rope (rope-frequencies (quot dim head-count) (+ cls-count feature-count))
                 prefix-blocks (butlast blocks)
                 encoded (reduce (fn [state block]
                                   (nn/forward block {:x state
                                                      :rope rope
                                                      :attention-mask attention-mask}))
                                 tokens
                                 prefix-blocks)
                 cls-output (nn/forward (last blocks)
                                        {:x (t/ix encoded :_ [0 cls-count] :_)
                                         :key encoded
                                         :value encoded
                                         :rope rope
                                         :attention-mask attention-mask})
                 cls-output (nn/forward output-norm cls-output)]
             (t/reshape cls-output
                        [batch-size sample-count (* cls-count dim)]))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel InContextLearning [model-dim heads feedforward-dim block-count max-classes dropout-probability]
  [label-embeddings (initialized-label-weight max-classes model-dim)
   label-bias (initialized-label-bias max-classes model-dim)
   blocks (vec (repeatedly block-count
                           #(initialized-transformer-block
                             model-dim heads feedforward-dim dropout-probability)))
   output-norm (nn/layernorm model-dim)
   decoder [(nn/linear model-dim (* 2 model-dim))
            (nn/gelu)
            (nn/linear (* 2 model-dim) max-classes)]
   dim model-dim]
  (forward [input]
           (let [{:keys [rows y-train]} input
                 [batch-size sample-count _] (t/size rows)
                 train-size (second (t/size y-train))
                 flat-labels (t/reshape y-train [(* batch-size train-size)])
                 label-vectors (t/add
                                (t/reshape
                                 (t/index-select (:tensor label-embeddings) 0 flat-labels)
                                 [batch-size train-size dim])
                                (:tensor label-bias))
                 training-rows (t/add (t/ix rows :_ [0 train-size] :_) label-vectors)
                 source (t/cat [training-rows (t/ix rows :_ [train-size sample-count] :_)] 1)
                 encoded (reduce (fn [state block]
                                   (let [context (t/ix state :_ [0 train-size] :_)]
                                     (nn/forward block {:x state
                                                        :key context
                                                        :value context})))
                                 source
                                 blocks)
                 logits (nn/forward decoder (nn/forward output-norm encoded))]
             (t/ix logits :_ [train-size sample-count] :_))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel TabICL [config]
  [column-embedding (ColumnEmbedding (:embedding-dim config)
                                     (:column-heads config)
                                     (* (:embedding-dim config) (:feedforward-factor config))
                                     (:inducing-points config)
                                     (:column-blocks config)
                                     (:dropout config))
   row-interaction (RowInteraction (:embedding-dim config)
                                   (:row-heads config)
                                   (* (:embedding-dim config) (:feedforward-factor config))
                                   (:class-tokens config)
                                   (:row-blocks config)
                                   (:dropout config))
   icl (InContextLearning (* (:embedding-dim config) (:class-tokens config))
                          (:icl-heads config)
                          (* (:embedding-dim config)
                             (:class-tokens config)
                             (:feedforward-factor config))
                          (:icl-blocks config)
                          (:max-classes config)
                          (:dropout config))]
  (forward [input]
           (let [{:keys [x y-train feature-counts]} input
                 train-size (second (t/size y-train))
                 columns (nn/forward column-embedding {:x x :train-size train-size})
                 rows (nn/forward row-interaction {:x columns
                                                   :feature-counts feature-counts})]
             (nn/forward icl {:rows rows :y-train y-train}))))

(defn tabicl
  "Constructs a TabICLv1 model from a configuration map."
  [config]
  (TabICL config))

(defn- column-input [column-embedding x]
  (let [[batch-size sample-count feature-count] (t/size x)
        columns (t/reshape (t/unsqueeze (t/transpose x 1 2) -1)
                           [(* batch-size feature-count) sample-count 1])]
    {:batch-size batch-size
     :sample-count sample-count
     :feature-count feature-count
     :columns columns
     :state (nn/forward (:input-projection column-embedding) columns)}))

(defn- finish-columns
  [column-embedding {:keys [batch-size sample-count feature-count columns]} state]
  (let [weights (nn/forward (:weight-norm column-embedding)
                            (nn/forward (:weight-projection column-embedding) state))
        biases (nn/forward (:bias-norm column-embedding)
                           (nn/forward (:bias-projection column-embedding) state))
        embedded (t/add (t/mul columns weights) biases)]
    (t/permute (t/reshape embedded
                          [batch-size feature-count sample-count (:dim column-embedding)])
               [0 2 1 3])))

(defn- prepare-column-cache
  [column-embedding x-train]
  (let [{:keys [state] :as input} (column-input column-embedding x-train)
        [state hidden]
        (reduce (fn [[state hidden] block]
                  (let [batch-size (first (t/size state))
                        inducing (t/expand
                                  (t/unsqueeze (:tensor (:inducing-points block)) 0)
                                  [batch-size (:num-inducing block) (:dim block)])
                        block-hidden (nn/forward (:inducing-attention block)
                                                 {:x inducing
                                                  :key state
                                                  :value state})
                        output (nn/forward (:input-attention block)
                                           {:x state
                                            :key block-hidden
                                            :value block-hidden})]
                    [output (conj hidden block-hidden)]))
                [state []]
                (:blocks column-embedding))]
    {:hidden hidden
     :training-columns (finish-columns column-embedding input state)}))

(defn- cached-columns
  [column-embedding x-test hidden]
  (let [{:keys [state] :as input} (column-input column-embedding x-test)
        state (reduce (fn [state [block block-hidden]]
                        (nn/forward (:input-attention block)
                                    {:x state
                                     :key block-hidden
                                     :value block-hidden}))
                      state
                      (map vector (:blocks column-embedding) hidden))]
    (finish-columns column-embedding input state)))

(defn- add-labels [icl rows y-train]
  (let [[batch-size train-size _] (t/size rows)
        labels (t/reshape y-train [(* batch-size train-size)])
        vectors (t/add
                 (t/reshape
                  (t/index-select (:tensor (:label-embeddings icl)) 0 labels)
                  [batch-size train-size (:dim icl)])
                 (:tensor (:label-bias icl)))]
    (t/add rows vectors)))

(defn- prepare-icl-cache
  [icl training-rows y-train]
  (let [source (add-labels icl training-rows y-train)
        [state entries]
        (reduce (fn [[state entries] block]
                  (let [result (nn/forward block {:x state :return-kv? true})]
                    [(:output result)
                     (conj entries {:key (:key result) :value (:value result)})]))
                [source []]
                (:blocks icl))]
    {:entries entries :training-state state}))

(defn prepare-cache
  "Precomputes the train-only column and ICL states reused across test batches."
  [model x-train y-train]
  (t/with-torch
    (let [column-cache (prepare-column-cache (:column-embedding model) x-train)
          training-rows (nn/forward (:row-interaction model)
                                    (:training-columns column-cache))
          icl-cache (prepare-icl-cache (:icl model) training-rows y-train)]
      {:column-hidden (:hidden column-cache)
       :icl-entries (:entries icl-cache)})))

(defn forward-with-cache
  "Runs test rows against a cache returned by prepare-cache."
  [model cache x-test]
  (let [columns (cached-columns (:column-embedding model)
                                x-test
                                (:column-hidden cache))
        rows (nn/forward (:row-interaction model) columns)
        icl (:icl model)
        state (reduce (fn [state [block entry]]
                        (nn/forward block {:x state
                                           :cached-key (:key entry)
                                           :cached-value (:value entry)}))
                      rows
                      (map vector (:blocks icl) (:icl-entries cache)))]
    (nn/forward (:decoder icl) (nn/forward (:output-norm icl) state))))

(def paper-config
  "The architecture dimensions reported in the TabICLv1 paper."
  {:embedding-dim 128
   :column-heads 4
   :column-blocks 3
   :inducing-points 128
   :row-heads 8
   :row-blocks 3
   :class-tokens 4
   :icl-heads 4
   :icl-blocks 12
   :feedforward-factor 2
   :max-classes 10
   :dropout 0.0})

(defn- block-destinations
  [destination-prefix tail tensor]
  (cond
    (= tail "attn.in_proj_weight")
    {(str destination-prefix ".attention.in-projection.weight") tensor}

    (= tail "attn.in_proj_bias")
    {(str destination-prefix ".attention.in-projection.bias") tensor}

    (str/starts-with? tail "attn.out_proj.")
    {(str destination-prefix ".attention.out-projection."
          (subs tail (count "attn.out_proj."))) tensor}

    (re-matches #"norm[12]\.(weight|bias)" tail)
    (let [[_ norm-index parameter] (re-matches #"norm([12])\.(weight|bias)" tail)]
      {(str destination-prefix ".norm" norm-index "."
            (if (= parameter "weight") "gamma" "beta")) tensor})

    (re-matches #"linear[12]\.(weight|bias)" tail)
    (let [[_ linear-index parameter] (re-matches #"linear([12])\.(weight|bias)" tail)
          vector-index (if (= linear-index "1") 0 3)]
      {(str destination-prefix ".feedforward[" vector-index "]." parameter) tensor})

    :else
    (throw (ex-info "Unknown TabICLv1 transformer parameter" {:tail tail}))))

(defn- checkpoint-destinations
  [source-name tensor]
  (if-let [[_ block-index attention-index tail]
           (re-matches #"col_embedder\.tf_col\.blocks\.(\d+)\.multihead_attn([12])\.(.+)"
                       source-name)]
    (block-destinations
     (str "column-embedding.blocks[" block-index "]."
          (if (= attention-index "1") "inducing-attention" "input-attention"))
     tail tensor)
    (if-let [[_ section block-index tail]
             (re-matches #"(row_interactor\.tf_row|icl_predictor\.tf_icl)\.blocks\.(\d+)\.(.+)"
                         source-name)]
      (block-destinations
       (str (if (= section "row_interactor.tf_row") "row-interaction.blocks[" "icl.blocks[")
            block-index "]")
       tail tensor)
      (cond
        (= source-name "col_embedder.in_linear.weight")
        {"column-embedding.input-projection.weight" tensor}
        (= source-name "col_embedder.in_linear.bias")
        {"column-embedding.input-projection.bias" tensor}
        (= source-name "col_embedder.out_w.weight")
        {"column-embedding.weight-projection.weight" tensor}
        (= source-name "col_embedder.out_w.bias")
        {"column-embedding.weight-projection.bias" tensor}
        (= source-name "col_embedder.out_b.weight")
        {"column-embedding.bias-projection.weight" tensor}
        (= source-name "col_embedder.out_b.bias")
        {"column-embedding.bias-projection.bias" tensor}
        (= source-name "col_embedder.ln_w.weight")
        {"column-embedding.weight-norm.gamma" tensor}
        (= source-name "col_embedder.ln_w.bias")
        {"column-embedding.weight-norm.beta" tensor}
        (= source-name "col_embedder.ln_b.weight")
        {"column-embedding.bias-norm.gamma" tensor}
        (= source-name "col_embedder.ln_b.bias")
        {"column-embedding.bias-norm.beta" tensor}
        (re-matches #"col_embedder\.tf_col\.blocks\.\d+\.ind_vectors" source-name)
        (let [[_ block-index] (re-matches #"col_embedder\.tf_col\.blocks\.(\d+)\.ind_vectors"
                                          source-name)]
          {(str "column-embedding.blocks[" block-index "].inducing-points") tensor})
        (= source-name "row_interactor.cls_tokens")
        {"row-interaction.class-tokens" tensor}
        (= source-name "row_interactor.out_ln.weight")
        {"row-interaction.output-norm.gamma" tensor}
        (= source-name "row_interactor.out_ln.bias")
        {"row-interaction.output-norm.beta" tensor}
        (= source-name "icl_predictor.y_encoder.weight")
        {"icl.label-embeddings" (t/transpose tensor 0 1)}
        (= source-name "icl_predictor.y_encoder.bias")
        {"icl.label-bias" tensor}
        (= source-name "icl_predictor.ln.weight")
        {"icl.output-norm.gamma" tensor}
        (= source-name "icl_predictor.ln.bias")
        {"icl.output-norm.beta" tensor}
        (re-matches #"icl_predictor\.decoder\.[02]\.(weight|bias)" source-name)
        (let [[_ layer parameter]
              (re-matches #"icl_predictor\.decoder\.([02])\.(weight|bias)" source-name)
              vector-index (if (= layer "0") 0 2)]
          {(str "icl.decoder[" vector-index "]." parameter) tensor})
        :else
        (throw (ex-info "Unknown TabICLv1 checkpoint parameter"
                        {:source-name source-name}))))))

(defn load-checkpoint!
  "Loads a safetensors archive containing the official TabICLv1 state dict."
  [model path]
  (let [source-state (safetensors/load-file path)
        values (into {}
                     (mapcat (fn [[source-name source]]
                               (checkpoint-destinations source-name source)))
                     source-state)]
    (let [expected (nn/flat-parameter-names model)
          supplied (set (keys values))
          missing (set/difference expected supplied)
          unexpected (set/difference supplied expected)]
      (when (or (seq missing) (seq unexpected))
        (throw (ex-info "Checkpoint and model parameters differ"
                        {:missing (sort missing)
                         :unexpected (sort unexpected)})))
      (let [loaded (nn/load-flat-state-dict! model values)]
        (when-not (= supplied loaded)
          (throw (ex-info "Checkpoint loading was incomplete"
                          {:not-loaded (sort (set/difference supplied loaded))})))))
    model))

(def smoke-config
  "A small configuration that exercises every architectural stage on CPU."
  (assoc paper-config
         :embedding-dim 16
         :column-heads 2
         :column-blocks 1
         :inducing-points 4
         :row-heads 2
         :row-blocks 2
         :class-tokens 2
         :icl-heads 2
         :icl-blocks 2
         :max-classes 3))

(defn smoke []
  (t/with-torch
    (t/manual-seed 7)
    (let [model (TabICL smoke-config)
          x (t/randn [2 7 5])
          y-train (t/tensor [[0 1 2 0] [2 1 0 2]] {:dtype :int64})
          logits (nn/forward model {:x x :y-train y-train})]
      {:shape (t/size logits)
       :parameters (count (nn/named-parameters model))
       :finite? (= 1.0 (double (t/item-float (t/to (t/all (t/isfinite logits)) :float32))))})))

(defn checkpoint-smoke
  "Compares the converted official checkpoint against PyTorch reference logits."
  [path]
  (let [model (load-checkpoint! (TabICL paper-config) path)]
    (t/with-torch
      (let [x (t/sub (t/div (t/reshape (t/arange 18) [1 6 3]) 10.0) 0.5)
            y-train (t/tensor [[0 1 2 1]] {:dtype :int64})
            expected (t/tensor
                      [[[1.323611021 2.527421713 1.918957949 -12.523500443
                         -19.382902145 -24.167066574 -27.598791122 -23.331361771
                         -28.479331970 -28.099742889]
                        [1.079133272 1.911582232 1.663178205 -3.163845062
                         -10.601795197 -15.732123375 -19.533498764 -16.520948410
                         -22.719383240 -22.619216919]]])
            actual (nn/forward model {:x x :y-train y-train})
            max-error (t/item-float (t/max (t/abs (t/sub actual expected))))]
        {:shape (t/size actual)
         :max-absolute-error max-error
         :matches? (< max-error 2.0e-5)}))))
