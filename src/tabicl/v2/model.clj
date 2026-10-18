(ns tabicl.v2.model
  "Checkpoint-compatible TabICLv2 classifier and quantile regressor."
  (:require [clojure.data.json :as json]
            [clojure.set :as set]
            [clojure.string :as str]
            [clorch.nn :as nn]
            [clorch.nn.functional :as F]
            [clorch.nn.init :as init]
            [clorch.safetensors :as safetensors]
            [clorch.torch :as t]))

(set! *warn-on-reflection* true)

(def default-config
  {:max-classes 10
   :num-quantiles 999
   :embed-dim 128
   :column-blocks 3
   :column-heads 8
   :column-inducing-points 128
   :column-affine? false
   :feature-group :same
   :feature-group-size 3
   :target-aware? true
   :column-ssmax :qassmax-mlp-elementwise
   :row-blocks 3
   :row-heads 8
   :class-tokens 4
   :rope-base 100000.0
   :rope-interleaved? false
   :icl-blocks 12
   :icl-heads 8
   :icl-ssmax :qassmax-mlp-elementwise
   :feedforward-factor 2
   :dropout 0.0
   :norm-first? true
   :bias-free-layernorm? false})

(defn- linear-sequence [input layers]
  (reduce (fn [value layer] (nn/forward layer value)) input layers))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel OneHotLinear [class-count output-dim]
  [linear (nn/linear class-count output-dim)
   classes class-count]
  (forward [input]
           (nn/forward linear
                       (t/to (F/one-hot (t/to input :int64) classes) :float32))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel SkippableLinear [input-dim output-dim]
  [linear (nn/linear input-dim output-dim)
   skip-value -100.0]
  (forward [input]
           (let [output (nn/forward linear input)
                 skipped? (t/all (t/eq input skip-value) -1 :keepdim true)]
             (t/where (t/expand-as skipped? output)
                      (t/full-like output skip-value)
                      output))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel QASSMax [heads head-dim]
  [base-mlp [(nn/linear 1 64) (nn/gelu) (nn/linear 64 (* heads head-dim))]
   query-mlp (let [layers [(nn/linear head-dim 64)
                           (nn/gelu)
                           (nn/linear 64 head-dim)]]
               (init/zeros! (get (nn/named-parameters (nth layers 2)) "weight"))
               (init/zeros! (get (nn/named-parameters (nth layers 2)) "bias"))
               layers)
   head-count heads
   dimension head-dim]
  (forward [input]
           (let [{:keys [query source-length]} input
                 log-length (-> (t/tensor [[(Math/log (double (max source-length 1)))]])
                                (t/to (.device ^org.bytedeco.pytorch.Tensor (t/->tensor query)))
                                (t/type-as query))
                 base (t/reshape (linear-sequence log-length base-mlp)
                                 [1 head-count 1 dimension])
                 modulation (t/add 1.0 (t/tanh (linear-sequence query query-mlp)))]
             (t/mul query (t/mul base modulation)))))

(defn- initialized-projection [rows columns]
  (let [parameter (nn/parameter (t/empty [rows columns]))]
    (init/xavier-uniform! (:tensor parameter))
    parameter))

(declare attention-block-forward)

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel AttentionBlock [model-dim heads feedforward-dim config]
  [linear1 (nn/linear model-dim feedforward-dim)
   linear2 (nn/linear feedforward-dim model-dim)
   norm1 (nn/layernorm model-dim :bias? (not (:bias-free-layernorm? config)))
   norm2 (nn/layernorm model-dim :bias? (not (:bias-free-layernorm? config)))
   in-projection-weight (initialized-projection (* 3 model-dim) model-dim)
   in-projection-bias (nn/parameter (t/zeros [(* 3 model-dim)]))
   out-projection (nn/linear model-dim model-dim)
   ssmax-layer (when (= :qassmax-mlp-elementwise (:ssmax config))
                 (QASSMax heads (quot model-dim heads)))
   dimension model-dim
   head-count heads
   dropout (:dropout config)
   norm-first? (:norm-first? config)]
  (forward [input]
           (attention-block-forward
            {:linear1 linear1
             :linear2 linear2
             :norm1 norm1
             :norm2 norm2
             :in-projection-weight in-projection-weight
             :in-projection-bias in-projection-bias
             :out-projection out-projection
             :ssmax-layer ssmax-layer
             :dimension dimension
             :head-count head-count
             :dropout dropout
             :norm-first? norm-first?}
            input)))

(defn- packed-projections [block query key value]
  (let [dimension (:dimension block)
        weight (:tensor (:in-projection-weight block))
        bias (:tensor (:in-projection-bias block))]
    (cond
      (and (identical? query key) (identical? key value))
      (t/chunk (F/linear query weight bias) 3 -1)

      (identical? key value)
      (let [[query-weight kv-weight] (t/split weight [dimension (* 2 dimension)] 0)
            [query-bias kv-bias] (t/split bias [dimension (* 2 dimension)] 0)
            [key-projection value-projection]
            (t/chunk (F/linear key kv-weight kv-bias) 2 -1)]
        [(F/linear query query-weight query-bias) key-projection value-projection])

      :else
      (let [[query-weight key-weight value-weight] (t/chunk weight 3 0)
            [query-bias key-bias value-bias] (t/chunk bias 3 0)]
        [(F/linear query query-weight query-bias)
         (F/linear key key-weight key-bias)
         (F/linear value value-weight value-bias)]))))

(defn- split-heads [tensor heads]
  (let [shape (t/size tensor)
        batch-shape (vec (drop-last 2 shape))
        sequence-length (nth shape (- (count shape) 2))
        dimension (peek shape)
        flat-batch (reduce * 1 batch-shape)]
    {:batch-shape batch-shape
     :sequence-length sequence-length
     :tensor (t/transpose
              (t/reshape tensor [flat-batch sequence-length heads (quot dimension heads)])
              1 2)}))

(defn- apply-attention-rope [tensor interleaved? theta]
  (let [sequence-length (nth (t/size tensor) 2)
        head-dim (peek (t/size tensor))
        device (.device ^org.bytedeco.pytorch.Tensor (t/->tensor tensor))
        [cosines sines] (t/precompute-rope-freqs
                         head-dim sequence-length
                         :theta theta :interleaved? interleaved? :device device)]
    (-> tensor
        (t/transpose 1 2)
        (t/apply-rope cosines sines :interleaved? interleaved?)
        (t/transpose 1 2))))

(defn- multihead-attention [block query key value {:keys [attention-mask key-padding-mask rope]}]
  (let [[query key value] (packed-projections block query key value)
        query-info (split-heads query (:head-count block))
        key-info (split-heads key (:head-count block))
        value-info (split-heads value (:head-count block))
        query (if rope
                (apply-attention-rope (:tensor query-info)
                                      (:interleaved? rope) (:theta rope))
                (:tensor query-info))
        key (if rope
              (apply-attention-rope (:tensor key-info)
                                    (:interleaved? rope) (:theta rope))
              (:tensor key-info))
        value (:tensor value-info)
        query (if-some [ssmax (:ssmax-layer block)]
                (nn/forward ssmax {:query query
                                   :source-length (:sequence-length key-info)})
                query)
        mask (cond
               (and attention-mask key-padding-mask)
               (F/combine-attention-masks attention-mask key-padding-mask)

               key-padding-mask
               (t/logical-not
                (t/reshape key-padding-mask
                           [(first (t/size query)) 1 1 (:sequence-length key-info)]))

               :else attention-mask)
        attended (F/scaled-dot-product-attention
                  query key value
                  :attention-mask mask
                  :dropout-p 0.0)
        output (-> attended
                   (t/transpose 1 2)
                   (t/contiguous)
                   (t/reshape (conj (:batch-shape query-info)
                                    (:sequence-length query-info)
                                    (:dimension block))))]
    (nn/forward (:out-projection block) output)))

(defn attention-block-forward [block input]
  (let [{:keys [query key value train-size] :as input}
        (if (map? input) input {:query input})
        key (or key (if train-size (t/slice query -2 0 train-size 1) query))
        value (or value key)]
    (if (:norm-first? block)
      (let [query-normalized (nn/forward (:norm1 block) query)
            key-normalized (if (identical? query key)
                             query-normalized
                             (nn/forward (:norm1 block) key))
            value-normalized (if (identical? key value)
                               key-normalized
                               (nn/forward (:norm1 block) value))
            attention (multihead-attention block query-normalized key-normalized value-normalized input)
            residual (t/add query attention)
            feedforward (nn/forward (:linear2 block)
                                    (F/gelu (nn/forward (:linear1 block)
                                                        (nn/forward (:norm2 block) residual))))]
        (t/add residual feedforward))
      (let [attention (multihead-attention block query key value input)
            residual (nn/forward (:norm1 block) (t/add query attention))
            feedforward (nn/forward (:linear2 block)
                                    (F/gelu (nn/forward (:linear1 block) residual)))]
        (nn/forward (:norm2 block) (t/add residual feedforward))))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel InducedAttention [model-dim heads feedforward-dim inducing-count config]
  [attention1 (AttentionBlock model-dim heads feedforward-dim config)
   attention2 (AttentionBlock model-dim heads feedforward-dim (assoc config :ssmax nil))
   inducing-points (let [parameter (nn/parameter (t/empty [inducing-count model-dim]))]
                     (init/trunc-normal! (:tensor parameter) :std 0.02)
                     parameter)
   inducing-count inducing-count
   dimension model-dim
   skip-value -100.0]
  (forward [input]
           (let [{:keys [source train-size]} input
                 shape (t/size source)
                 batch-shape (vec (drop-last 2 shape))
                 inducing (t/expand (:tensor inducing-points)
                                    (conj batch-shape inducing-count dimension))
                 context (if train-size (t/slice source -2 0 train-size 1) source)
                 hidden (nn/forward attention1 {:query inducing :key context :value context})
                 output (nn/forward attention2 {:query source :key hidden :value hidden})
                 skipped? (-> source (t/eq skip-value) (t/all -1) (t/all -1))
                 skipped? (-> skipped? (t/unsqueeze -1) (t/unsqueeze -1)
                              (t/expand-as output))]
             (t/where skipped? (t/full-like output skip-value) output))))

(defn- run-induced-blocks [blocks source train-size]
  (reduce (fn [state block]
            (nn/forward block {:source state :train-size train-size}))
          source blocks))

(defn- feature-groups [x group-size]
  (let [[_ _ feature-count] (t/size x)
        device (.device ^org.bytedeco.pytorch.Tensor (t/->tensor x))]
    (t/stack
     (mapv (fn [offset]
             (let [indices (-> (t/tensor
                                (mapv #(mod (+ % (bit-shift-left 1 offset)) feature-count)
                                      (range feature-count))
                                {:dtype :int64})
                               (t/to device))]
               (t/index-select x 2 indices)))
           (range group-size))
     -1)))

(defn- target-embeddings [column y-train]
  (if (pos? (:max-classes column))
    (nn/forward (:y-encoder column) y-train)
    (nn/forward (:y-encoder column) (t/unsqueeze y-train -1))))

(defn- compute-column-embeddings [column features y-train]
  (let [train-size (last (t/size y-train))
        source (nn/forward (:input-linear column) features)
        y-embedding (target-embeddings column y-train)
        source (t/cat [(t/add (t/ix source :_ :_ [0 train-size] :_)
                              y-embedding)
                       (t/ix source :_ :_ [train-size nil] :_)]
                      -2)]
    (run-induced-blocks (:blocks column) source train-size)))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel ColumnEmbedder [config]
  [input-linear (SkippableLinear (:feature-group-size config) (:embed-dim config))
   blocks (vec (repeatedly (:column-blocks config)
                           #(InducedAttention (:embed-dim config)
                                              (:column-heads config)
                                              (* (:embed-dim config)
                                                 (:feedforward-factor config))
                                              (:column-inducing-points config)
                                              {:ssmax (:column-ssmax config)
                                               :dropout (:dropout config)
                                               :norm-first? (:norm-first? config)
                                               :bias-free-layernorm?
                                               (:bias-free-layernorm? config)})))
   y-encoder (if (pos? (:max-classes config))
               (OneHotLinear (:max-classes config) (:embed-dim config))
               (nn/linear 1 (:embed-dim config)))
   max-classes (:max-classes config)
   group-size (:feature-group-size config)
   reserve-count (:class-tokens config)]
  (forward [input]
           (let [{:keys [x y-train]} input
                 grouped (feature-groups x group-size)
                 grouped (F/pad grouped [0 0 reserve-count 0] :value -100.0)
                 features (t/transpose grouped 1 2)
                 labels (t/expand (t/unsqueeze y-train 1)
                                  [(first (t/size features))
                                   (second (t/size features))
                                   (second (t/size y-train))])]
             (t/transpose
              (compute-column-embeddings
               {:input-linear input-linear
                :blocks blocks
                :y-encoder y-encoder
                :max-classes max-classes}
               features labels)
              1 2))))

(defn- run-attention-blocks [blocks source options]
  (reduce (fn [state block]
            (nn/forward block (assoc options :query state)))
          source blocks))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel RowInteractor [config]
  [blocks (vec (repeatedly (:row-blocks config)
                           #(AttentionBlock (:embed-dim config)
                                            (:row-heads config)
                                            (* (:embed-dim config)
                                               (:feedforward-factor config))
                                            {:ssmax nil
                                             :dropout (:dropout config)
                                             :norm-first? (:norm-first? config)
                                             :bias-free-layernorm?
                                             (:bias-free-layernorm? config)})))
   class-token-values (let [parameter (nn/parameter
                                       (t/empty [(:class-tokens config)
                                                 (:embed-dim config)]))]
                        (init/trunc-normal! (:tensor parameter) :std 0.02)
                        parameter)
   output-norm (nn/layernorm (:embed-dim config)
                             :bias? (not (:bias-free-layernorm? config)))
   class-count (:class-tokens config)
   dimension (:embed-dim config)
   rope {:theta (:rope-base config)
         :interleaved? (:rope-interleaved? config)}]
  (forward [embeddings]
           (let [[batch-size sample-count token-count _] (t/size embeddings)
                 class-tokens (t/expand (:tensor class-token-values)
                                        [batch-size sample-count class-count dimension])
                 source (t/cat [class-tokens
                                (t/ix embeddings :_ :_ [class-count token-count] :_)]
                               2)
                 flat (t/reshape source
                                 [(* batch-size sample-count) token-count dimension])
                 prefix (run-attention-blocks (butlast blocks) flat {:rope rope})
                 classes (nn/forward (last blocks)
                                     {:query (t/ix prefix :_ [0 class-count] :_)
                                      :key prefix :value prefix :rope rope})]
             (-> (nn/forward output-norm classes)
                 (t/reshape [batch-size sample-count (* class-count dimension)])))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel ICLPredictor [config]
  [blocks (vec (repeatedly (:icl-blocks config)
                           #(AttentionBlock (* (:embed-dim config) (:class-tokens config))
                                            (:icl-heads config)
                                            (* (:embed-dim config)
                                               (:class-tokens config)
                                               (:feedforward-factor config))
                                            {:ssmax (:icl-ssmax config)
                                             :dropout (:dropout config)
                                             :norm-first? (:norm-first? config)
                                             :bias-free-layernorm?
                                             (:bias-free-layernorm? config)})))
   y-encoder (if (pos? (:max-classes config))
               (OneHotLinear (:max-classes config)
                             (* (:embed-dim config) (:class-tokens config)))
               (nn/linear 1 (* (:embed-dim config) (:class-tokens config))))
   output-norm (nn/layernorm (* (:embed-dim config) (:class-tokens config))
                             :bias? (not (:bias-free-layernorm? config)))
   decoder [(nn/linear (* (:embed-dim config) (:class-tokens config))
                       (* 2 (:embed-dim config) (:class-tokens config)))
            (nn/gelu)
            (nn/linear (* 2 (:embed-dim config) (:class-tokens config))
                       (if (pos? (:max-classes config))
                         (:max-classes config)
                         (:num-quantiles config)))]
   max-classes (:max-classes config)]
  (forward [input]
           (let [{:keys [representations y-train]} input
                 train-size (second (t/size y-train))
                 y-embedding (if (pos? max-classes)
                               (nn/forward y-encoder y-train)
                               (nn/forward y-encoder (t/unsqueeze y-train -1)))
                 source (t/cat [(t/add (t/ix representations :_ [0 train-size] :_)
                                       y-embedding)
                                (t/ix representations :_ [train-size nil] :_)]
                               1)
                 encoded (run-attention-blocks blocks source {:train-size train-size})]
             (linear-sequence (nn/forward output-norm encoded) decoder))))

#_{:clj-kondo/ignore [:unresolved-symbol]}
(nn/defmodel TabICLv2 [config]
  [column-embedder (ColumnEmbedder config)
   row-interactor (RowInteractor config)
   icl-predictor (ICLPredictor config)
   model-config config]
  (forward [input]
           (let [{:keys [x y-train]} input
                 columns (nn/forward column-embedder {:x x :y-train y-train})
                 rows (nn/forward row-interactor columns)]
             (nn/forward icl-predictor
                         {:representations rows :y-train y-train}))))

(defn tabicl-v2
  "Creates a TabICLv2 model. Pass `:max-classes 0` for quantile regression."
  ([] (tabicl-v2 {}))
  ([config]
   (let [config (merge default-config config)]
     (when (and (zero? (:max-classes config))
                (not (pos? (:num-quantiles config))))
       (throw (ex-info "Regression requires a positive quantile count" config)))
     (TabICLv2 config))))

(defn predict
  "Returns query logits for classification or raw quantiles for regression."
  [model x y-train]
  (let [train-size (second (t/size y-train))
        output (nn/forward model {:x x :y-train y-train})]
    (t/ix output :_ [train-size nil] :_)))

(defn predict-probabilities
  "Returns query probabilities for a classifier."
  [model x y-train & {:keys [temperature] :or {temperature 0.9}}]
  (let [class-count (count (t/tseq (t/unique (t/ix y-train 0 :_))))]
    (t/softmax (t/div (t/ix (predict model x y-train)
                            :_ :_ [0 class-count])
                      temperature)
               -1)))

(defn predict-mean
  "Returns the mean of raw query quantiles for a regressor."
  [model x y-train]
  (t/mean (predict model x y-train) -1))

(defn- checkpoint-name [source-name]
  (-> source-name
      (str/replace "col_embedder" "column-embedder")
      (str/replace "row_interactor" "row-interactor")
      (str/replace "icl_predictor" "icl-predictor")
      (str/replace #"tf_col\.blocks\.(\d+)" "blocks[$1]")
      (str/replace #"tf_row\.blocks\.(\d+)" "blocks[$1]")
      (str/replace #"tf_icl\.blocks\.(\d+)" "blocks[$1]")
      (str/replace "multihead_attn1" "attention1")
      (str/replace "multihead_attn2" "attention2")
      (str/replace "ind_vectors" "inducing-points")
      (str/replace "in_linear" "input-linear.linear")
      (str/replace "cls_tokens" "class-token-values")
      (str/replace "out_ln" "output-norm")
      (str/replace ".ln." ".output-norm.")
      (str/replace "y_encoder.linear" "y-encoder.linear")
      (str/replace "y_encoder" "y-encoder")
      (str/replace "attn.in_proj_weight" "in-projection-weight")
      (str/replace "attn.in_proj_bias" "in-projection-bias")
      (str/replace "attn.out_proj" "out-projection")
      (str/replace "attn.ssmax_layer" "ssmax-layer")
      (str/replace "base_mlp" "base-mlp")
      (str/replace "query_mlp" "query-mlp")
      (str/replace #"\.([02])\." "[$1].")
      (str/replace #"(norm[12]|output-norm)\.weight$" "$1.gamma")
      (str/replace #"(norm[12]|output-norm)\.bias$" "$1.beta")))

(defn load-checkpoint!
  "Loads a converted TabICLv2 SafeTensors checkpoint with strict key matching."
  [model path]
  (let [source (safetensors/load-file path)
        expected (nn/flat-parameter-names model)
        values (into {}
                     (map (fn [[name value]]
                            (let [mapped (checkpoint-name name)
                                  mapped (if (contains? expected mapped)
                                           mapped
                                           (str/replace mapped
                                                        ".y-encoder."
                                                        ".y-encoder.linear."))]
                              [mapped value])))
                     source)
        supplied (set (keys values))
        missing (set/difference expected supplied)
        unexpected (set/difference supplied expected)]
    (when (or (seq missing) (seq unexpected))
      (throw (ex-info "Checkpoint and TabICLv2 parameters differ"
                      {:missing (sort missing)
                       :unexpected (sort unexpected)})))
    (nn/load-flat-state-dict! model values)
    model))

(def ^:private checkpoint-config-keys
  {:max-classes :max-classes
   :num-quantiles :num-quantiles
   :embed-dim :embed-dim
   :col-num-blocks :column-blocks
   :col-nhead :column-heads
   :col-num-inds :column-inducing-points
   :col-affine :column-affine?
   :col-feature-group :feature-group
   :col-feature-group-size :feature-group-size
   :col-target-aware :target-aware?
   :col-ssmax :column-ssmax
   :row-num-blocks :row-blocks
   :row-nhead :row-heads
   :row-num-cls :class-tokens
   :row-rope-base :rope-base
   :row-rope-interleaved :rope-interleaved?
   :icl-num-blocks :icl-blocks
   :icl-nhead :icl-heads
   :icl-ssmax :icl-ssmax
   :ff-factor :feedforward-factor
   :dropout :dropout
   :norm-first :norm-first?
   :bias-free-ln :bias-free-layernorm?})

(defn- converted-config [path]
  (let [raw (json/read-str (slurp path) :key-fn #(keyword (str/replace % "_" "-")))]
    (reduce-kv (fn [config source-key destination-key]
                 (if-some [value (get raw source-key)]
                   (assoc config destination-key
                          (if (and (#{:feature-group :column-ssmax :icl-ssmax}
                                    destination-key)
                                   (string? value))
                            (keyword value)
                            value))
                   config))
               {}
               checkpoint-config-keys)))

(defn load-converted-model
  "Constructs and strict-loads a converted checkpoint.
  The converter writes the required JSON configuration beside the SafeTensors file."
  [safetensors-path]
  (let [config-path (str/replace (str safetensors-path) #"\.[^.]+$" ".json")
        model (tabicl-v2 (converted-config config-path))]
    (load-checkpoint! model safetensors-path)))
