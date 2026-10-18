(ns tabicl.noj
  "Metamorph.ml model registrations for pretrained TabICLv2 classification and regression."
  (:require [clorch.cuda :as cuda]
            [clorch.nn :as nn]
            [scicloj.metamorph.ml :as ml]
            [tabicl.v2.finetune :as finetune]
            [tabicl.v2.model :as v2]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]))

(set! *warn-on-reflection* true)

(def classifier-model-type :tabicl/classifier)
(def regressor-model-type :tabicl/regressor)

(defn- rows [dataset]
  (mapv (fn [row] (mapv double row))
        (tc/rows dataset :as-vectors)))

(defn- single-target [target-dataset]
  (let [names (vec (ds/column-names target-dataset))]
    (when-not (= 1 (count names))
      (throw (ex-info "TabICL requires exactly one inference target"
                      {:target-columns names})))
    (let [name (first names)
          column (get target-dataset name)]
      {:name name
       :values (vec column)
       :metadata (meta column)})))

(defn- resolved-device [options]
  (or (:device options)
      (if (cuda/available?) :cuda :cpu)))

(defn- loaded-model [options]
  (let [device (resolved-device options)
        tabicl (or (:model options)
                   (when-let [checkpoint (:checkpoint options)]
                     (v2/load-converted-model checkpoint))
                   (throw (ex-info "TabICL requires :checkpoint or a preloaded :model"
                                   {:model-type (:model-type options)})))]
    {:model (nn/to tabicl device)
     :device device}))

(defn- prediction-options [options device]
  (merge {:device device}
         (select-keys options [:n-estimators :norm-methods
                               :outlier-threshold :temperature])))

(defn- train-classifier [feature-dataset target-dataset options]
  (let [{target-name :name
         target-values :values
         target-metadata :metadata} (single-target target-dataset)
        classes (vec (distinct target-values))
        class->index (zipmap classes (range))
        {tabicl :model device :device} (loaded-model options)
        max-classes (get-in tabicl [:model-config :max-classes])]
    (when-not (pos? (long (or max-classes 0)))
      (throw (ex-info "The supplied checkpoint is not a classifier" {})))
    (when (> (count classes) max-classes)
      (throw (ex-info "The target has more classes than this checkpoint supports"
                      {:classes (count classes) :maximum max-classes})))
    {:model tabicl
     :device device
     :x-context (rows feature-dataset)
     :y-context (mapv class->index target-values)
     :classes classes
     :target-name target-name
     :target-metadata target-metadata
     :prediction-options (prediction-options options device)}))

(defn- classification-dataset [probabilities classes target-name target-metadata]
  (let [columns (into {}
                      (map-indexed
                       (fn [class-index class]
                         [class (mapv #(nth % class-index) probabilities)])
                       classes))
        predictions (mapv (fn [probability]
                            (nth classes
                                 (apply max-key #(nth probability %) (range (count classes)))))
                          probabilities)]
    (-> (ds/->dataset (assoc columns target-name predictions))
        (ds/update-columnwise classes vary-meta assoc
                              :column-type :probability-distribution)
        (ds/update-column target-name
                          #(with-meta % (assoc target-metadata
                                               :column-type :prediction)))
        (vary-meta assoc :model-type :classification))))

(defn- predict-classifier [feature-dataset model-data _]
  (let [{:keys [model device x-context y-context classes target-name
                target-metadata prediction-options]} model-data
        probabilities (finetune/predict model x-context y-context
                                        (rows feature-dataset) true
                                        (assoc prediction-options :device device))]
    (classification-dataset probabilities classes target-name target-metadata)))

(defn- train-regressor [feature-dataset target-dataset options]
  (let [{target-name :name
         target-values :values
         target-metadata :metadata} (single-target target-dataset)
        {tabicl :model device :device} (loaded-model options)
        max-classes (get-in tabicl [:model-config :max-classes])]
    (when-not (zero? (long (or max-classes -1)))
      (throw (ex-info "The supplied checkpoint is not a regressor" {})))
    {:model tabicl
     :device device
     :x-context (rows feature-dataset)
     :y-context (mapv double target-values)
     :target-name target-name
     :target-metadata target-metadata
     :prediction-options (prediction-options options device)}))

(defn- predict-regressor [feature-dataset model-data _]
  (let [{:keys [model device x-context y-context target-name
                target-metadata prediction-options]} model-data
        predictions (finetune/predict model x-context y-context
                                      (rows feature-dataset) false
                                      (assoc prediction-options :device device))]
    (-> (ds/->dataset {target-name predictions})
        (ds/update-column target-name
                          #(with-meta % (assoc target-metadata
                                               :column-type :prediction)))
        (vary-meta assoc :model-type :regression))))

(def ^:private common-documentation
  {:user-guide "https://github.com/antlobach/tabicl-clorch"
   :code-example
   "(ml/model {:model-type :tabicl/classifier :checkpoint \"model.safetensors\"})"})

(ml/define-model! classifier-model-type
  train-classifier
  predict-classifier
  {:documentation common-documentation})

(ml/define-model! regressor-model-type
  train-regressor
  predict-regressor
  {:documentation common-documentation})
