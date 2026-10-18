(ns tabicl.noj-test
  (:require [clojure.test :refer [deftest is]]
            [clorch.nn :as nn]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [tabicl.noj :as tabicl-noj]
            [tabicl.v2-finetune-test :refer [tiny-config]]
            [tabicl.v2.model :as v2]
            [tablecloth.api :as tc]
            [tech.v3.dataset.column-filters :as column-filters]
            [tech.v3.dataset.modelling :as modelling]))

(defn- fit-and-transform [model-options training-data query-data]
  (let [pipeline (mm/pipeline
                  {:metamorph/id :model}
                  (ml/model model-options))
        fitted (pipeline {:metamorph/data training-data
                          :metamorph/mode :fit})]
    (pipeline (assoc fitted
                     :metamorph/data query-data
                     :metamorph/mode :transform))))

(deftest model-types-register-on-require
  (is (contains? (set (ml/model-definition-names))
                 tabicl-noj/classifier-model-type))
  (is (contains? (set (ml/model-definition-names))
                 tabicl-noj/regressor-model-type)))

(deftest classifier-plugs-into-a-metamorph-pipeline
  (let [training (-> (tc/dataset {:x1 [0.0 1.0 -1.0 2.0]
                                  :x2 [1.0 0.0 0.5 -0.5]
                                  :label ["left" "right" "left" "right"]})
                     (modelling/set-inference-target :label))
        query (tc/dataset {:x1 [0.25 -0.75]
                           :x2 [0.75 0.25]})
        model (nn/to (v2/tabicl-v2 tiny-config) :cpu)
        result (fit-and-transform
                {:model-type :tabicl/classifier
                 :model model
                 :device :cpu
                 :n-estimators 2}
                training query)
        prediction-dataset (:metamorph/data result)]
    (is (= 2 (tc/row-count prediction-dataset)))
    (is (= #{"left" "right"}
           (set (:label prediction-dataset))))
    (is (= #{"left" "right"}
           (set (tc/column-names
                 (column-filters/probability-distribution prediction-dataset)))))
    (is (= [:label]
           (vec (tc/column-names
                 (column-filters/prediction prediction-dataset)))))))

(deftest regressor-plugs-into-a-metamorph-pipeline
  (let [config (assoc tiny-config
                      :max-classes 0
                      :bias-free-layernorm? true)
        training (-> (tc/dataset {:x [-1.0 0.0 1.0 2.0]
                                  :target [-0.8 0.0 0.9 0.4]})
                     (modelling/set-inference-target :target))
        query (tc/dataset {:x [0.5 1.5]})
        model (nn/to (v2/tabicl-v2 config) :cpu)
        result (fit-and-transform
                {:model-type :tabicl/regressor
                 :model model
                 :device :cpu
                 :n-estimators 2}
                training query)
        prediction-dataset (:metamorph/data result)]
    (is (= 2 (tc/row-count prediction-dataset)))
    (is (= [:target]
           (vec (tc/column-names
                 (column-filters/prediction prediction-dataset)))))
    (is (every? #(Double/isFinite (double %)) (:target prediction-dataset)))))
