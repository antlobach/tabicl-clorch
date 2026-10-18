(ns tabicl.examples.finetune-classifier
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clorch.nn :as nn]
            [tabicl.v2.finetune :as finetune]
            [tabicl.v2.model :as v2]))

(defn- tutorial-data []
  (json/read-str (slurp (io/resource "tabicl/examples/finetune_classifier_data.json"))))

(defn execute!
  "Replicates the official TabICLv2 classifier fine-tuning tutorial on CUDA."
  [checkpoint]
  (let [data (tutorial-data)
        x-train (get data "x-train")
        y-train (get data "y-train")
        x-validation (get data "x-validation")
        y-validation (get data "y-validation")
        x-test (get data "x-test")
        y-test (get data "y-test")
        model (nn/to (v2/load-converted-model checkpoint) :cuda)
        options {:epochs 30
                 :learning-rate 1.0e-5
                 :n-estimators 2
                 :early-stopping? true
                 :patience 5
                 :random-state 0
                 :device :cuda}
        pretrained (finetune/classification-metrics
                    (finetune/predict model x-train y-train x-test true options)
                    y-test)
        result (finetune/fine-tune! model x-train y-train
                                    x-validation y-validation true options)
        fine-tuned (finetune/classification-metrics
                    (finetune/predict model x-train y-train x-test true options)
                    y-test)]
    {:pretrained pretrained
     :fine-tuned fine-tuned
     :best-validation (:best-metric result)
     :epochs-run (:epochs-run result)}))

(defn -main [& [checkpoint]]
  (when-not checkpoint
    (throw (IllegalArgumentException.
            "Pass the converted TabICLv2 classifier .safetensors path")))
  (println (execute! checkpoint))
  (shutdown-agents))
