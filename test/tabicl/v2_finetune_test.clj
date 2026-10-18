(ns tabicl.v2-finetune-test
  (:require [clojure.test :refer [deftest is testing]]
            [clorch.nn :as nn]
            [tabicl.v2.finetune :as finetune]
            [tabicl.v2.model :as v2]
            [tabicl.v2.preprocessing :as preprocessing]))

(def tiny-config
  {:embed-dim 16
   :column-blocks 1
   :column-heads 2
   :column-inducing-points 4
   :feature-group :none
   :feature-group-size 1
   :target-aware? true
   :column-ssmax :none
   :row-blocks 1
   :row-heads 2
   :class-tokens 2
   :icl-blocks 1
   :icl-heads 2
   :icl-ssmax :none
   :feedforward-factor 2
   :max-classes 3
   :num-quantiles 9})

(deftest scheduler-and-chunk-count-match-upstream-contract
  (is (= 0.0 (finetune/cosine-warmup-factor 0 3 30)))
  (is (= 1.0 (finetune/cosine-warmup-factor 3 3 30)))
  (is (= 1 (finetune/count-chunks 40 10000 50)))
  (is (= 2 (finetune/count-chunks 205 100 50)))
  (is (= 3 (finetune/count-chunks 250 100 50))))

(deftest noj-pipeline-reuses-context-fitted-state
  (let [fitted (preprocessing/fit-preprocessor
                [[1.0 2.0] [2.0 4.0] [3.0 8.0]] :power 4.0)
        first-result (preprocessing/transform fitted [[2.0 4.0]])
        second-result (preprocessing/transform fitted [[2.0 4.0]])]
    (is (= first-result second-result))
    (is (= [1 2] [(count first-result) (count (first first-result))]))))

(deftest meta-batch-preserves-ensemble-context-query-contract
  (let [x (mapv (fn [index] [(double index) (double (mod index 3))]) (range 20))
        y (mapv #(mod % 2) (range 20))
        batch (finetune/meta-batch x y true 0 {:n-estimators 2})]
    (is (= 2 (count (:x batch))))
    (is (= 4 (count (first (:y-query batch)))))
    (is (= 16 (:train-size batch)))
    (is (= 20 (count (first (:x batch)))))))

(deftest classifier-fine-tuning-runs-and-restores-validation-state
  (let [x (mapv (fn [index]
                  [(double index) (Math/sin index) (double (mod index 3))])
                (range 20))
        y (mapv #(mod % 2) (range 20))
        model (nn/to (v2/tabicl-v2 tiny-config) :cpu)
        result (finetune/fine-tune! model
                                    (subvec x 0 16) (subvec y 0 16)
                                    (subvec x 16) (subvec y 16)
                                    true
                                    {:epochs 1
                                     :patience 1
                                     :amp? false
                                     :device :cpu})]
    (testing "one complete optimization and validation epoch ran"
      (is (= 1 (:epochs-run result)))
      (is (= 1 (count (:history result))))
      (is (number? (:best-metric result))))))
