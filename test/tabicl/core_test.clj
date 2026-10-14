(ns tabicl.core-test
  (:require [clojure.test :refer [deftest is]]
            [clorch.nn :as nn]
            [clorch.torch :as t]
            [tabicl.core :as tabicl]))

(set! *warn-on-reflection* true)

(defn- max-error [left right]
  (t/item-float (t/max (t/abs (t/sub left right)))))

(deftest cached-forward-matches-full-forward
  (t/manual-seed 11)
  (let [model (tabicl/tabicl tabicl/smoke-config)
        x (t/randn [1 7 3])
        x-train (t/ix x :_ [0 4] :_)
        x-test (t/ix x :_ [4 7] :_)
        y-train (t/tensor [[0 1 2 1]] {:dtype :int64})
        cache (tabicl/prepare-cache model x-train y-train)]
    (t/with-torch
      (let [full (nn/forward model {:x x :y-train y-train})
            cached (tabicl/forward-with-cache model cache x-test)]
        (is (= [1 3 3] (t/size cached)))
        (is (< (max-error full cached) 1.0e-4))))))

(deftest mixed-width-mask-ignores-padded-features
  (t/manual-seed 13)
  (let [model (tabicl/tabicl tabicl/smoke-config)]
    (t/with-torch
      (let [wide (t/randn [1 7 3])
            narrow (t/ix wide :_ :_ [0 2])
            padded (t/cat [narrow (t/full [1 7 1] 999.0)] 2)
            batch (t/cat [wide padded] 0)
            y-train (t/tensor [[0 1 2 1] [0 1 2 1]] {:dtype :int64})
            batched (nn/forward model
                                {:x batch
                                 :y-train y-train
                                 :feature-counts (t/tensor [3 2] {:dtype :int64})})
            separate (nn/forward model
                                 {:x narrow
                                  :y-train (t/ix y-train [0 1] :_)})]
        (is (< (max-error (t/ix batched 1 :_ :_)
                          (t/ix separate 0 :_ :_))
               1.0e-4))))))
