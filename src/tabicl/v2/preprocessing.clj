(ns tabicl.v2.preprocessing
  (:require [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml.preprocessing :as preprocessing]
            [tablecloth.api :as tc]))

(set! *warn-on-reflection* true)

(defn- transpose [rows]
  (apply mapv vector rows))

(defn- mean [values]
  (/ (reduce + 0.0 values) (double (count values))))

(defn- variance [values center sample?]
  (let [n (count values)
        divisor (if (and sample? (> n 1)) (dec n) n)]
    (/ (reduce (fn [sum value]
                 (let [delta (- (double value) center)]
                   (+ sum (* delta delta))))
               0.0
               values)
       (double (max 1 divisor)))))

(defn- yeo-johnson [value lambda]
  (let [value (double value)
        lambda (double lambda)]
    (if (>= value 0.0)
      (if (< (Math/abs lambda) 1.0e-8)
        (Math/log1p value)
        (/ (- (Math/pow (inc value) lambda) 1.0) lambda))
      (if (< (Math/abs (- lambda 2.0)) 1.0e-8)
        (- (Math/log1p (- value)))
        (- (/ (- (Math/pow (- 1.0 value) (- 2.0 lambda)) 1.0)
              (- 2.0 lambda)))))))

(defn- power-objective [values lambda]
  (let [transformed (mapv #(yeo-johnson % lambda) values)
        center (mean transformed)
        var (max 1.0e-24 (variance transformed center false))
        jacobian (* (- lambda 1.0)
                    (reduce + 0.0
                            (map (fn [value]
                                   (* (Math/signum (double value))
                                      (Math/log1p (Math/abs (double value)))))
                                 values)))]
    (- (* 0.5 (count values) (Math/log var)) jacobian)))

(defn- optimize-power [values]
  (let [ratio (/ (inc (Math/sqrt 5.0)) 2.0)]
    (loop [lo -2.0
           hi 2.0
           left (- 2.0 (/ 4.0 ratio))
           right (+ -2.0 (/ 4.0 ratio))
           iteration 0]
      (if (= iteration 80)
        (/ (+ lo hi) 2.0)
        (if (< (power-objective values left) (power-objective values right))
          (let [next-left (- right (/ (- right lo) ratio))]
            (recur lo right next-left left (inc iteration)))
          (let [next-right (+ left (/ (- hi left) ratio))]
            (recur left hi right next-right (inc iteration))))))))

(defn- rows->dataset [rows columns]
  (tc/dataset rows {:layout :as-row :column-names columns}))

(defn- dataset->rows [dataset]
  (mapv (fn [row] (mapv double row))
        (tc/rows dataset :as-vectors)))

(defn- power-transform [rows lambdas]
  (mapv #(mapv yeo-johnson % lambdas) rows))

(defn- power-step [enabled? columns]
  (fn [{:metamorph/keys [data id mode] :as context}]
    (if-not enabled?
      context
      (case mode
        :fit
        (let [rows (dataset->rows data)
              lambdas (mapv optimize-power (transpose rows))]
          (assoc context
                 id {:lambdas lambdas}
                 :metamorph/data (rows->dataset (power-transform rows lambdas) columns)))

        :transform
        (let [lambdas (get-in context [id :lambdas])]
          (assoc context :metamorph/data
                 (rows->dataset (power-transform (dataset->rows data) lambdas)
                                columns)))))))

(defn- outlier-bounds [rows threshold]
  (let [columns (transpose rows)
        initial-means (mapv mean columns)
        initial-stds (mapv (fn [column center]
                             (max 1.0e-6 (Math/sqrt (variance column center true))))
                           columns initial-means)
        clean-columns (mapv (fn [column center std]
                              (filterv #(<= (- center (* threshold std))
                                            %
                                            (+ center (* threshold std)))
                                       column))
                            columns initial-means initial-stds)
        clean-means (mapv mean clean-columns)
        clean-stds (mapv (fn [column center]
                           (max 1.0e-6 (Math/sqrt (variance column center true))))
                         clean-columns clean-means)]
    {:lower (mapv #(- %1 (* threshold %2)) clean-means clean-stds)
     :upper (mapv #(+ %1 (* threshold %2)) clean-means clean-stds)}))

(defn- remove-outliers [rows lower upper]
  (mapv (fn [row]
          (mapv (fn [value lo hi]
                  (let [value (double value)
                        adjustment (Math/log1p (Math/abs value))]
                    (-> value
                        (max (- lo adjustment))
                        (min (+ hi adjustment)))))
                row lower upper))
        rows))

(defn- outlier-step [threshold columns]
  (fn [{:metamorph/keys [data id mode] :as context}]
    (case mode
      :fit
      (let [rows (dataset->rows data)
            {:keys [lower upper] :as bounds} (outlier-bounds rows threshold)]
        (assoc context
               id bounds
               :metamorph/data (rows->dataset (remove-outliers rows lower upper)
                                              columns)))

      :transform
      (let [{:keys [lower upper]} (get context id)]
        (assoc context :metamorph/data
               (rows->dataset (remove-outliers (dataset->rows data) lower upper)
                              columns))))))

(defn fit-preprocessor
  "Fits a Noj/Metamorph feature pipeline on context rows only.

  The pipeline uses Noj standard scaling, optional Yeo-Johnson normalization,
  and TabICLv2's soft outlier bounds. Its fitted context is reused for query
  and inference transforms."
  [rows method outlier-threshold]
  (let [columns (vec (range (count (first rows))))
        pipeline (mm/pipeline
                  (preprocessing/std-scale columns {})
                  (power-step (= method :power) columns)
                  (outlier-step outlier-threshold columns))
        fitted-context (pipeline {:metamorph/data (rows->dataset rows columns)
                                  :metamorph/mode :fit})]
    {:pipeline pipeline
     :columns columns
     :fitted-context (dissoc fitted-context :metamorph/data)}))

(defn transform
  "Applies a fitted Noj/Metamorph feature pipeline."
  [{:keys [pipeline columns fitted-context]} rows]
  (-> (pipeline (assoc fitted-context
                       :metamorph/data (rows->dataset rows columns)
                       :metamorph/mode :transform))
      :metamorph/data
      dataset->rows))

(defn fit-ensemble
  "Fits per-estimator Noj preprocessing pipelines and shuffle configurations."
  [x-context y-context & {:keys [classification? n-estimators norm-methods outlier-threshold]
                          :or {classification? false
                               n-estimators 2
                               norm-methods [:power :none]
                               outlier-threshold 4.0}}]
  (let [feature-count (count (first x-context))
        class-count (when classification? (inc (apply max (map long y-context))))
        base-feature-order (vec (reverse (range feature-count)))
        base-class-order (when classification? (vec (range class-count)))]
    (mapv (fn [index]
            (let [method (nth norm-methods (mod index (count norm-methods)))]
              {:preprocessor (fit-preprocessor x-context method outlier-threshold)
               :feature-order base-feature-order
               :class-order base-class-order}))
          (range n-estimators))))

(defn transform-variant
  "Transforms rows and labels with one fitted ensemble configuration."
  [configuration rows labels]
  (let [{:keys [preprocessor feature-order class-order]} configuration
        transformed (transform preprocessor rows)]
    {:x (mapv (fn [row] (mapv row feature-order)) transformed)
     :y (if class-order
          (mapv #(nth class-order (long %)) labels)
          (mapv double labels))}))
