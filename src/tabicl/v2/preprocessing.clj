(ns tabicl.v2.preprocessing)

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

(defn- clip [value lower upper]
  (max lower (min upper value)))

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

(defn fit-preprocessor
  "Fits TabICLv2's standardization, optional Yeo-Johnson transform, and outlier bounds."
  [rows method outlier-threshold]
  (let [columns (transpose rows)
        means (mapv mean columns)
        scales (mapv (fn [column center]
                       (+ (Math/sqrt (variance column center false)) 1.0e-6))
                     columns means)
        standardized (mapv (fn [row]
                             (mapv #(clip (/ (- (double %1) %2) %3) -100.0 100.0)
                                   row means scales))
                           rows)
        lambdas (when (= method :power)
                  (mapv optimize-power (transpose standardized)))
        normalized (if lambdas
                     (mapv #(mapv yeo-johnson % lambdas) standardized)
                     standardized)
        initial-columns (transpose normalized)
        initial-means (mapv mean initial-columns)
        initial-stds (mapv (fn [column center]
                             (max 1.0e-6 (Math/sqrt (variance column center true))))
                           initial-columns initial-means)
        clean-columns (mapv (fn [column center std]
                              (filterv #(<= (- center (* outlier-threshold std))
                                            %
                                            (+ center (* outlier-threshold std)))
                                       column))
                            initial-columns initial-means initial-stds)
        clean-means (mapv mean clean-columns)
        clean-stds (mapv (fn [column center]
                           (max 1.0e-6 (Math/sqrt (variance column center true))))
                         clean-columns clean-means)]
    {:method method
     :means means
     :scales scales
     :lambdas lambdas
     :lower (mapv #(- %1 (* outlier-threshold %2)) clean-means clean-stds)
     :upper (mapv #(+ %1 (* outlier-threshold %2)) clean-means clean-stds)}))

(defn transform
  "Applies a fitted TabICLv2 feature preprocessor."
  [preprocessor rows]
  (let [{:keys [means scales lambdas lower upper]} preprocessor]
    (mapv (fn [row]
            (mapv (fn [value center scale lambda lo hi]
                    (let [standardized (clip (/ (- (double value) center) scale)
                                             -100.0 100.0)
                          normalized (double (if lambda
                                               (yeo-johnson standardized lambda)
                                               standardized))]
                      (-> normalized
                          (max (+ lo (- (Math/log1p (Math/abs normalized)))))
                          (min (+ hi (Math/log1p (Math/abs normalized)))))))
                  row means scales (or lambdas (repeat nil)) lower upper))
          rows)))

(defn fit-ensemble
  "Fits the per-estimator feature preprocessing and shuffle configuration used by TabICLv2."
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
