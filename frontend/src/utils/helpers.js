// ==================== SCORE & METRIC HELPERS ====================

export function getScoreColor(score, higherBetter = true) {
  const s = higherBetter ? score : Math.max(0, 1 - score);
  if (s >= 0.9) return { bg: 'bg-emerald-50 dark:bg-emerald-950/30', border: 'border-emerald-300 dark:border-emerald-800', text: 'text-emerald-700 dark:text-emerald-400', label: 'Excellent' };
  if (s >= 0.7) return { bg: 'bg-sky-50 dark:bg-sky-950/30', border: 'border-sky-300 dark:border-sky-800', text: 'text-sky-700 dark:text-sky-400', label: 'Good' };
  if (s >= 0.5) return { bg: 'bg-amber-50 dark:bg-amber-950/30', border: 'border-amber-300 dark:border-amber-800', text: 'text-amber-700 dark:text-amber-400', label: 'Fair' };
  return { bg: 'bg-red-50 dark:bg-red-950/30', border: 'border-red-300 dark:border-red-800', text: 'text-red-700 dark:text-red-400', label: 'Needs Work' };
}

export function interpretMetric(key, value) {
  if (value === undefined || value === null) return null;
  const v = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(v)) return null;
  switch (key) {
    case 'r2':
      if (v < 0) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'The model performs worse than simply predicting the average value. This indicates a poor fit and unreliable predictions.' };
      if (v < 0.3) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'The model has very little predictive power. Consider trying different features or algorithms.' };
      if (v < 0.5) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'The model explains some variation but misses important patterns. There is room for significant improvement.' };
      if (v < 0.7) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'The model captures a moderate amount of the variation. Decent performance but could be improved further.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'The model explains most of the variation in the data and performs well.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent fit — the model explains nearly all variation in the data.' };
    case 'accuracy':
      if (v < 0.5) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'The model is wrong more often than right. Consider different features or algorithms.' };
      if (v < 0.7) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Below average accuracy. The model needs improvement. Note: accuracy alone can be misleading for imbalanced datasets.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good accuracy — the model correctly predicts most cases. For imbalanced data, also check precision and recall.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Very high accuracy — the model is correct in nearly all predictions.' };
    case 'f1':
      if (v < 0.5) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'Poor performance — the model struggles to balance finding positives (recall) with being correct about them (precision).' };
      if (v < 0.7) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Moderate performance. The balance between precision and recall could be improved.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good performance — the model effectively balances precision and recall.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent — the model achieves both high precision and high recall.' };
    case 'precision':
      if (v < 0.5) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'Many false alarms — when the model predicts positive, it is often wrong.' };
      if (v < 0.7) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Moderate — a noticeable portion of positive predictions are incorrect.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good — most positive predictions are correct, with few false alarms.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent — the model very rarely raises false alarms.' };
    case 'recall':
      if (v < 0.5) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'The model misses most actual positives. Many real cases go undetected.' };
      if (v < 0.7) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Moderate — the model catches some positives but misses a significant portion.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good — the model catches most actual positives with relatively few misses.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent — the model detects nearly all actual positives.' };
    case 'rmse':
      return { rating: null, color: 'text-muted-foreground', text: `Average prediction error of ${v.toFixed(2)}. Lower values mean more accurate predictions. Compare to your target variable's range.` };
    case 'mae':
      return { rating: null, color: 'text-muted-foreground', text: `On average, predictions are off by ${v.toFixed(2)}. Lower is better.` };
    case 'silhouette':
      if (v < 0) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'Negative score — data points may be in wrong clusters. The clustering structure is poor.' };
      if (v < 0.25) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Clusters overlap significantly. The data may not have a clear clustering structure.' };
      if (v < 0.5) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Moderate cluster separation. Clusters are reasonable but have some overlap.' };
      if (v < 0.75) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good — data points generally fit well within their assigned clusters.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent — each cluster is distinct and well-defined.' };
    case 'daviesBouldin':
      if (v === Infinity) return null;
      if (v > 2) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'Clusters are too similar. Consider fewer clusters or a different algorithm.' };
      if (v > 1) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Moderate cluster distinction. Some overlap exists between clusters.' };
      if (v > 0.5) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good separation — each cluster has distinct characteristics.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent distinction — clusters are clearly separated with minimal overlap.' };
    case 'calinskiHarabasz':
      if (v < 50) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Low score — clusters are not very dense or well-separated.' };
      if (v < 200) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Decent clustering — data points within clusters are reasonably compact.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Strong clustering — dense clusters that are well-separated from each other.' };
    case 'cvScore':
      if (v < 0.5) return { rating: 'poor', color: 'text-red-600 dark:text-red-400', text: 'Low cross-validation score — the model does not generalize well to unseen data.' };
      if (v < 0.7) return { rating: 'fair', color: 'text-amber-600 dark:text-amber-400', text: 'Moderate generalization. The model may benefit from more data or better features.' };
      if (v < 0.9) return { rating: 'good', color: 'text-sky-600 dark:text-sky-400', text: 'Good — the model performs consistently across different data splits.' };
      return { rating: 'excellent', color: 'text-emerald-600 dark:text-emerald-400', text: 'Excellent — very consistent performance across all cross-validation folds.' };
    default: return null;
  }
}

// ==================== PERF HELPERS (stack-safe min/max) ====================
export function arrayMin(arr) { let m = Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]; return m; }
export function arrayMax(arr) { let m = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]; return m; }
export function arrayMinMax(arr) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }

export function generateId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : ((r & 0x3) | 0x8);
    return v.toString(16);
  });
}
