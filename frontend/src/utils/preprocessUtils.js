/**
 * Data Preprocessing Utilities
 * Provides missing value handling, outlier treatment, and feature scaling.
 */

/** Fill missing values in rows based on strategy */
export function handleMissingValues(rows, headers, numericColumns, strategy = 'auto') {
  if (!rows.length || strategy === 'none') return { rows, log: [] };
  const log = [];
  const filled = rows.map(r => ({ ...r }));

  headers.forEach(col => {
    const missingIdxs = filled.map((r, i) => (r[col] === '' || r[col] == null || (typeof r[col] === 'number' && isNaN(r[col]))) ? i : -1).filter(i => i >= 0);
    if (missingIdxs.length === 0) return;

    const isNum = numericColumns.includes(col);
    let fillValue;
    const method = strategy === 'auto' ? (isNum ? 'median' : 'mode') : strategy;

    if (method === 'drop') return; // handled below
    if (method === 'zero') {
      fillValue = isNum ? 0 : '';
    } else if (isNum) {
      const vals = filled.filter((r, i) => !missingIdxs.includes(i)).map(r => Number(r[col])).filter(v => !isNaN(v));
      if (method === 'mean') fillValue = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      else if (method === 'median') { const s = [...vals].sort((a, b) => a - b); fillValue = s[Math.floor(s.length / 2)] || 0; }
      else fillValue = vals.reduce((a, b) => a + b, 0) / (vals.length || 1); // default mean
    } else {
      // Categorical: mode
      const counts = {};
      filled.filter((r, i) => !missingIdxs.includes(i)).forEach(r => { const v = String(r[col]); counts[v] = (counts[v] || 0) + 1; });
      fillValue = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0] || '';
    }

    missingIdxs.forEach(i => { filled[i][col] = fillValue; });
    log.push({ col, count: missingIdxs.length, strategy: method, fillValue: typeof fillValue === 'number' ? fillValue.toFixed(2) : fillValue });
  });

  if (strategy === 'drop') {
    const before = filled.length;
    const cleaned = filled.filter(r => headers.every(h => r[h] !== '' && r[h] != null && !(typeof r[h] === 'number' && isNaN(r[h]))));
    log.push({ col: 'ALL', count: before - cleaned.length, strategy: 'drop', fillValue: 'N/A' });
    return { rows: cleaned, log };
  }

  return { rows: filled, log };
}

/** Remove or clip outliers using IQR method */
export function handleOutliers(rows, numericColumns, method = 'none', threshold = 1.5) {
  if (!rows.length || method === 'none') return { rows, log: [] };
  const log = [];
  let processed = rows.map(r => ({ ...r }));

  numericColumns.forEach(col => {
    const vals = processed.map(r => Number(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    if (vals.length < 4) return;
    const q1 = vals[Math.floor(vals.length * 0.25)];
    const q3 = vals[Math.floor(vals.length * 0.75)];
    const iqr = q3 - q1;
    if (iqr === 0) return;
    const lower = q1 - threshold * iqr;
    const upper = q3 + threshold * iqr;
    let affected = 0;

    if (method === 'clip') {
      processed.forEach(r => {
        const v = Number(r[col]);
        if (!isNaN(v) && (v < lower || v > upper)) {
          r[col] = Math.max(lower, Math.min(upper, v));
          affected++;
        }
      });
    } else if (method === 'remove') {
      const before = processed.length;
      processed = processed.filter(r => {
        const v = Number(r[col]);
        return isNaN(v) || (v >= lower && v <= upper);
      });
      affected = before - processed.length;
    }
    if (affected > 0) log.push({ col, affected, method, range: `[${lower.toFixed(2)}, ${upper.toFixed(2)}]` });
  });

  return { rows: processed, log };
}

/** Standardize (z-score) or min-max normalize feature matrix */
export function scaleFeatures(X, method = 'none') {
  if (!X.length || method === 'none') return { X, scaleParams: null };
  const nFeatures = X[0].length;
  const params = [];

  if (method === 'standard') {
    for (let j = 0; j < nFeatures; j++) {
      const vals = X.map(row => row[j]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 1;
      params.push({ mean, std });
    }
    const scaled = X.map(row => row.map((v, j) => (v - params[j].mean) / params[j].std));
    return { X: scaled, scaleParams: { method: 'standard', params } };
  }

  if (method === 'minmax') {
    for (let j = 0; j < nFeatures; j++) {
      const vals = X.map(row => row[j]);
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const range = max - min || 1;
      params.push({ min, range });
    }
    const scaled = X.map(row => row.map((v, j) => (v - params[j].min) / params[j].range));
    return { X: scaled, scaleParams: { method: 'minmax', params } };
  }

  return { X, scaleParams: null };
}

/** Apply stored scale params to new data for prediction */
export function applyScaling(X, scaleParams) {
  if (!scaleParams || !X.length) return X;
  if (scaleParams.method === 'standard') {
    return X.map(row => row.map((v, j) => (v - scaleParams.params[j].mean) / scaleParams.params[j].std));
  }
  if (scaleParams.method === 'minmax') {
    return X.map(row => row.map((v, j) => (v - scaleParams.params[j].min) / scaleParams.params[j].range));
  }
  return X;
}

/** Get default preprocessing config */
export function getDefaultPreprocessConfig() {
  return {
    missingValues: 'auto',
    scaling: 'none',
    outlierMethod: 'none',
    outlierThreshold: 1.5,
    excludeFeatures: [],
  };
}
