import { arrayMinMax, arrayMin, arrayMax } from './helpers';

// ==================== CSV PARSING ====================

export function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    if (values.length === headers.length) {
      const row = {};
      headers.forEach((h, idx) => { const num = Number(values[idx]); row[h] = isNaN(num) || values[idx] === '' ? values[idx] : num; });
      rows.push(row);
    }
  }
  return { headers, rows };
}

// ==================== DATASET PROFILING ====================

export function profileDataset(text) {
  const { headers, rows } = parseCSV(text);
  if (!rows.length) return null;
  const columns = headers.map(col => {
    const values = rows.map(r => r[col]);
    const numericValues = values.filter(v => typeof v === 'number');
    const isNumeric = numericValues.length === values.length && numericValues.length > 0;
    const uniqueCount = new Set(values.map(String)).size;
    const profile = { name: col, type: isNumeric ? 'numeric' : 'categorical', uniqueCount, missingCount: values.filter(v => v === '' || v === null || v === undefined).length, sampleValues: [...new Set(values.map(String))].slice(0, 5) };
    if (isNumeric && numericValues.length > 0) {
      const [lo, hi] = arrayMinMax(numericValues);
      profile.min = lo; profile.max = hi;
      profile.mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
      profile.std = Math.sqrt(numericValues.reduce((s, v) => s + (v - profile.mean) ** 2, 0) / numericValues.length);
    }
    return profile;
  });
  return { rowCount: rows.length, columnCount: headers.length, columns, headers, rows, numericColumns: columns.filter(c => c.type === 'numeric').map(c => c.name), categoricalColumns: columns.filter(c => c.type === 'categorical').map(c => c.name) };
}

// ==================== DATASET SUMMARY GENERATOR ====================

export const DOMAIN_KEYWORDS = {
  telecom: ['churn', 'churned', 'tenure', 'contract', 'monthly', 'internet', 'fiber', 'dsl', 'streaming', 'phone', 'bandwidth', 'plan', 'subscriber', 'telecom', 'service'],
  finance: ['loan', 'credit', 'income', 'salary', 'interest', 'debt', 'balance', 'payment', 'bank', 'finance', 'mortgage', 'investment', 'revenue', 'profit', 'loss', 'amount', 'fee', 'tax', 'asset', 'liability', 'stock', 'portfolio'],
  health: ['age', 'bmi', 'blood', 'heart', 'disease', 'patient', 'diagnosis', 'medical', 'health', 'hospital', 'treatment', 'symptom', 'drug', 'medicine', 'cholesterol', 'glucose', 'pressure', 'cancer', 'diabetes', 'weight', 'height', 'clinical'],
  sales: ['sales', 'revenue', 'customer', 'product', 'order', 'purchase', 'quantity', 'discount', 'store', 'retail', 'marketing', 'campaign', 'conversion', 'churn', 'subscription'],
  education: ['student', 'grade', 'score', 'gpa', 'exam', 'course', 'school', 'university', 'education', 'class', 'teacher', 'attendance', 'enrollment'],
  realestate: ['house', 'property', 'price', 'sqft', 'bedroom', 'bathroom', 'area', 'location', 'rent', 'apartment', 'building', 'floor', 'garage', 'lot', 'neighborhood'],
  hr: ['employee', 'department', 'position', 'hire', 'attrition', 'performance', 'satisfaction', 'overtime', 'tenure', 'promotion', 'salary', 'role'],
  insurance: ['insurance', 'claim', 'premium', 'policy', 'coverage', 'deductible', 'beneficiary', 'charges', 'smoker', 'region'],
  ecommerce: ['cart', 'click', 'session', 'page', 'bounce', 'visitor', 'item', 'shipping', 'review', 'rating'],
  transportation: ['trip', 'distance', 'speed', 'vehicle', 'route', 'fare', 'driver', 'passenger', 'flight', 'delay'],
  environment: ['temperature', 'humidity', 'weather', 'rainfall', 'pollution', 'emission', 'co2', 'energy', 'solar', 'wind'],
};

export function generateDatasetSummary(profile) {
  if (!profile || !profile.columns || profile.columns.length === 0) return null;
  const colNamesLower = profile.columns.map(c => c.name.toLowerCase());
  const allNamesJoined = colNamesLower.join(' ');

  let bestDomain = 'general';
  let bestScore = 0;
  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (allNamesJoined.includes(kw)) score++;
    }
    if (score > bestScore) { bestScore = score; bestDomain = domain; }
  }
  const domainLabels = { telecom: 'Telecom / Customer Churn', finance: 'Financial / Credit Analysis', health: 'Healthcare / Medical', sales: 'Sales / Marketing', education: 'Education / Academic', realestate: 'Real Estate / Housing', hr: 'Human Resources / Employee', insurance: 'Insurance', ecommerce: 'E-Commerce', transportation: 'Transportation / Travel', environment: 'Environmental / Climate', general: 'General Data Analysis' };
  const domainLabel = domainLabels[bestDomain] || 'General Data Analysis';

  const scored = profile.columns.map(c => {
    let s = 0;
    if (c.type === 'numeric' && c.std > 0) s += 3;
    if (c.uniqueCount >= 2 && c.uniqueCount <= 10 && c.type === 'categorical') s += 4;
    if (c.uniqueCount === 2) s += 2;
    if (c.missingCount === 0) s += 1;
    return { ...c, score: s };
  }).sort((a, b) => b.score - a.score);
  const keyVars = scored.slice(0, Math.min(5, scored.length));

  let possibleTarget = null;
  for (const c of profile.columns) {
    if (c.type === 'categorical' && c.uniqueCount >= 2 && c.uniqueCount <= 10) {
      possibleTarget = { name: c.name, reason: `categorical with ${c.uniqueCount} classes`, task: 'classification' };
      break;
    }
  }
  if (!possibleTarget) {
    for (const c of profile.columns) {
      if (c.type === 'numeric' && c.uniqueCount > 10 && c.std > 0) {
        possibleTarget = { name: c.name, reason: `continuous numeric variable`, task: 'regression' };
        break;
      }
    }
  }

  const numCols = profile.numericColumns.length;
  const catCols = profile.categoricalColumns.length;
  const keyNames = keyVars.map(v => v.name);
  const description = [
    `This dataset contains ${profile.rowCount} records with ${profile.columnCount} variables, covering the domain of ${domainLabel}.`,
    `It includes ${numCols} numeric feature${numCols !== 1 ? 's' : ''} and ${catCols} categorical feature${catCols !== 1 ? 's' : ''}, providing a mix of quantitative measurements and categorical groupings.`,
    possibleTarget ? `The most likely objective is ${possibleTarget.task} — predicting "${possibleTarget.name}" (${possibleTarget.reason}).` : `The dataset can be used for clustering or exploratory analysis to uncover patterns.`,
    `Key variables include ${keyNames.slice(0, 3).join(', ')}${keyNames.length > 3 ? `, among others` : ''}.`,
    numCols > 0 ? `Numeric features capture measurable quantities, while categorical variables represent groups or labels.` : `The data is primarily categorical, suitable for classification or grouping tasks.`,
  ];

  const focusLine = `This dataset mainly focuses on ${domainLabel.toLowerCase()}, with key variables like ${keyNames.slice(0, 3).join(', ')}.`;

  return { domain: domainLabel, description, focusLine, keyVariables: keyVars, possibleTarget };
}

export function suggestTask(profile, targetColumn) {
  if (!targetColumn || targetColumn === '__none__') return { task: 'clustering', message: `No target selected. Clustering recommended (${profile.numericColumns.length} numeric features).`, icon: 'layers' };
  const tc = profile.columns.find(c => c.name === targetColumn);
  if (!tc) return { task: 'unknown', message: 'Target column not found.', icon: 'alert' };
  if (tc.type === 'numeric') {
    if (tc.uniqueCount === 2) return { task: 'classification', message: `Binary Classification: "${targetColumn}" has 2 unique values.`, icon: 'target' };
    if (tc.uniqueCount / profile.rowCount < 0.05) return { task: 'classification', message: `Classification: "${targetColumn}" has ${tc.uniqueCount} discrete values.`, icon: 'target' };
    return { task: 'regression', message: `Regression: "${targetColumn}" is continuous (${tc.uniqueCount} unique, range ${tc.min?.toFixed(1)}–${tc.max?.toFixed(1)}).`, icon: 'trending' };
  }
  return { task: 'classification', message: `Classification: "${targetColumn}" is categorical (${tc.uniqueCount} classes).`, icon: 'target' };
}

// ==================== DATASET SCANNER ====================

export function scanDataset(csvText, targetCol) {
  const { rows, headers } = parseCSV(csvText);
  const n = rows.length, p = headers.length;
  const numericCols = [], categoricalCols = [];
  headers.forEach(h => {
    const vals = rows.map(r => r[h]).filter(v => v !== '' && v != null);
    (vals.filter(v => !isNaN(Number(v))).length > vals.length * 0.5 ? numericCols : categoricalCols).push(h);
  });
  let totalMissing = 0;
  const missingCols = [];
  headers.forEach(h => {
    const missing = rows.filter(r => r[h] === '' || r[h] == null).length;
    totalMissing += missing;
    if (missing > 0) missingCols.push({ col: h, count: missing, pct: +(missing / n * 100).toFixed(1) });
  });
  const duplicateCount = n - new Set(rows.map(r => headers.map(h => String(r[h])).join('|'))).size;
  let totalOutliers = 0;
  const outlierCols = [];
  numericCols.forEach(h => {
    const vals = rows.map(r => Number(r[h])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    if (vals.length < 4) return;
    const q1 = vals[Math.floor(vals.length * 0.25)], q3 = vals[Math.floor(vals.length * 0.75)], iqr = q3 - q1;
    const cnt = vals.filter(v => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length;
    if (cnt > 0) { outlierCols.push({ col: h, count: cnt }); totalOutliers += cnt; }
  });
  const constantCols = headers.filter(h => new Set(rows.map(r => r[h])).size <= 1);
  const highCorr = [];
  for (let i = 0; i < numericCols.length; i++) {
    for (let j = i + 1; j < numericCols.length; j++) {
      const a = rows.map(r => Number(r[numericCols[i]])), b = rows.map(r => Number(r[numericCols[j]]));
      const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
      let cov = 0, sa = 0, sb = 0;
      for (let k = 0; k < n; k++) { cov += (a[k] - ma) * (b[k] - mb); sa += (a[k] - ma) ** 2; sb += (b[k] - mb) ** 2; }
      const corr = sa > 0 && sb > 0 ? cov / Math.sqrt(sa * sb) : 0;
      if (Math.abs(corr) > 0.9) highCorr.push({ col1: numericCols[i], col2: numericCols[j], r: +corr.toFixed(3) });
    }
  }
  let targetInfo = null;
  if (targetCol && targetCol !== '__none__' && headers.includes(targetCol)) {
    const vals = rows.map(r => r[targetCol]).filter(v => v !== '' && v != null);
    const isNum = numericCols.includes(targetCol);
    targetInfo = { exists: true, uniqueValues: [...new Set(vals)].length, task: isNum ? 'regression' : 'classification' };
    if (!isNum) {
      const counts = {}; vals.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
      const maxPct = arrayMax(Object.values(counts)) / vals.length;
      targetInfo.imbalanced = maxPct > 0.8; targetInfo.majorityPct = +(maxPct * 100).toFixed(1);
    }
  }
  let scaleIssue = false;
  if (numericCols.length > 1) {
    const ranges = numericCols.map(h => { const v = rows.map(r => Number(r[h])).filter(v => !isNaN(v)); const [lo, hi] = arrayMinMax(v); return hi - lo; }).filter(r => r > 0);
    if (ranges.length > 1 && arrayMax(ranges) / arrayMin(ranges) > 100) scaleIssue = true;
  }
  const sizeWarning = n < 100;
  let score = 100;
  if (totalMissing > 0) score -= Math.min(25, totalMissing / (n * p) * 100);
  if (duplicateCount > 0) score -= Math.min(15, duplicateCount / n * 50);
  if (totalOutliers > 0) score -= Math.min(15, totalOutliers / (n * (numericCols.length || 1)) * 50);
  if (constantCols.length > 0) score -= constantCols.length * 5;
  if (highCorr.length > 0) score -= Math.min(10, highCorr.length * 2);
  if (sizeWarning) score -= 5;
  if (targetInfo?.imbalanced) score -= 10;
  if (scaleIssue) score -= 3;
  score = Math.max(0, Math.round(score));
  const warnings = [];
  if (totalMissing > 0) warnings.push(`${totalMissing} missing values`);
  if (duplicateCount > 0) warnings.push(`${duplicateCount} duplicate rows`);
  if (totalOutliers > 0) warnings.push(`${totalOutliers} outliers detected`);
  if (constantCols.length > 0) warnings.push(`${constantCols.length} constant column(s)`);
  if (highCorr.length > 0) warnings.push(`${highCorr.length} high-correlation pair(s)`);
  if (sizeWarning) warnings.push('Small dataset (<100 rows)');
  if (targetInfo?.imbalanced) warnings.push('Class imbalance detected');
  if (scaleIssue) warnings.push('Feature scale differences');
  return { rows: n, columns: p, numericCount: numericCols.length, categoricalCount: categoricalCols.length,
    totalMissing, missingCols, duplicateCount, totalOutliers, outlierCols, constantCols, highCorr,
    targetInfo, scaleIssue, sizeWarning, score, warnings, ready: score >= 50 };
}

// ==================== DATA CLEANING FUNCTIONS ====================

export function rebuildCsv(headers, rows) {
  return [headers.join(','), ...rows.map(r => headers.map(h => { const v = String(r[h] ?? ''); return v.includes(',') ? `"${v}"` : v; }).join(','))].join('\n');
}

export function cleanRemoveDuplicates(csvText) {
  const lines = csvText.trim().split('\n');
  const header = lines[0];
  const unique = new Set(); const kept = [header]; let removed = 0;
  for (let i = 1; i < lines.length; i++) { if (!unique.has(lines[i])) { unique.add(lines[i]); kept.push(lines[i]); } else removed++; }
  return { text: kept.join('\n'), removed };
}

export function cleanFillMissing(csvText) {
  const { rows, headers } = parseCSV(csvText);
  let filled = 0;
  const isNum = {}; headers.forEach(h => { const vals = rows.map(r => r[h]).filter(v => v !== '' && v != null); isNum[h] = vals.filter(v => !isNaN(Number(v))).length > vals.length * 0.5; });
  const fillVals = {};
  headers.forEach(h => {
    const vals = rows.map(r => r[h]).filter(v => v !== '' && v != null);
    if (isNum[h]) { const nums = vals.map(Number).filter(v => !isNaN(v)).sort((a, b) => a - b); fillVals[h] = nums.length > 0 ? nums[Math.floor(nums.length / 2)] : 0; }
    else { const c = {}; vals.forEach(v => { c[v] = (c[v] || 0) + 1; }); fillVals[h] = Object.entries(c).sort((a, b) => b[1] - a[1])[0]?.[0] || ''; }
  });
  rows.forEach(r => { headers.forEach(h => { if (r[h] === '' || r[h] == null) { r[h] = fillVals[h]; filled++; } }); });
  return { text: rebuildCsv(headers, rows), filled };
}

export function cleanRemoveOutliers(csvText) {
  const { rows, headers } = parseCSV(csvText);
  const numCols = headers.filter(h => rows.map(r => r[h]).filter(v => v !== '' && v != null).filter(v => !isNaN(Number(v))).length > rows.length * 0.5);
  const bounds = {};
  numCols.forEach(h => {
    const vals = rows.map(r => Number(r[h])).filter(v => !isNaN(v)).sort((a, b) => a - b);
    if (vals.length < 4) return;
    const q1 = vals[Math.floor(vals.length * 0.25)], q3 = vals[Math.floor(vals.length * 0.75)], iqr = q3 - q1;
    bounds[h] = { lo: q1 - 1.5 * iqr, hi: q3 + 1.5 * iqr };
  });
  let removed = 0;
  const kept = rows.filter(r => { for (const h of Object.keys(bounds)) { const v = Number(r[h]); if (!isNaN(v) && (v < bounds[h].lo || v > bounds[h].hi)) { removed++; return false; } } return true; });
  return { text: rebuildCsv(headers, kept), removed };
}

export function cleanDropConstants(csvText) {
  const { rows, headers } = parseCSV(csvText);
  const dropped = []; const kept = headers.filter(h => { if (new Set(rows.map(r => r[h])).size <= 1) { dropped.push(h); return false; } return true; });
  return { text: rebuildCsv(kept, rows), dropped };
}

export function cleanNormalize(csvText) {
  const { rows, headers } = parseCSV(csvText);
  let count = 0;
  const numCols = headers.filter(h => rows.map(r => r[h]).filter(v => v !== '' && v != null).filter(v => !isNaN(Number(v))).length > rows.length * 0.5);
  numCols.forEach(h => {
    const vals = rows.map(r => Number(r[h])).filter(v => !isNaN(v));
    const min = arrayMin(vals), range = arrayMax(vals) - min;
    if (range === 0) return;
    rows.forEach(r => { const v = Number(r[h]); if (!isNaN(v)) r[h] = ((v - min) / range).toFixed(4); });
    count++;
  });
  return { text: rebuildCsv(headers, rows), count };
}

// ==================== CLUSTERING ====================

export function runKMeansClustering(rows, numericCols, k, kmeans) {
  const stats = {};
  numericCols.forEach(col => { const vals = rows.map(r => typeof r[col] === 'number' ? r[col] : 0); const mean = vals.reduce((a, b) => a + b, 0) / vals.length; const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 1; stats[col] = { mean, std }; });
  const Xstd = rows.map(row => numericCols.map(col => { const v = typeof row[col] === 'number' ? row[col] : stats[col].mean; return (v - stats[col].mean) / stats[col].std; }));
  const result = kmeans(Xstd, k, { initialization: 'kmeans++' });
  const clusterStats = Array.from({ length: k }, (_, i) => {
    const indices = result.clusters.reduce((arr, c, idx) => { if (c === i) arr.push(idx); return arr; }, []);
    const means = numericCols.map((col, j) => ({ feature: col, mean: indices.length > 0 ? indices.reduce((s, idx) => s + (typeof rows[idx][col] === 'number' ? rows[idx][col] : 0), 0) / indices.length : 0 }));
    return { clusterId: i, size: indices.length, means };
  });
  const f1 = numericCols[0] || 'x', f2 = numericCols[1] || numericCols[0] || 'y';
  const points = rows.map((row, idx) => ({ x: typeof row[f1] === 'number' ? row[f1] : 0, y: typeof row[f2] === 'number' ? row[f2] : 0, cluster: result.clusters[idx], index: idx }));
  return { clusters: result.clusters, clusterStats, points, k, features: numericCols, xFeature: f1, yFeature: f2 };
}

// ==================== ANOMALY DETECTION ====================

export function detectAnomaliesFunc(rows, numericCols, method, threshold) {
  const anomalies = {}, anomalyRows = new Set(), columnStats = {};
  numericCols.forEach(col => {
    const values = rows.map(r => r[col]).filter(v => typeof v === 'number');
    if (values.length === 0) return;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)], q3 = sorted[Math.floor(sorted.length * 0.75)], iqr = q3 - q1;
    columnStats[col] = { mean, std, q1, q3, iqr };
    anomalies[col] = [];
    if (method === 'zscore') {
      rows.forEach((row, idx) => { if (typeof row[col] !== 'number' || std === 0) return; const z = Math.abs((row[col] - mean) / std); if (z > threshold) { anomalies[col].push({ index: idx, value: row[col], score: z }); anomalyRows.add(idx); } });
    } else {
      const lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr;
      rows.forEach((row, idx) => { if (typeof row[col] !== 'number') return; if (row[col] < lower || row[col] > upper) { anomalies[col].push({ index: idx, value: row[col], bound: row[col] < lower ? 'below' : 'above' }); anomalyRows.add(idx); } });
    }
  });
  const f1 = numericCols[0], f2 = numericCols[1] || numericCols[0];
  const normalPts = [], anomalyPts = [];
  rows.forEach((row, idx) => { const pt = { x: typeof row[f1] === 'number' ? row[f1] : 0, y: typeof row[f2] === 'number' ? row[f2] : 0, index: idx }; (anomalyRows.has(idx) ? anomalyPts : normalPts).push(pt); });
  return { anomalies, anomalyRowIndices: [...anomalyRows], totalAnomalies: anomalyRows.size, totalRows: rows.length, method, threshold, columnStats, normalPoints: normalPts, anomalyPoints: anomalyPts, xFeature: f1, yFeature: f2 };
}
