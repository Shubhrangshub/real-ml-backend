// ==================== FEATURE PREPARATION ====================

export function detectProblemType(values) {
  if (values.some(v => typeof v !== 'number')) return 'classification';
  const uv = [...new Set(values)];
  if (uv.length === 2) return 'classification';
  if (uv.length / values.length < 0.05) return 'classification';
  return 'regression';
}

export function prepareFeatures(rows, targetCol) {
  const allCols = Object.keys(rows[0]).filter(k => k !== targetCol);
  const leakageKeywords = ['id', '_id', 'date', 'year', 'added', 'created', 'updated'];
  const leakageCols = [];
  const safeCols = allCols.filter(col => { const low = col.toLowerCase(); const isLeak = leakageKeywords.some(kw => low.includes(kw)); if (isLeak) leakageCols.push(col); return !isLeak; });
  const numericCols = [], categoricalCols = [], textCols = [];
  safeCols.forEach(col => {
    if (rows.every(row => typeof row[col] === 'number')) numericCols.push(col);
    else { const avgLen = rows.reduce((s, r) => s + String(r[col]).length, 0) / rows.length; (avgLen > 20 ? textCols : categoricalCols).push(col); }
  });
  const encodingMap = {};
  categoricalCols.forEach(col => { encodingMap[col] = [...new Set(rows.map(r => String(r[col])))].sort(); });
  const featureNames = [...numericCols];
  categoricalCols.forEach(col => encodingMap[col].slice(1).forEach(val => featureNames.push(`${col}_${val}`)));
  const X = rows.map(row => {
    const f = [];
    numericCols.forEach(col => f.push(row[col]));
    categoricalCols.forEach(col => encodingMap[col].slice(1).forEach(val => f.push(String(row[col]) === val ? 1 : 0)));
    return f;
  });
  let y = rows.map(row => row[targetCol]);
  let targetEncoding = null;
  if (y.some(v => typeof v !== 'number')) {
    const uniq = [...new Set(y.map(String))].sort();
    targetEncoding = uniq;
    y = y.map(v => uniq.indexOf(String(v)));
  }
  return { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols };
}

// ==================== TRAIN-TEST SPLIT ====================

export function trainTestSplit(X, y, testSize = 0.2) {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [indices[i], indices[j]] = [indices[j], indices[i]]; }
  const splitIdx = Math.floor(n * (1 - testSize));
  return {
    X_train: indices.slice(0, splitIdx).map(i => X[i]), X_test: indices.slice(splitIdx).map(i => X[i]),
    y_train: indices.slice(0, splitIdx).map(i => y[i]), y_test: indices.slice(splitIdx).map(i => y[i]),
    trainSize: splitIdx, testSize: n - splitIdx
  };
}

// ==================== LINEAR ALGEBRA ====================

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) { if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row; }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-10) continue;
    for (let row = col + 1; row < n; row++) { const f = M[row][col] / M[col][col]; for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j]; }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { if (Math.abs(M[i][i]) < 1e-10) continue; x[i] = M[i][n]; for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j]; x[i] /= M[i][i]; }
  return x;
}

function computeXtXAndXty(Xa, y, pa, n) {
  const XtX = Array(pa).fill(null).map(() => Array(pa).fill(0));
  for (let i = 0; i < pa; i++) for (let j = i; j < pa; j++) { let s = 0; for (let k = 0; k < n; k++) s += Xa[k][i] * Xa[k][j]; XtX[i][j] = s; XtX[j][i] = s; }
  const Xty = Array(pa).fill(0);
  for (let i = 0; i < pa; i++) { let s = 0; for (let k = 0; k < n; k++) s += Xa[k][i] * y[k]; Xty[i] = s; }
  return { XtX, Xty };
}

// ==================== REGRESSION MODELS ====================

function trainLinearRegression(X, y) {
  const n = X.length, p = X[0].length, pa = p + 1;
  const Xa = X.map(row => [1, ...row]);
  const { XtX, Xty } = computeXtXAndXty(Xa, y, pa, n);
  for (let i = 0; i < pa; i++) XtX[i][i] += 0.001;
  return { type: 'linear_regression', coefficients: solveLinearSystem(XtX, Xty) };
}

function trainRidgeRegression(X, y, lambda = 1.0) {
  const n = X.length, p = X[0].length, pa = p + 1;
  const Xa = X.map(row => [1, ...row]);
  const { XtX, Xty } = computeXtXAndXty(Xa, y, pa, n);
  for (let i = 1; i < pa; i++) XtX[i][i] += lambda;
  XtX[0][0] += 0.001;
  return { type: 'ridge_regression', coefficients: solveLinearSystem(XtX, Xty) };
}

// ==================== CLASSIFICATION MODELS ====================

function trainLogisticRegression(X, y) {
  return { type: 'logistic_regression', coefficients: trainLinearRegression(X, y).coefficients };
}

// ==================== DECISION TREE ====================

export function buildDecisionTree(X, y, maxDepth, minSamples, isClassification) {
  const numFeatures = X[0].length;
  const maxFeatPerSplit = Math.max(1, Math.floor(Math.sqrt(numFeatures)));

  function gini(indices) {
    const n = indices.length; if (n === 0) return 0;
    const counts = {};
    for (let i = 0; i < n; i++) { const v = y[indices[i]]; counts[v] = (counts[v] || 0) + 1; }
    let g = 1;
    for (const c of Object.values(counts)) { const p = c / n; g -= p * p; }
    return g;
  }

  function mse(indices) {
    const n = indices.length; if (n === 0) return 0;
    let sum = 0; for (let i = 0; i < n; i++) sum += y[indices[i]];
    const mean = sum / n;
    let s = 0; for (let i = 0; i < n; i++) s += (y[indices[i]] - mean) ** 2;
    return s / n;
  }

  const impurity = isClassification ? gini : mse;

  function leafVal(indices) {
    if (isClassification) {
      const counts = {};
      for (let i = 0; i < indices.length; i++) { const v = y[indices[i]]; counts[v] = (counts[v] || 0) + 1; }
      let best = -1, bestC = 0;
      for (const [k, v] of Object.entries(counts)) { if (v > bestC) { bestC = v; best = Number(k); } }
      return best;
    }
    let s = 0; for (let i = 0; i < indices.length; i++) s += y[indices[i]];
    return s / indices.length;
  }

  function build(indices, depth) {
    const n = indices.length;
    if (depth >= maxDepth || n < minSamples) return { leaf: true, value: leafVal(indices), n };
    if (isClassification) { const first = y[indices[0]]; if (indices.every(i => y[i] === first)) return { leaf: true, value: first, n }; }

    const parentImp = impurity(indices);
    const allFeats = Array.from({ length: numFeatures }, (_, i) => i);
    for (let i = allFeats.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [allFeats[i], allFeats[j]] = [allFeats[j], allFeats[i]]; }
    const featSubset = allFeats.slice(0, maxFeatPerSplit);

    let bestGain = 0, bestF = -1, bestT = 0, bestL = null, bestR = null;

    const sortBuf = new Array(n);
    for (const f of featSubset) {
      for (let i = 0; i < n; i++) sortBuf[i] = i;
      sortBuf.sort((a, b) => X[indices[a]][f] - X[indices[b]][f]);
      const step = Math.max(1, Math.floor(n / 25));
      for (let t = step; t < n; t += step) {
        const lv = X[indices[sortBuf[t - 1]]][f], rv = X[indices[sortBuf[t]]][f];
        if (rv === lv) continue;
        const thresh = (lv + rv) / 2;
        const left = new Array(t), right = new Array(n - t);
        for (let i = 0; i < t; i++) left[i] = indices[sortBuf[i]];
        for (let i = t; i < n; i++) right[i - t] = indices[sortBuf[i]];
        const gain = parentImp - (t / n) * impurity(left) - ((n - t) / n) * impurity(right);
        if (gain > bestGain) { bestGain = gain; bestF = f; bestT = thresh; bestL = left; bestR = right; }
      }
    }
    if (bestF === -1) return { leaf: true, value: leafVal(indices), n };
    return { leaf: false, feature: bestF, threshold: bestT, left: build(bestL, depth + 1), right: build(bestR, depth + 1), n };
  }

  return build(Array.from({ length: X.length }, (_, i) => i), 0);
}

export function predictTree(tree, x) {
  let node = tree;
  while (!node.leaf) node = x[node.feature] <= node.threshold ? node.left : node.right;
  return node.value;
}

function treeFeatureImportance(tree, featureNames) {
  const imp = new Array(featureNames.length).fill(0);
  function visit(node) { if (!node.leaf) { imp[node.feature] += node.n; visit(node.left); visit(node.right); } }
  visit(tree);
  const total = imp.reduce((a, b) => a + b, 0);
  if (total === 0) return [];
  return featureNames.map((name, i) => ({ feature: name, importance: imp[i] / total })).filter(f => f.importance > 0).sort((a, b) => b.importance - a.importance).slice(0, 10);
}

// ==================== RANDOM FOREST ====================

function trainRandomForest(X, y, isClassification, nTrees = 10, maxDepth = 8) {
  const n = X.length;
  const trees = [];
  for (let t = 0; t < nTrees; t++) {
    const idx = Array.from({ length: n }, () => Math.floor(Math.random() * n));
    const tree = buildDecisionTree(idx.map(i => X[i]), idx.map(i => y[i]), maxDepth, 2, isClassification);
    trees.push(tree);
  }
  return { type: isClassification ? 'random_forest_classifier' : 'random_forest_regressor', trees, isClassification };
}

// ==================== GRADIENT BOOSTING REGRESSOR ====================

function trainGradientBoosting(X, y, nTrees = 30, lr = 0.1, maxDepth = 4) {
  const n = X.length;
  const baseMean = y.reduce((a, b) => a + b, 0) / n;
  const trees = [];
  const residuals = y.map(v => v - baseMean);
  for (let t = 0; t < nTrees; t++) {
    const tree = buildDecisionTree(X, [...residuals], maxDepth, 2, false);
    trees.push(tree);
    for (let i = 0; i < n; i++) residuals[i] -= lr * predictTree(tree, X[i]);
  }
  return { type: 'gradient_boosting', trees, baseMean, learningRate: lr };
}

// ==================== KNN CLASSIFIER ====================

export function standardizeFeatures(X) {
  const n = X.length, p = X[0].length;
  const means = new Array(p).fill(0), stds = new Array(p).fill(1);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) means[j] += X[i][j];
    means[j] /= n;
    let s = 0;
    for (let i = 0; i < n; i++) s += (X[i][j] - means[j]) ** 2;
    stds[j] = Math.sqrt(s / n) || 1;
  }
  return { means, stds, Xs: X.map(row => row.map((v, j) => (v - means[j]) / stds[j])) };
}

function trainKNN(X, y, k = 5) {
  const { means, stds, Xs } = standardizeFeatures(X);
  return { type: 'knn', X_train: Xs, y_train: [...y], k, means, stds };
}

// ==================== LINEAR SVM ====================

function fitBinarySVM(X, yBin, p, C, lr, epochs) {
  const n = X.length, w = new Array(p).fill(0);
  let b = 0;
  for (let e = 0; e < epochs; e++) {
    for (let i = 0; i < n; i++) {
      const m = yBin[i] * (X[i].reduce((s, v, j) => s + v * w[j], 0) + b);
      if (m < 1) { for (let j = 0; j < p; j++) w[j] += lr * (C * yBin[i] * X[i][j] - w[j] / n); b += lr * C * yBin[i]; }
      else { for (let j = 0; j < p; j++) w[j] *= (1 - lr / n); }
    }
  }
  return { w: [...w], b };
}

function trainSVM(X, y, C = 1.0, lr = 0.01, epochs = 200) {
  const p = X[0].length;
  const { means, stds, Xs } = standardizeFeatures(X);
  const classes = [...new Set(y)].sort((a, b) => a - b);
  if (classes.length <= 2) {
    const yBin = y.map(v => v === classes[0] ? -1 : 1);
    const { w, b } = fitBinarySVM(Xs, yBin, p, C, lr, epochs);
    return { type: 'svm', w, b, classes, means, stds, multiclass: false };
  }
  const classifiers = classes.map(cls => fitBinarySVM(Xs, y.map(v => v === cls ? 1 : -1), p, C, lr, epochs));
  return { type: 'svm', classifiers, classes, means, stds, multiclass: true };
}

// ==================== NAIVE BAYES ====================

function trainNaiveBayes(X, y) {
  const classes = [...new Set(y)].sort((a, b) => a - b);
  const n = X.length, p = X[0].length;
  const classStats = {};
  classes.forEach(cls => {
    const idx = []; for (let i = 0; i < n; i++) if (y[i] === cls) idx.push(i);
    const means = new Array(p).fill(0), variances = new Array(p).fill(0);
    idx.forEach(i => { for (let j = 0; j < p; j++) means[j] += X[i][j]; });
    for (let j = 0; j < p; j++) means[j] /= idx.length;
    idx.forEach(i => { for (let j = 0; j < p; j++) variances[j] += (X[i][j] - means[j]) ** 2; });
    for (let j = 0; j < p; j++) variances[j] = variances[j] / idx.length + 1e-9;
    classStats[cls] = { prior: idx.length / n, means, variances };
  });
  return { type: 'naive_bayes', classStats, classes };
}

// ==================== BASELINE ====================

function trainBaseline(y, problemType) {
  if (problemType === 'regression') return { type: 'baseline', value: y.reduce((a, b) => a + b, 0) / y.length };
  const counts = {};
  y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
  return { type: 'baseline', value: Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]) };
}

// ==================== UNIVERSAL PREDICT ====================

export function predictOne(modelObj, x) {
  const t = modelObj.type;
  if (t === 'baseline') return modelObj.value;
  if (t === 'decision_tree') return predictTree(modelObj.tree, x);
  if (t === 'random_forest_regressor') return modelObj.trees.reduce((s, tr) => s + predictTree(tr, x), 0) / modelObj.trees.length;
  if (t === 'random_forest_classifier') {
    const votes = {}; let bestVote = -1, bestCount = 0;
    for (let i = 0; i < modelObj.trees.length; i++) {
      const p = predictTree(modelObj.trees[i], x);
      const c = (votes[p] || 0) + 1;
      votes[p] = c;
      if (c > bestCount) { bestCount = c; bestVote = p; }
    }
    return Number(bestVote);
  }
  if (t === 'gradient_boosting') {
    let p = modelObj.baseMean; modelObj.trees.forEach(tr => { p += modelObj.learningRate * predictTree(tr, x); }); return p;
  }
  if (t === 'knn') {
    const xs = modelObj.means ? x.map((v, j) => (v - modelObj.means[j]) / modelObj.stds[j]) : x;
    const k = modelObj.k, xt = modelObj.X_train, yt = modelObj.y_train, nTr = xt.length;
    const topD = new Float64Array(k).fill(Infinity);
    const topL = new Array(k);
    for (let i = 0; i < nTr; i++) {
      const xi = xt[i];
      let d = 0;
      for (let j = 0; j < xs.length; j++) { const diff = xi[j] - xs[j]; d += diff * diff; }
      if (d < topD[k - 1]) {
        let pos = k - 1;
        while (pos > 0 && d < topD[pos - 1]) { topD[pos] = topD[pos - 1]; topL[pos] = topL[pos - 1]; pos--; }
        topD[pos] = d; topL[pos] = yt[i];
      }
    }
    const counts = {}; let bestLabel = topL[0], bestC = 0;
    for (let i = 0; i < k; i++) { const l = topL[i]; const c = (counts[l] || 0) + 1; counts[l] = c; if (c > bestC) { bestC = c; bestLabel = l; } }
    return Number(bestLabel);
  }
  if (t === 'svm') {
    const xs = modelObj.means ? x.map((v, j) => (v - modelObj.means[j]) / modelObj.stds[j]) : x;
    if (modelObj.multiclass) { const sc = modelObj.classifiers.map(c => xs.reduce((s, v, j) => s + v * c.w[j], 0) + c.b); let mx = -Infinity, mi = 0; for (let i = 0; i < sc.length; i++) if (sc[i] > mx) { mx = sc[i]; mi = i; } return modelObj.classes[mi]; }
    return (xs.reduce((s, v, j) => s + v * modelObj.w[j], 0) + modelObj.b) >= 0 ? modelObj.classes[1] : modelObj.classes[0];
  }
  if (t === 'naive_bayes') {
    let best = modelObj.classes[0], bestLP = -Infinity;
    for (const cls of modelObj.classes) {
      const s = modelObj.classStats[cls]; let lp = Math.log(s.prior);
      for (let j = 0; j < Math.min(x.length, s.means.length); j++) lp += -0.5 * Math.log(2 * Math.PI * s.variances[j]) - (x[j] - s.means[j]) ** 2 / (2 * s.variances[j]);
      if (lp > bestLP) { bestLP = lp; best = cls; }
    }
    return best;
  }
  if (t === 'logistic_regression') {
    const z = modelObj.coefficients[0] + x.reduce((s, v, i) => s + v * modelObj.coefficients[i + 1], 0);
    return (1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))))) >= 0.5 ? 1 : 0;
  }
  return modelObj.coefficients[0] + x.reduce((s, v, i) => s + v * modelObj.coefficients[i + 1], 0);
}

export function predictBatch(modelObj, X) { return X.map(x => predictOne(modelObj, x)); }

// ==================== METRICS ====================

export function calcRegressionMetrics(actual, predicted) {
  const n = actual.length;
  const meanA = actual.reduce((a, b) => a + b, 0) / n;
  const ssTot = actual.reduce((s, v) => s + (v - meanA) ** 2, 0);
  const ssRes = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  return { r2: ssTot > 0 ? 1 - ssRes / ssTot : 0, mae: actual.reduce((s, v, i) => s + Math.abs(v - predicted[i]), 0) / n, rmse: Math.sqrt(ssRes / n) };
}

export function calcClassificationMetrics(actual, predicted) {
  const n = actual.length;
  const correct = actual.filter((v, i) => v === predicted[i]).length;
  const accuracy = correct / n;
  const classes = [...new Set([...actual, ...predicted])].sort((a, b) => a - b);
  const matrix = classes.map(() => classes.map(() => 0));
  actual.forEach((a, i) => { const ai = classes.indexOf(a), pi = classes.indexOf(predicted[i]); if (ai >= 0 && pi >= 0) matrix[ai][pi]++; });
  const perClass = classes.map((cls, ci) => {
    const tp = matrix[ci][ci];
    const fp = classes.reduce((s, _, j) => s + (j !== ci ? matrix[j][ci] : 0), 0);
    const fn = classes.reduce((s, _, j) => s + (j !== ci ? matrix[ci][j] : 0), 0);
    return { class: cls, precision: tp + fp > 0 ? tp / (tp + fp) : 0, recall: tp + fn > 0 ? tp / (tp + fn) : 0 };
  });
  const macroPrecision = perClass.reduce((s, c) => s + c.precision, 0) / perClass.length;
  const macroRecall = perClass.reduce((s, c) => s + c.recall, 0) / perClass.length;
  const macroF1 = (macroPrecision + macroRecall) > 0 ? 2 * macroPrecision * macroRecall / (macroPrecision + macroRecall) : 0;
  const perClassWithF1 = perClass.map(c => ({ ...c, f1: (c.precision + c.recall) > 0 ? 2 * c.precision * c.recall / (c.precision + c.recall) : 0 }));
  return {
    accuracy,
    precision: macroPrecision,
    recall: macroRecall,
    f1: macroF1,
    confusionMatrix: { matrix, classes },
    perClassMetrics: perClassWithF1
  };
}

export function extractImportance(modelObj, featureNames) {
  if (modelObj.type === 'decision_tree') return treeFeatureImportance(modelObj.tree, featureNames);
  if (modelObj.type === 'random_forest_regressor' || modelObj.type === 'random_forest_classifier' || modelObj.type === 'gradient_boosting') {
    const imp = new Array(featureNames.length).fill(0);
    const fnMap = new Map(); featureNames.forEach((f, i) => fnMap.set(f, i));
    modelObj.trees.forEach(tree => { const ti = treeFeatureImportance(tree, featureNames); ti.forEach(f => { const idx = fnMap.get(f.feature); if (idx !== undefined) imp[idx] += f.importance; }); });
    const total = imp.reduce((a, b) => a + b, 0);
    if (total === 0) return [];
    return featureNames.map((name, i) => ({ feature: name, importance: imp[i] / total })).filter(f => f.importance > 0).sort((a, b) => b.importance - a.importance).slice(0, 10);
  }
  if (modelObj.type === 'svm') {
    const w = modelObj.multiclass ? featureNames.map((_, j) => modelObj.classifiers.reduce((s, c) => s + Math.abs(c.w[j]), 0)) : modelObj.w.map(c => Math.abs(c));
    const total = w.reduce((a, b) => a + b, 0);
    if (total === 0) return [];
    return featureNames.map((name, i) => ({ feature: name, importance: w[i] / total })).sort((a, b) => b.importance - a.importance).slice(0, 10);
  }
  if (modelObj.type === 'baseline' || modelObj.type === 'knn' || modelObj.type === 'naive_bayes') return [];
  if (!modelObj.coefficients) return [];
  const w = modelObj.coefficients.slice(1).map(c => Math.abs(c));
  const total = w.reduce((a, b) => a + b, 0);
  if (total === 0) return [];
  return featureNames.map((name, i) => ({ feature: name, importance: w[i] / total })).sort((a, b) => b.importance - a.importance).slice(0, 10);
}

export function prepareInputForPrediction(inputRows, modelData) {
  const { numericCols, categoricalCols, encodingMap } = modelData;
  return inputRows.map(row => {
    const f = [];
    numericCols.forEach(col => f.push(Number(row[col]) || 0));
    categoricalCols.forEach(col => encodingMap[col].slice(1).forEach(val => f.push(String(row[col] || '') === val ? 1 : 0)));
    return f;
  });
}

// ==================== K-FOLD CROSS VALIDATION ====================

export function kFoldCrossValidation(X, y, k, trainFn, problemType) {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [indices[i], indices[j]] = [indices[j], indices[i]]; }
  const foldSize = Math.floor(n / k);
  const foldScores = [];
  for (let fold = 0; fold < k; fold++) {
    const vs = fold * foldSize, ve = fold === k - 1 ? n : (fold + 1) * foldSize;
    const valIdx = indices.slice(vs, ve);
    const trainIdx = [...indices.slice(0, vs), ...indices.slice(ve)];
    try {
      const model = trainFn(trainIdx.map(i => X[i]), trainIdx.map(i => y[i]));
      const preds = predictBatch(model, valIdx.map(i => X[i]));
      const metrics = problemType === 'regression' ? calcRegressionMetrics(valIdx.map(i => y[i]), preds) : calcClassificationMetrics(valIdx.map(i => y[i]), preds);
      foldScores.push(problemType === 'regression' ? metrics.r2 : metrics.accuracy);
    } catch { foldScores.push(0); }
  }
  return { cvScore: foldScores.reduce((a, b) => a + b, 0) / foldScores.length, foldScores };
}

export function buildModelForAlgo(algo, X_train, y_train, problemType) {
  if (algo === 'linear_regression') return trainLinearRegression(X_train, y_train);
  if (algo === 'ridge_regression') return trainRidgeRegression(X_train, y_train, 1.0);
  if (algo === 'logistic_regression') return trainLogisticRegression(X_train, y_train);
  if (algo === 'decision_tree') {
    const maxDepth = Math.min(15, Math.max(3, Math.floor(Math.log2(X_train.length))));
    return { type: 'decision_tree', tree: buildDecisionTree(X_train, y_train, maxDepth, 2, problemType === 'classification') };
  }
  if (algo === 'random_forest') {
    const maxDepth = Math.min(8, Math.max(3, Math.floor(Math.log2(X_train.length))));
    return trainRandomForest(X_train, y_train, problemType === 'classification', 10, maxDepth);
  }
  if (algo === 'gradient_boosting') return trainGradientBoosting(X_train, y_train, 30, 0.1, 4);
  if (algo === 'knn') return trainKNN(X_train, y_train, Math.min(5, Math.floor(X_train.length / 2)));
  if (algo === 'svm') return trainSVM(X_train, y_train, 1.0, 0.01, 200);
  if (algo === 'naive_bayes') return trainNaiveBayes(X_train, y_train);
  return trainBaseline(y_train, problemType);
}
