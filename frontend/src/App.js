import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, Sparkles, TrendingUp, Activity, Database, Zap, Settings, Upload, Play,
  Eye, Trash2, ChevronRight, ArrowUpRight, FileText, Target, Cpu, BarChart3,
  Download, AlertCircle, Layers, ShieldAlert, Table2, Info
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell, PieChart, Pie
} from 'recharts';
import { kmeans } from 'ml-kmeans';
import './App.css';

// ==================== CONSTANTS ====================
const CLUSTER_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#e11d48', '#4f46e5', '#059669', '#d97706'];

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};
const staggerContainer = {
  animate: { transition: { staggerChildren: 0.1 } }
};

// ==================== CLIENT-SIDE ML ENGINE ====================

function generateId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : ((r & 0x3) | 0x8);
    return v.toString(16);
  });
}

function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return { headers: [], rows: [] };
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    if (values.length === headers.length) {
      const row = {};
      headers.forEach((h, idx) => {
        const num = Number(values[idx]);
        row[h] = isNaN(num) || values[idx] === '' ? values[idx] : num;
      });
      rows.push(row);
    }
  }
  return { headers, rows };
}

// ==================== DATASET PROFILING ====================

function profileDataset(text) {
  const { headers, rows } = parseCSV(text);
  if (!rows.length) return null;

  const columns = headers.map(col => {
    const values = rows.map(r => r[col]);
    const numericValues = values.filter(v => typeof v === 'number');
    const isNumeric = numericValues.length === values.length && numericValues.length > 0;
    const uniqueCount = new Set(values.map(String)).size;
    const profile = {
      name: col,
      type: isNumeric ? 'numeric' : 'categorical',
      uniqueCount,
      missingCount: values.filter(v => v === '' || v === null || v === undefined).length,
      sampleValues: [...new Set(values.map(String))].slice(0, 5)
    };
    if (isNumeric && numericValues.length > 0) {
      profile.min = Math.min(...numericValues);
      profile.max = Math.max(...numericValues);
      profile.mean = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
      profile.std = Math.sqrt(numericValues.reduce((s, v) => s + (v - profile.mean) ** 2, 0) / numericValues.length);
    }
    return profile;
  });

  return {
    rowCount: rows.length,
    columnCount: headers.length,
    columns,
    headers,
    rows,
    numericColumns: columns.filter(c => c.type === 'numeric').map(c => c.name),
    categoricalColumns: columns.filter(c => c.type === 'categorical').map(c => c.name)
  };
}

function suggestTask(profile, targetColumn) {
  if (!targetColumn || targetColumn === '__none__') {
    return { task: 'clustering', message: `No target selected. Clustering recommended (${profile.numericColumns.length} numeric features available).`, icon: 'layers' };
  }
  const tc = profile.columns.find(c => c.name === targetColumn);
  if (!tc) return { task: 'unknown', message: 'Target column not found.', icon: 'alert' };
  if (tc.type === 'numeric') {
    if (tc.uniqueCount === 2) return { task: 'classification', message: `Binary Classification detected: "${targetColumn}" has 2 unique values.`, icon: 'target' };
    const ratio = tc.uniqueCount / profile.rowCount;
    if (ratio < 0.05) return { task: 'classification', message: `Classification detected: "${targetColumn}" has ${tc.uniqueCount} discrete values.`, icon: 'target' };
    return { task: 'regression', message: `Regression detected: "${targetColumn}" is continuous (${tc.uniqueCount} unique values, range ${tc.min?.toFixed(1)}–${tc.max?.toFixed(1)}).`, icon: 'trending' };
  }
  return { task: 'classification', message: `Classification detected: "${targetColumn}" is categorical (${tc.uniqueCount} classes).`, icon: 'target' };
}

// ==================== FEATURE PREPARATION ====================

function detectProblemType(values) {
  if (values.some(v => typeof v !== 'number')) return 'classification';
  const uniqueValues = [...new Set(values)];
  if (uniqueValues.length === 2) return 'classification';
  if (uniqueValues.length / values.length < 0.05) return 'classification';
  return 'regression';
}

function prepareFeatures(rows, targetCol) {
  const allCols = Object.keys(rows[0]).filter(k => k !== targetCol);
  const leakageKeywords = ['id', '_id', 'date', 'year', 'added', 'created', 'updated'];
  const leakageCols = [];
  const safeCols = allCols.filter(col => {
    const low = col.toLowerCase();
    const isLeak = leakageKeywords.some(kw => low.includes(kw));
    if (isLeak) leakageCols.push(col);
    return !isLeak;
  });
  const numericCols = [], categoricalCols = [], textCols = [];
  safeCols.forEach(col => {
    if (rows.every(row => typeof row[col] === 'number')) { numericCols.push(col); }
    else {
      const avgLen = rows.reduce((s, r) => s + String(r[col]).length, 0) / rows.length;
      (avgLen > 20 ? textCols : categoricalCols).push(col);
    }
  });
  const encodingMap = {};
  categoricalCols.forEach(col => {
    encodingMap[col] = [...new Set(rows.map(r => String(r[col])))].sort();
  });
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

// ==================== LINEAR ALGEBRA ====================

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) { if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row; }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    if (Math.abs(M[col][col]) < 1e-10) continue;
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(M[i][i]) < 1e-10) continue;
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

// ==================== MODEL TRAINING ====================

function trainLinearRegression(X, y) {
  const n = X.length, p = X[0].length;
  const Xa = X.map(row => [1, ...row]), pa = p + 1;
  const XtX = Array(pa).fill(null).map(() => Array(pa).fill(0));
  for (let i = 0; i < pa; i++) for (let j = 0; j < pa; j++) for (let k = 0; k < n; k++) XtX[i][j] += Xa[k][i] * Xa[k][j];
  for (let i = 0; i < pa; i++) XtX[i][i] += 0.01;
  const Xty = Array(pa).fill(0);
  for (let i = 0; i < pa; i++) for (let k = 0; k < n; k++) Xty[i] += Xa[k][i] * y[k];
  const beta = solveLinearSystem(XtX, Xty);
  return { type: 'linear_regression', coefficients: beta };
}

function trainLogisticRegression(X, y) {
  const lin = trainLinearRegression(X, y);
  return { type: 'logistic_regression', coefficients: lin.coefficients };
}

function trainBaseline(y, problemType) {
  if (problemType === 'regression') {
    return { type: 'baseline', coefficients: [y.reduce((a, b) => a + b, 0) / y.length] };
  }
  const counts = {};
  y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
  return { type: 'baseline', coefficients: [Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0])] };
}

function predictSingle(coefficients, type, x) {
  const z = coefficients[0] + x.reduce((s, v, i) => s + v * coefficients[i + 1], 0);
  if (type === 'logistic_regression') {
    const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
    return sig >= 0.5 ? 1 : 0;
  }
  if (type === 'baseline') return coefficients[0];
  return z;
}

function calculateMetrics(actual, predicted, problemType) {
  if (problemType === 'regression') {
    const meanA = actual.reduce((a, b) => a + b, 0) / actual.length;
    const ssTot = actual.reduce((s, v) => s + (v - meanA) ** 2, 0);
    const ssRes = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
    const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
    const mae = actual.reduce((s, v, i) => s + Math.abs(v - predicted[i]), 0) / actual.length;
    const mse = ssRes / actual.length;
    return { train_r2: r2, train_mae: mae, train_mse: mse, train_rmse: Math.sqrt(mse) };
  }
  const correct = actual.filter((v, i) => v === predicted[i]).length;
  return { train_accuracy: correct / actual.length };
}

function extractFeatureImportance(coefficients, featureNames) {
  const weights = coefficients.slice(1).map(c => Math.abs(c));
  const total = weights.reduce((a, b) => a + b, 0);
  if (total === 0) return [];
  return featureNames.map((name, i) => ({ feature: name, importance: weights[i] / total }))
    .sort((a, b) => b.importance - a.importance).slice(0, 10);
}

function prepareInputForPrediction(inputRows, modelData) {
  const { numericCols, categoricalCols, encodingMap } = modelData;
  return inputRows.map(row => {
    const f = [];
    numericCols.forEach(col => f.push(Number(row[col]) || 0));
    categoricalCols.forEach(col => encodingMap[col].slice(1).forEach(val => f.push(String(row[col] || '') === val ? 1 : 0)));
    return f;
  });
}

// ==================== CLUSTERING ====================

function runKMeansClustering(rows, numericCols, k) {
  const stats = {};
  numericCols.forEach(col => {
    const vals = rows.map(r => typeof r[col] === 'number' ? r[col] : 0);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length) || 1;
    stats[col] = { mean, std };
  });
  const Xstd = rows.map(row => numericCols.map(col => {
    const v = typeof row[col] === 'number' ? row[col] : stats[col].mean;
    return (v - stats[col].mean) / stats[col].std;
  }));
  const result = kmeans(Xstd, k, { initialization: 'kmeans++' });
  const clusterStats = Array.from({ length: k }, (_, i) => {
    const indices = result.clusters.reduce((arr, c, idx) => { if (c === i) arr.push(idx); return arr; }, []);
    const means = numericCols.map((col, j) => ({
      feature: col,
      mean: indices.length > 0 ? indices.reduce((s, idx) => s + (typeof rows[idx][col] === 'number' ? rows[idx][col] : 0), 0) / indices.length : 0
    }));
    return { clusterId: i, size: indices.length, means };
  });
  const feat1 = numericCols[0] || 'x';
  const feat2 = numericCols[1] || numericCols[0] || 'y';
  const points = rows.map((row, idx) => ({
    x: typeof row[feat1] === 'number' ? row[feat1] : 0,
    y: typeof row[feat2] === 'number' ? row[feat2] : 0,
    cluster: result.clusters[idx],
    index: idx
  }));
  return { clusters: result.clusters, clusterStats, points, k, features: numericCols, xFeature: feat1, yFeature: feat2 };
}

// ==================== ANOMALY DETECTION ====================

function detectAnomalies(rows, numericCols, method, threshold) {
  const anomalies = {};
  const anomalyRows = new Set();
  const columnStats = {};

  numericCols.forEach(col => {
    const values = rows.map(r => r[col]).filter(v => typeof v === 'number');
    if (values.length === 0) return;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    columnStats[col] = { mean, std, q1, q3, iqr };
    anomalies[col] = [];

    if (method === 'zscore') {
      rows.forEach((row, idx) => {
        if (typeof row[col] !== 'number' || std === 0) return;
        const z = Math.abs((row[col] - mean) / std);
        if (z > threshold) { anomalies[col].push({ index: idx, value: row[col], score: z }); anomalyRows.add(idx); }
      });
    } else {
      const lower = q1 - 1.5 * iqr;
      const upper = q3 + 1.5 * iqr;
      rows.forEach((row, idx) => {
        if (typeof row[col] !== 'number') return;
        if (row[col] < lower || row[col] > upper) {
          anomalies[col].push({ index: idx, value: row[col], bound: row[col] < lower ? 'below' : 'above' });
          anomalyRows.add(idx);
        }
      });
    }
  });

  const feat1 = numericCols[0], feat2 = numericCols[1] || numericCols[0];
  const normalPoints = [], anomalyPoints = [];
  rows.forEach((row, idx) => {
    const pt = { x: typeof row[feat1] === 'number' ? row[feat1] : 0, y: typeof row[feat2] === 'number' ? row[feat2] : 0, index: idx };
    (anomalyRows.has(idx) ? anomalyPoints : normalPoints).push(pt);
  });

  return { anomalies, anomalyRowIndices: [...anomalyRows], totalAnomalies: anomalyRows.size, totalRows: rows.length, method, threshold, columnStats, normalPoints, anomalyPoints, xFeature: feat1, yFeature: feat2 };
}

// ==================== APP COMPONENT ====================

function App() {
  // Core UI state
  const [activeView, setActiveView] = useState('dashboard');
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // Shared data state
  const [csvText, setCsvText] = useState('');
  const [columns, setColumns] = useState([]);
  const [dataProfile, setDataProfile] = useState(null);

  // Analysis/Training state
  const [targetColumn, setTargetColumn] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);

  // Model state
  const [models, setModels] = useState([]);

  // Prediction state
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);

  // Clustering state
  const [numClusters, setNumClusters] = useState(3);
  const [clusterResult, setClusterResult] = useState(null);

  // Anomaly state
  const [anomalyMethod, setAnomalyMethod] = useState('zscore');
  const [anomalyThreshold, setAnomalyThreshold] = useState(3);
  const [anomalyResult, setAnomalyResult] = useState(null);

  // ==================== LOCALSTORAGE PERSISTENCE ====================
  const [hasLoadedFromStorage, setHasLoadedFromStorage] = useState(false);
  
  // Load models from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('automl_models');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setModels(parsed);
        }
      }
    } catch (e) { console.error('Failed to load models:', e); }
    setHasLoadedFromStorage(true);
  }, []);

  // Save models to localStorage only after initial load is complete
  useEffect(() => {
    if (!hasLoadedFromStorage) return; // Skip saving until we've tried to load first
    try {
      const serializable = models.map(m => ({
        ...m,
        modelData: { coefficients: m.modelData.coefficients, featureNames: m.modelData.featureNames, numericCols: m.modelData.numericCols, categoricalCols: m.modelData.categoricalCols, encodingMap: m.modelData.encodingMap, targetEncoding: m.modelData.targetEncoding, type: m.modelData.type }
      }));
      localStorage.setItem('automl_models', JSON.stringify(serializable));
    } catch (e) { console.error('Failed to save models:', e); }
  }, [models, hasLoadedFromStorage]);

  // ==================== SAMPLE DATASETS ====================
  const sampleDatasets = [
    { name: 'Loan Approval', description: 'Binary classification', data: `age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1` },
    { name: 'House Prices', description: 'Regression', data: `size,bedrooms,age,location_score,price\n1200,2,5,7,250000\n1800,3,10,8,380000\n2500,4,3,9,520000\n1000,1,15,6,180000\n2200,3,7,8,450000\n1500,2,8,7,300000\n3000,5,2,9,620000\n1100,1,20,5,170000\n1900,3,5,8,400000\n2800,4,1,10,580000` },
    { name: 'Insurance Costs', description: 'Financial regression', data: `age,sex,bmi,children,smoker,region,charges\n19,female,27.9,0,yes,southwest,16884.92\n18,male,33.77,1,no,southeast,1725.55\n28,male,33.0,3,no,southeast,4449.46\n33,male,22.705,0,no,northwest,21984.47\n32,male,28.88,0,no,northwest,3866.86\n31,female,25.74,0,no,southeast,3756.62\n46,female,33.44,1,no,southeast,8240.59\n37,female,27.74,3,no,northwest,7281.51\n37,male,29.83,2,no,northeast,6406.41\n60,female,25.84,0,no,northwest,28923.14\n25,male,26.22,0,no,northeast,2721.32\n62,female,26.29,0,yes,southeast,27808.73\n23,male,34.4,0,no,southwest,1826.84\n56,female,39.82,0,no,southeast,11090.72\n27,male,42.13,0,yes,southeast,39611.76\n19,male,24.6,1,no,southwest,1837.24\n52,female,30.78,1,no,northeast,10797.34\n23,female,23.845,0,no,northeast,2395.17\n56,male,40.3,0,no,southwest,10602.39\n30,male,35.3,0,yes,southwest,36837.47` },
    { name: 'TV Shows', description: 'Text features', data: `show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description\ns1,TV Show,Breaking Bad,Vince Gilligan,Bryan Cranston,United States,July 1 2020,2008,TV-MA,5 Seasons,Crime TV Shows,A high school chemistry teacher turned meth producer teams up with a former student\ns2,Movie,The Shawshank Redemption,Frank Darabont,Tim Robbins,United States,June 15 2019,1994,R,142 min,Dramas,Two imprisoned men bond over a number of years finding redemption through acts of common decency\ns3,TV Show,Stranger Things,The Duffer Brothers,Millie Bobby Brown,United States,July 15 2016,2016,TV-14,4 Seasons,Sci-Fi TV Shows,When a young boy disappears his mother and friends must confront terrifying supernatural forces\ns4,Movie,The Dark Knight,Christopher Nolan,Christian Bale,United States,January 1 2021,2008,PG-13,152 min,Action & Adventure,When the menace known as the Joker wreaks havoc on Gotham Batman must accept one of the greatest tests\ns5,TV Show,Game of Thrones,David Benioff,Emilia Clarke,United States,April 17 2019,2011,TV-MA,8 Seasons,Fantasy TV Shows,Nine noble families fight for control over the lands of Westeros while an ancient enemy returns` }
  ];

  // ==================== COMPUTED STATS ====================
  const stats = useMemo(() => {
    const totalModels = models.length;
    let avgMetric = 0;
    if (totalModels > 0) {
      const metrics = models.map(m => m.problemType === 'classification' ? (m.metrics?.train_accuracy || 0) : (m.metrics?.train_r2 || 0));
      avgMetric = metrics.reduce((a, b) => a + b, 0) / metrics.length;
    }
    return { totalModels, avgMetric, totalTrainings: totalModels, bestModel: totalModels > 0 ? models[models.length - 1].algorithm : '--' };
  }, [models]);

  const taskSuggestion = useMemo(() => {
    if (!dataProfile) return null;
    return suggestTask(dataProfile, targetColumn);
  }, [dataProfile, targetColumn]);

  // ==================== DATA HANDLERS ====================
  const handleCsvTextChange = useCallback((text) => {
    setCsvText(text);
    setTrainingResult(null);
    setClusterResult(null);
    setAnomalyResult(null);
    if (text.trim()) {
      const profile = profileDataset(text);
      setDataProfile(profile);
      setColumns(profile?.headers || []);
    } else {
      setDataProfile(null);
      setColumns([]);
    }
  }, []);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) { const reader = new FileReader(); reader.onload = (e) => handleCsvTextChange(e.target.result); reader.readAsText(file); }
  };

  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover'); };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false);
    if (e.dataTransfer.files?.[0]) { const reader = new FileReader(); reader.onload = (ev) => handleCsvTextChange(ev.target.result); reader.readAsText(e.dataTransfer.files[0]); }
  };

  // ==================== TRAINING ====================
  const handleTrain = () => {
    setError(''); setTrainingResult(null);
    if (!csvText) { setError('Please provide CSV data'); return; }
    if (!targetColumn || targetColumn === '__none__') { setError('Please select a target column for training'); return; }
    setIsTraining(true);
    const startTime = performance.now();
    try {
      const { rows } = parseCSV(csvText);
      if (!rows.length) throw new Error('No data rows found');
      const prepared = prepareFeatures(rows, targetColumn);
      const { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols } = prepared;
      if (featureNames.length === 0) throw new Error('No usable features after preprocessing');
      if (X.length < 2) throw new Error('Need at least 2 data rows');
      const problemType = detectProblemType(y);
      const leaderboard = [];
      const mainAlgo = problemType === 'regression' ? 'linear_regression' : 'logistic_regression';
      const mainModel = problemType === 'regression' ? trainLinearRegression(X, y) : trainLogisticRegression(X, y);
      const mainPreds = X.map(x => predictSingle(mainModel.coefficients, mainModel.type, x));
      const mainMetrics = calculateMetrics(y, mainPreds, problemType);
      const mainFI = extractFeatureImportance(mainModel.coefficients, featureNames);
      const mainId = generateId();
      const mainDur = (performance.now() - startTime) / 1000;
      leaderboard.push({ modelId: mainId, algorithm: mainAlgo, status: 'ok', metrics: mainMetrics, featureImportance: mainFI, durationSec: mainDur });
      const baseModel = trainBaseline(y, problemType);
      const basePreds = X.map(() => predictSingle(baseModel.coefficients, baseModel.type, []));
      const baseMetrics = calculateMetrics(y, basePreds, problemType);
      leaderboard.push({ modelId: generateId(), algorithm: 'baseline', status: 'ok', metrics: baseMetrics, featureImportance: [], durationSec: (performance.now() - startTime) / 1000 });
      const metricKey = problemType === 'classification' ? 'train_accuracy' : 'train_r2';
      leaderboard.sort((a, b) => (b.metrics[metricKey] || 0) - (a.metrics[metricKey] || 0));
      let predictionsVsActual = null, residuals = null, residualStats = null;
      if (problemType === 'regression') {
        predictionsVsActual = { actual: [...y], predicted: mainPreds };
        const resArr = y.map((v, i) => v - mainPreds[i]);
        residuals = resArr;
        const mr = resArr.reduce((a, b) => a + b, 0) / resArr.length;
        const sr = Math.sqrt(resArr.reduce((s, v) => s + (v - mr) ** 2, 0) / resArr.length);
        residualStats = { mean: mr, std: sr, mean_abs: resArr.reduce((s, v) => s + Math.abs(v), 0) / resArr.length, predictive_power: Math.abs(mr) < sr * 0.1 ? 'Good' : 'Low' };
      }
      setModels(prev => [...prev, {
        modelId: mainId, algorithm: mainAlgo, problemType, metrics: mainMetrics, featureImportance: mainFI,
        createdAt: new Date().toISOString(), durationSec: mainDur,
        modelData: { coefficients: mainModel.coefficients, featureNames, numericCols, categoricalCols, encodingMap, targetEncoding, type: mainModel.type }
      }]);
      setTrainingResult({
        status: 'success', problemType, bestModel: leaderboard[0], leaderboard, totalTime: (performance.now() - startTime) / 1000,
        dataInfo: { numSamples: rows.length, numFeatures: featureNames.length, targetColumn, columns: featureNames, removedLeakageColumns: leakageCols, textColumns: textCols, numericColumns: numericCols },
        predictionsVsActual, residuals, residualStats
      });
    } catch (err) { setError(err.message || 'Training failed'); }
    finally { setIsTraining(false); }
  };

  // ==================== PREDICTION ====================
  const handlePredict = () => {
    setError(''); setPredictionResult(null);
    const activeModel = models[models.length - 1];
    if (!activeModel) { setError('No trained model available.'); return; }
    if (!predictionInput) { setError('Please provide prediction data'); return; }
    try {
      const rawData = JSON.parse(predictionInput);
      const items = Array.isArray(rawData) ? rawData : [rawData];
      const featureVectors = prepareInputForPrediction(items, activeModel.modelData);
      const { coefficients, type, targetEncoding } = activeModel.modelData;
      const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
      const predictions = featureVectors.map(x => {
        const z = coefficients[0] + x.reduce((s, v, i) => s + v * coefficients[i + 1], 0);
        if (type === 'logistic_regression') { const cls = sigmoid(z) >= 0.5 ? 1 : 0; return targetEncoding ? targetEncoding[cls] : cls; }
        return z;
      });
      const probabilities = type === 'logistic_regression' ? featureVectors.map(x => { const z = coefficients[0] + x.reduce((s, v, i) => s + v * coefficients[i + 1], 0); const p = sigmoid(z); return [1 - p, p]; }) : null;
      setPredictionResult({ status: 'success', modelId: activeModel.modelId, algorithm: activeModel.algorithm, predictions, probabilities, problemType: activeModel.problemType });
    } catch (err) { setError('Prediction failed: ' + err.message); }
  };

  // ==================== CLUSTERING ====================
  const handleClustering = () => {
    setError(''); setClusterResult(null);
    if (!dataProfile) { setError('Please upload data first'); return; }
    if (dataProfile.numericColumns.length < 1) { setError('Need at least 1 numeric column for clustering'); return; }
    try {
      const result = runKMeansClustering(dataProfile.rows, dataProfile.numericColumns, numClusters);
      setClusterResult(result);
    } catch (err) { setError('Clustering failed: ' + err.message); }
  };

  // ==================== ANOMALY DETECTION ====================
  const handleAnomalyDetection = () => {
    setError(''); setAnomalyResult(null);
    if (!dataProfile) { setError('Please upload data first'); return; }
    if (dataProfile.numericColumns.length < 1) { setError('Need at least 1 numeric column'); return; }
    try {
      const result = detectAnomalies(dataProfile.rows, dataProfile.numericColumns, anomalyMethod, anomalyThreshold);
      setAnomalyResult(result);
    } catch (err) { setError('Anomaly detection failed: ' + err.message); }
  };

  // ==================== MODEL MANAGEMENT ====================
  const handleDeleteModel = (modelId) => setModels(prev => prev.filter(m => m.modelId !== modelId));
  const handleDownloadModel = (modelId) => {
    const model = models.find(m => m.modelId === modelId);
    if (!model) return;
    const blob = new Blob([JSON.stringify({ modelId: model.modelId, algorithm: model.algorithm, problemType: model.problemType, metrics: model.metrics, modelData: { coefficients: model.modelData.coefficients, featureNames: model.modelData.featureNames, type: model.modelData.type }, createdAt: model.createdAt }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `${model.algorithm}_${modelId.substring(0, 8)}.json`;
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  };

  // ==================== SUB-COMPONENTS ====================
  const StatCard = ({ title, value, icon: Icon, metricValue }) => (
    <motion.div variants={fadeInUp}>
      <Card className="hover:shadow-lg transition-shadow duration-300" data-testid={`stat-card-${title.toLowerCase().replace(/\s+/g, '-')}`}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">{title}</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold tracking-tight" data-testid={`stat-value-${title.toLowerCase().replace(/\s+/g, '-')}`}>{value}</h3>
                {metricValue !== undefined && (
                  <Badge variant={metricValue > 0 ? 'default' : 'secondary'} className="gap-1" data-testid={`stat-trend-${title.toLowerCase().replace(/\s+/g, '-')}`}>
                    {metricValue > 0 ? <><ArrowUpRight className="h-3 w-3" />{`+${metricValue}`}</> : 'N/A'}
                  </Badge>
                )}
              </div>
            </div>
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center"><Icon className="h-6 w-6 text-primary" /></div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );

  const DataUploadMini = ({ onDataLoaded }) => (
    <Card data-testid="data-upload-mini">
      <CardContent className="p-6">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Upload className="h-12 w-12 text-muted-foreground/50 mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Data Loaded</h3>
          <p className="text-sm text-muted-foreground mb-4">Upload data in the Analysis tab or select a sample dataset below</p>
          <div className="flex gap-2 flex-wrap justify-center">
            {sampleDatasets.slice(0, 3).map((ds, i) => (
              <Button key={i} variant="outline" size="sm" onClick={() => { handleCsvTextChange(ds.data); if (onDataLoaded) onDataLoaded(); }} data-testid={`mini-sample-${i}`}>
                {ds.name}
              </Button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  // ==================== RENDER ====================
  return (
    <div className="min-h-screen bg-background" data-testid="app-root">
      {/* Sidebar */}
      <motion.aside initial={{ x: -300 }} animate={{ x: 0 }} className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-sidebar" data-testid="app-sidebar">
        <div className="flex h-full flex-col gap-2">
          <div className="flex h-16 items-center border-b border-sidebar-border px-6">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground"><Brain className="h-6 w-6" /></div>
              <div><h1 className="text-lg font-bold text-sidebar-foreground">AutoML</h1><p className="text-xs text-sidebar-foreground/60">Universal Dashboard</p></div>
            </div>
          </div>
          <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'analysis', label: 'Analysis', icon: Zap },
              { id: 'predict', label: 'Predictions', icon: Sparkles },
              { id: 'clusters', label: 'Clusters', icon: Layers },
              { id: 'anomalies', label: 'Anomalies', icon: ShieldAlert },
              { id: 'models', label: 'Model Library', icon: Database },
            ].map((item) => (
              <Button key={item.id} variant={activeView === item.id ? 'secondary' : 'ghost'} className="w-full justify-start gap-3" onClick={() => setActiveView(item.id)} data-testid={`nav-${item.id}`}>
                <item.icon className="h-4 w-4" />{item.label}
              </Button>
            ))}
          </nav>
          <div className="border-t border-sidebar-border p-4">
            <Card className="bg-sidebar-accent"><CardContent className="p-4"><p className="text-xs font-medium text-sidebar-foreground">Client-Side ML</p><p className="text-xs text-sidebar-foreground/70">All analysis runs in your browser</p></CardContent></Card>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <div className="pl-64">
        <motion.header initial={{ y: -100 }} animate={{ y: 0 }} className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-16 items-center justify-between px-8">
            <div>
              <h2 className="text-2xl font-bold tracking-tight" data-testid="page-title">
                {activeView === 'dashboard' && 'Dashboard'}
                {activeView === 'analysis' && 'Universal Analysis'}
                {activeView === 'predict' && 'Make Predictions'}
                {activeView === 'clusters' && 'K-Means Clustering'}
                {activeView === 'anomalies' && 'Anomaly Detection'}
                {activeView === 'models' && 'Model Library'}
              </h2>
              <p className="text-sm text-muted-foreground">
                {activeView === 'dashboard' && 'Monitor your ML operations'}
                {activeView === 'analysis' && 'Upload data, auto-detect tasks, and train models'}
                {activeView === 'predict' && 'Generate predictions from trained models'}
                {activeView === 'clusters' && 'Discover patterns with K-Means clustering'}
                {activeView === 'anomalies' && 'Detect outliers in your data'}
                {activeView === 'models' && 'Manage and explore your models'}
              </p>
            </div>
            <Button variant="outline" size="icon"><Settings className="h-4 w-4" /></Button>
          </div>
        </motion.header>

        <AnimatePresence>
          {error && (
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="mx-8 mt-4" data-testid="error-banner">
              <Card className="border-destructive bg-destructive/10"><CardContent className="p-4"><p className="text-sm text-destructive font-medium flex items-center gap-2"><AlertCircle className="h-4 w-4" /> {error}</p></CardContent></Card>
            </motion.div>
          )}
        </AnimatePresence>

        <main className="p-8">
          <AnimatePresence mode="wait">

            {/* ==================== DASHBOARD ==================== */}
            {activeView === 'dashboard' && (
              <motion.div key="dashboard" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-8" data-testid="dashboard-view">
                <motion.div variants={staggerContainer} className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                  <StatCard title="Total Models" value={stats.totalModels} metricValue={stats.totalModels} icon={Database} />
                  <StatCard title="Avg Metric" value={stats.totalModels > 0 ? `${(stats.avgMetric * 100).toFixed(0)}%` : '0%'} metricValue={stats.totalModels > 0 ? `${(stats.avgMetric * 100).toFixed(0)}%` : 0} icon={TrendingUp} />
                  <StatCard title="Total Trainings" value={stats.totalTrainings} metricValue={stats.totalTrainings} icon={Activity} />
                  <StatCard title="Best Algorithm" value={stats.bestModel} icon={Sparkles} />
                </motion.div>
                <div className="grid gap-6 lg:grid-cols-2">
                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]"><CardHeader><CardTitle>Model Performance</CardTitle><CardDescription>{models.length > 0 ? 'Training metrics per model' : 'Train models to see performance data'}</CardDescription></CardHeader>
                      <CardContent><ResponsiveContainer width="100%" height={280}>
                        <LineChart data={models.map((m, i) => ({ name: `Model ${i + 1}`, metric: m.problemType === 'classification' ? (m.metrics?.train_accuracy || 0) : (m.metrics?.train_r2 || 0) }))}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" /><XAxis dataKey="name" className="text-xs" /><YAxis className="text-xs" /><Tooltip /><Line type="monotone" dataKey="metric" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ fill: 'hsl(var(--primary))' }} />
                        </LineChart></ResponsiveContainer></CardContent></Card>
                  </motion.div>
                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]"><CardHeader><CardTitle>Algorithm Distribution</CardTitle><CardDescription>{models.length > 0 ? 'Usage by algorithm type' : 'Train models to see distribution'}</CardDescription></CardHeader>
                      <CardContent><ResponsiveContainer width="100%" height={280}>
                        <BarChart data={(() => { const c = {}; models.forEach(m => { c[m.algorithm] = (c[m.algorithm] || 0) + 1; }); return Object.entries(c).map(([name, count]) => ({ name, count })); })()}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" /><XAxis dataKey="name" className="text-xs" /><YAxis className="text-xs" /><Tooltip /><Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                        </BarChart></ResponsiveContainer></CardContent></Card>
                  </motion.div>
                </div>
                <motion.div variants={fadeInUp}>
                  <Card><CardHeader><CardTitle>Recent Training Jobs</CardTitle></CardHeader><CardContent><div className="space-y-4">
                    {models.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center" data-testid="empty-dashboard"><Database className="h-12 w-12 text-muted-foreground/50 mb-4" /><p className="text-muted-foreground">No models trained yet</p><Button className="mt-4" onClick={() => setActiveView('analysis')} data-testid="train-first-model-btn">Train Your First Model</Button></div>
                    ) : models.slice(-5).reverse().map((model, idx) => (
                      <div key={model.modelId} className="flex items-center justify-between rounded-lg border p-4 hover:bg-accent/50 transition-colors" data-testid={`recent-model-${idx}`}>
                        <div className="flex items-center gap-4"><div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center"><Brain className="h-5 w-5 text-primary" /></div><div><p className="font-medium">{model.algorithm}</p><p className="text-sm text-muted-foreground">{model.problemType}</p></div></div>
                        <div className="flex items-center gap-4"><Badge variant="secondary">Success</Badge><ChevronRight className="h-4 w-4 text-muted-foreground" /></div>
                      </div>
                    ))}
                  </div></CardContent></Card>
                </motion.div>
              </motion.div>
            )}

            {/* ==================== ANALYSIS (Enhanced Train) ==================== */}
            {activeView === 'analysis' && (
              <motion.div key="analysis" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="analysis-view">
                {/* Sample Datasets */}
                <motion.div variants={fadeInUp}>
                  <Card><CardHeader><CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Quick Start with Sample Data</CardTitle><CardDescription>Try pre-loaded datasets</CardDescription></CardHeader>
                    <CardContent><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                      {sampleDatasets.map((sample, idx) => (
                        <Card key={idx} className="cursor-pointer hover:shadow-md transition-shadow border-2 hover:border-primary" onClick={() => handleCsvTextChange(sample.data)} data-testid={`sample-dataset-${idx}`}>
                          <CardContent className="p-4"><div className="flex items-center justify-between"><div><p className="font-medium">{sample.name}</p><p className="text-sm text-muted-foreground">{sample.description}</p></div><ChevronRight className="h-5 w-5 text-muted-foreground" /></div></CardContent>
                        </Card>
                      ))}
                    </div></CardContent></Card>
                </motion.div>

                {/* Upload */}
                <motion.div variants={fadeInUp}>
                  <Card><CardHeader><CardTitle className="flex items-center gap-2"><Upload className="h-5 w-5" />Upload Your Data</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                      <div onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'} hover:border-primary hover:bg-accent/50`} data-testid="csv-dropzone">
                        <input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="csv-file-input" />
                        <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" /><p className="text-lg font-medium mb-2">Drop your CSV file here</p><p className="text-sm text-muted-foreground">or click to browse</p>
                      </div>
                      <Separator className="my-6" />
                      <div><label className="text-sm font-medium mb-2 block">Or paste CSV data:</label>
                        <textarea value={csvText} onChange={(e) => handleCsvTextChange(e.target.value)} placeholder="Paste your CSV data here..." rows={6} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2" data-testid="csv-text-input" />
                      </div>
                    </CardContent></Card>
                </motion.div>

                {/* Data Profile - Auto Analysis */}
                {dataProfile && (
                  <motion.div variants={fadeInUp}>
                    <Card className="border-2 border-blue-500/30" data-testid="data-profile-card">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2"><Table2 className="h-5 w-5" />Dataset Profile</CardTitle>
                        <CardDescription>{dataProfile.rowCount} rows x {dataProfile.columnCount} columns</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid gap-3 md:grid-cols-3">
                          <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Numeric Columns</p><p className="text-2xl font-bold text-blue-600">{dataProfile.numericColumns.length}</p></div>
                          <div className="bg-emerald-50 dark:bg-emerald-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Categorical Columns</p><p className="text-2xl font-bold text-emerald-600">{dataProfile.categoricalColumns.length}</p></div>
                          <div className="bg-violet-50 dark:bg-violet-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Total Rows</p><p className="text-2xl font-bold text-violet-600">{dataProfile.rowCount}</p></div>
                        </div>
                        <div className="rounded-md border overflow-auto max-h-64">
                          <table className="w-full text-sm">
                            <thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Column</th><th className="p-2 text-left font-medium">Type</th><th className="p-2 text-left font-medium">Unique</th><th className="p-2 text-left font-medium">Range / Values</th></tr></thead>
                            <tbody>
                              {dataProfile.columns.map((col, idx) => (
                                <tr key={idx} className="border-b last:border-0">
                                  <td className="p-2 font-mono text-xs">{col.name}</td>
                                  <td className="p-2"><Badge variant={col.type === 'numeric' ? 'default' : 'secondary'} className="text-xs">{col.type}</Badge></td>
                                  <td className="p-2 text-xs">{col.uniqueCount}</td>
                                  <td className="p-2 text-xs text-muted-foreground">{col.type === 'numeric' ? `${col.min?.toFixed(1)} — ${col.max?.toFixed(1)} (mean: ${col.mean?.toFixed(1)})` : col.sampleValues.slice(0, 3).join(', ')}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>

                        {/* Task Suggestion Banner */}
                        {taskSuggestion && (
                          <div className={`p-4 rounded-lg border-2 flex items-start gap-3 ${taskSuggestion.task === 'regression' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : taskSuggestion.task === 'classification' ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30' : 'border-violet-500 bg-violet-50 dark:bg-violet-950/30'}`} data-testid="task-suggestion">
                            <Info className="h-5 w-5 mt-0.5 shrink-0" />
                            <div>
                              <p className="font-semibold text-sm">{taskSuggestion.message}</p>
                              {taskSuggestion.task === 'clustering' && (
                                <Button size="sm" className="mt-2" onClick={() => setActiveView('clusters')} data-testid="go-to-clusters-btn">
                                  <Layers className="h-3 w-3 mr-1" />Go to Clustering
                                </Button>
                              )}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {/* Configuration */}
                {columns.length > 0 && (
                  <motion.div variants={fadeInUp}>
                    <Card><CardHeader><CardTitle className="flex items-center gap-2"><Target className="h-5 w-5" />Model Configuration</CardTitle><CardDescription>Select target variable and model type</CardDescription></CardHeader>
                      <CardContent>
                        <div className="grid gap-6 md:grid-cols-2">
                          <div className="space-y-2"><label className="text-sm font-medium">Target Variable</label>
                            <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid="target-column-select">
                              <option value="">-- Select Target --</option>
                              <option value="__none__">No target (use Clustering)</option>
                              {columns.map((col, idx) => <option key={idx} value={col}>{col}</option>)}
                            </select>
                          </div>
                          <div className="space-y-2"><label className="text-sm font-medium">Algorithm</label>
                            <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid="algorithm-select">
                              <option value="auto">Auto (Best Available)</option>
                              <option value="linear">Linear Regression</option>
                              <option value="logistic">Logistic Regression</option>
                            </select>
                          </div>
                        </div>
                        <Button onClick={handleTrain} disabled={isTraining || !targetColumn || targetColumn === '__none__'} className="w-full mt-6 h-12" size="lg" data-testid="start-training-btn">
                          {isTraining ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Training...</> : <><Play className="h-4 w-4 mr-2" />Start Training</>}
                        </Button>
                      </CardContent></Card>
                  </motion.div>
                )}

                {/* Training Results */}
                {trainingResult && (
                  <motion.div variants={fadeInUp} initial="initial" animate="animate" data-testid="training-results">
                    <Card className="border-2 border-primary"><CardHeader><CardTitle className="flex items-center gap-2 text-primary"><Sparkles className="h-5 w-5" />Training Complete!</CardTitle></CardHeader>
                      <CardContent className="space-y-6">
                        <div className="grid gap-4 md:grid-cols-4">
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Problem Type</p><p className="text-2xl font-bold mt-1" data-testid="result-problem-type">{trainingResult.problemType}</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Best Model</p><p className="text-2xl font-bold mt-1" data-testid="result-best-model">{trainingResult.bestModel?.algorithm}</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Training Time</p><p className="text-2xl font-bold mt-1">{trainingResult.totalTime?.toFixed(2)}s</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Samples</p><p className="text-2xl font-bold mt-1">{trainingResult.dataInfo?.numSamples}</p></CardContent></Card>
                        </div>
                        {trainingResult.dataInfo?.removedLeakageColumns?.length > 0 && (
                          <Card className="border-2 border-orange-500 bg-orange-50 dark:bg-orange-950" data-testid="leakage-warning"><CardContent className="p-4"><div className="flex items-start gap-3"><AlertCircle className="h-6 w-6 text-orange-600 mt-0.5 shrink-0" /><div><p className="font-semibold text-orange-900 dark:text-orange-100">Data Leakage Prevention</p><div className="flex flex-wrap gap-2 mt-2">{trainingResult.dataInfo.removedLeakageColumns.map((col, idx) => <Badge key={idx} variant="outline" className="bg-orange-100 dark:bg-orange-900">{col}</Badge>)}</div></div></div></CardContent></Card>
                        )}
                        <Card><CardHeader><CardTitle className="text-lg">Best Model Metrics</CardTitle></CardHeader><CardContent><div className="grid gap-3 md:grid-cols-3">
                          {trainingResult.bestModel?.metrics && Object.entries(trainingResult.bestModel.metrics).map(([key, value]) => (
                            <div key={key} className="bg-muted/50 rounded-lg p-3" data-testid={`metric-${key}`}><p className="text-xs text-muted-foreground uppercase">{key.replace(/_/g, ' ')}</p><p className="text-lg font-bold">{key.includes('mse') || key.includes('mae') || key.includes('rmse') ? value.toFixed(2) : (value * 100).toFixed(2) + '%'}</p></div>
                          ))}
                        </div></CardContent></Card>
                        {trainingResult.problemType === 'regression' && trainingResult.predictionsVsActual && (<>
                          <Card><CardHeader><CardTitle className="text-lg">Predicted vs Actual</CardTitle></CardHeader><CardContent>
                            <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="actual" name="Actual" type="number" /><YAxis dataKey="predicted" name="Predicted" type="number" /><ZAxis range={[50, 50]} /><Tooltip /><Scatter name="Predictions" data={trainingResult.predictionsVsActual.actual.map((a, i) => ({ actual: a, predicted: trainingResult.predictionsVsActual.predicted[i] }))} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer>
                          </CardContent></Card>
                          <Card><CardHeader><CardTitle className="text-lg">Residual Plot</CardTitle></CardHeader><CardContent>
                            <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="predicted" name="Predicted" type="number" /><YAxis dataKey="residual" name="Residual" type="number" /><ZAxis range={[50, 50]} /><Tooltip /><Scatter name="Residuals" data={trainingResult.predictionsVsActual.predicted.map((p, i) => ({ predicted: p, residual: trainingResult.predictionsVsActual.actual[i] - p }))} fill="hsl(var(--chart-2))" /></ScatterChart></ResponsiveContainer>
                            {trainingResult.residualStats?.predictive_power === 'Low' && <div className="mt-4 p-4 bg-destructive/10 border-2 border-destructive rounded-lg" data-testid="low-power-warning"><p className="font-semibold text-destructive">Low Predictive Power Detected</p><p className="text-sm text-muted-foreground mt-1">Mean: {trainingResult.residualStats.mean.toFixed(3)}, Std: {trainingResult.residualStats.std.toFixed(3)}</p></div>}
                          </CardContent></Card>
                        </>)}
                        {trainingResult.bestModel?.featureImportance?.length > 0 && (
                          <Card><CardHeader><CardTitle className="text-lg">Feature Importance</CardTitle></CardHeader><CardContent>
                            <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.bestModel.featureImportance}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} /><YAxis /><Tooltip /><Bar dataKey="importance" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>
                          </CardContent></Card>
                        )}
                        <Card><CardHeader><CardTitle className="text-lg">Leaderboard</CardTitle></CardHeader><CardContent><div className="space-y-2" data-testid="leaderboard">
                          {trainingResult.leaderboard?.map((model, idx) => (
                            <div key={idx} className="flex items-center justify-between p-3 rounded-lg border" data-testid={`leaderboard-entry-${idx}`}>
                              <div className="flex items-center gap-3"><Badge variant={idx === 0 ? 'default' : 'secondary'}>{idx + 1}</Badge><div><p className="font-medium">{model.algorithm}</p><p className="text-xs text-muted-foreground">{model.status === 'ok' ? 'Success' : 'Failed'}</p></div></div>
                              <div className="text-right">{model.metrics && <p className="font-mono text-sm">{(() => { const k = Object.keys(model.metrics)[0]; const v = model.metrics[k]; return k.includes('mse') || k.includes('mae') || k.includes('rmse') ? v.toFixed(2) : (v * 100).toFixed(2) + '%'; })()}</p>}<p className="text-xs text-muted-foreground">{model.durationSec ? `${model.durationSec.toFixed(2)}s` : '-'}</p></div>
                            </div>
                          ))}
                        </div></CardContent></Card>
                      </CardContent></Card>
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* ==================== PREDICTIONS ==================== */}
            {activeView === 'predict' && (
              <motion.div key="predict" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="predict-view">
                {models.length === 0 ? (
                  <motion.div variants={fadeInUp}><Card className="border-2 border-orange-500" data-testid="no-model-warning"><CardContent className="py-16 text-center">
                    <AlertCircle className="h-16 w-16 text-orange-500 mx-auto mb-6" />
                    <h3 className="text-xl font-semibold mb-3" data-testid="no-model-warning-title">No trained model available</h3>
                    <p className="text-muted-foreground mb-6" data-testid="no-model-warning-message">Please train a model in the Analysis section before making predictions.</p>
                    <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="go-to-train-btn"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
                  </CardContent></Card></motion.div>
                ) : (<>
                  <motion.div variants={fadeInUp}><Card data-testid="active-model-card"><CardHeader><CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5" />Active Model</CardTitle><CardDescription>Using the most recently trained model</CardDescription></CardHeader>
                    <CardContent>{(() => { const am = models[models.length - 1]; return (
                      <div className="flex items-center gap-4 p-4 rounded-lg border bg-primary/5 border-primary/20">
                        <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center"><Brain className="h-6 w-6 text-primary" /></div>
                        <div className="flex-1"><p className="font-semibold" data-testid="active-model-algorithm">{am.algorithm}</p><p className="text-sm text-muted-foreground">{am.problemType} &middot; ID: {am.modelId.substring(0, 8)}...</p></div>
                        <Badge variant="default">Active</Badge>
                      </div>
                    ); })()}</CardContent></Card></motion.div>
                  <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Input Data</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                      <textarea value={predictionInput} onChange={(e) => setPredictionInput(e.target.value)} placeholder={'[{"feature1": "value1", "feature2": "value2"}]'} rows={10} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid="prediction-input" />
                      <Button onClick={handlePredict} className="w-full h-12" size="lg" data-testid="generate-predictions-btn"><Sparkles className="h-4 w-4 mr-2" />Generate Predictions</Button>
                    </CardContent></Card></motion.div>
                  {predictionResult && (
                    <motion.div variants={fadeInUp} initial="initial" animate="animate"><Card className="border-2 border-primary" data-testid="prediction-results"><CardHeader><CardTitle className="flex items-center gap-2 text-primary"><Eye className="h-5 w-5" />Prediction Results</CardTitle></CardHeader>
                      <CardContent><div className="space-y-4">{predictionResult.predictions.map((pred, idx) => (
                        <div key={idx} className="bg-muted/50 rounded-lg p-4" data-testid={`prediction-result-${idx}`}>
                          <div className="flex items-center justify-between"><span className="text-sm font-medium text-muted-foreground">Prediction {idx + 1}</span><span className="text-2xl font-bold text-primary">{typeof pred === 'number' ? pred.toFixed(4) : pred}</span></div>
                          {predictionResult.probabilities?.[idx] && <div className="mt-2 text-xs text-muted-foreground">Probabilities: [{predictionResult.probabilities[idx].map(p => p.toFixed(4)).join(', ')}]</div>}
                        </div>
                      ))}</div></CardContent></Card></motion.div>
                  )}
                </>)}
              </motion.div>
            )}

            {/* ==================== CLUSTERS ==================== */}
            {activeView === 'clusters' && (
              <motion.div key="clusters" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="clusters-view">
                {!dataProfile ? <DataUploadMini /> : (<>
                  <motion.div variants={fadeInUp}>
                    <Card data-testid="cluster-config-card">
                      <CardHeader><CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Cluster Configuration</CardTitle>
                        <CardDescription>Using {dataProfile.numericColumns.length} numeric features from {dataProfile.rowCount} rows</CardDescription></CardHeader>
                      <CardContent>
                        <div className="flex items-center gap-6 flex-wrap">
                          <div className="space-y-2 flex-1 min-w-[200px]">
                            <label className="text-sm font-medium">Number of Clusters (k): <span className="text-primary font-bold">{numClusters}</span></label>
                            <input type="range" min={2} max={Math.min(10, dataProfile.rowCount - 1)} value={numClusters} onChange={(e) => setNumClusters(Number(e.target.value))} className="w-full accent-primary" data-testid="cluster-k-slider" />
                            <div className="flex justify-between text-xs text-muted-foreground"><span>2</span><span>{Math.min(10, dataProfile.rowCount - 1)}</span></div>
                          </div>
                          <Button onClick={handleClustering} size="lg" className="h-12" data-testid="run-clustering-btn"><Layers className="h-4 w-4 mr-2" />Run K-Means</Button>
                        </div>
                        <div className="mt-4 text-xs text-muted-foreground"><p>Features: {dataProfile.numericColumns.join(', ')}</p></div>
                      </CardContent>
                    </Card>
                  </motion.div>

                  {clusterResult && (<>
                    {/* Cluster Summary */}
                    <motion.div variants={fadeInUp}>
                      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5">
                        {clusterResult.clusterStats.map((cs) => (
                          <Card key={cs.clusterId} data-testid={`cluster-stat-${cs.clusterId}`}>
                            <CardContent className="p-4 text-center">
                              <div className="h-4 w-full rounded-full mb-3" style={{ backgroundColor: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length], opacity: 0.3 }} />
                              <p className="text-sm text-muted-foreground">Cluster {cs.clusterId}</p>
                              <p className="text-3xl font-bold" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>{cs.size}</p>
                              <p className="text-xs text-muted-foreground">data points</p>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </motion.div>

                    {/* Scatter Plot */}
                    <motion.div variants={fadeInUp}>
                      <Card data-testid="cluster-scatter-chart">
                        <CardHeader><CardTitle>Cluster Visualization</CardTitle><CardDescription>{clusterResult.xFeature} vs {clusterResult.yFeature}</CardDescription></CardHeader>
                        <CardContent>
                          <ResponsiveContainer width="100%" height={400}>
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="x" name={clusterResult.xFeature} type="number" label={{ value: clusterResult.xFeature, position: 'insideBottom', offset: -5 }} />
                              <YAxis dataKey="y" name={clusterResult.yFeature} type="number" label={{ value: clusterResult.yFeature, angle: -90, position: 'insideLeft' }} />
                              <ZAxis range={[60, 60]} />
                              <Tooltip content={({ payload }) => payload?.[0] ? <div className="bg-background border rounded p-2 text-xs"><p>Cluster: {payload[0].payload.cluster}</p><p>{clusterResult.xFeature}: {payload[0].payload.x?.toFixed(2)}</p><p>{clusterResult.yFeature}: {payload[0].payload.y?.toFixed(2)}</p></div> : null} />
                              <Legend />
                              {Array.from({ length: clusterResult.k }, (_, i) => (
                                <Scatter key={i} name={`Cluster ${i}`} data={clusterResult.points.filter(p => p.cluster === i)} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />
                              ))}
                            </ScatterChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    </motion.div>

                    {/* Cluster Size Distribution */}
                    <motion.div variants={fadeInUp}>
                      <Card><CardHeader><CardTitle>Cluster Size Distribution</CardTitle></CardHeader>
                        <CardContent>
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={clusterResult.clusterStats.map(cs => ({ name: `Cluster ${cs.clusterId}`, size: cs.size }))}>
                              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip />
                              <Bar dataKey="size" radius={[4, 4, 0, 0]}>
                                {clusterResult.clusterStats.map((_, i) => <Cell key={i} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />)}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    </motion.div>

                    {/* Cluster Means Table */}
                    <motion.div variants={fadeInUp}>
                      <Card data-testid="cluster-means-table"><CardHeader><CardTitle>Cluster Centers</CardTitle></CardHeader>
                        <CardContent><div className="rounded-md border overflow-auto">
                          <table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Feature</th>
                            {clusterResult.clusterStats.map(cs => <th key={cs.clusterId} className="p-3 text-center font-medium" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>Cluster {cs.clusterId}</th>)}
                          </tr></thead>
                          <tbody>{clusterResult.features.map((feat, fi) => (
                            <tr key={fi} className="border-b last:border-0"><td className="p-3 font-mono text-xs">{feat}</td>
                              {clusterResult.clusterStats.map(cs => <td key={cs.clusterId} className="p-3 text-center text-xs">{cs.means[fi]?.mean?.toFixed(2)}</td>)}
                            </tr>
                          ))}</tbody></table>
                        </div></CardContent></Card>
                    </motion.div>
                  </>)}
                </>)}
              </motion.div>
            )}

            {/* ==================== ANOMALIES ==================== */}
            {activeView === 'anomalies' && (
              <motion.div key="anomalies" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="anomalies-view">
                {!dataProfile ? <DataUploadMini /> : (<>
                  <motion.div variants={fadeInUp}>
                    <Card data-testid="anomaly-config-card">
                      <CardHeader><CardTitle className="flex items-center gap-2"><ShieldAlert className="h-5 w-5" />Detection Configuration</CardTitle>
                        <CardDescription>Analyzing {dataProfile.numericColumns.length} numeric features</CardDescription></CardHeader>
                      <CardContent>
                        <div className="flex items-center gap-6 flex-wrap">
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Method</label>
                            <select value={anomalyMethod} onChange={(e) => setAnomalyMethod(e.target.value)} className="rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid="anomaly-method-select">
                              <option value="zscore">Z-Score</option>
                              <option value="iqr">IQR (Interquartile Range)</option>
                            </select>
                          </div>
                          {anomalyMethod === 'zscore' && (
                            <div className="space-y-2">
                              <label className="text-sm font-medium">Z-Score Threshold: <span className="text-primary font-bold">{anomalyThreshold}</span></label>
                              <input type="range" min={1.5} max={4} step={0.5} value={anomalyThreshold} onChange={(e) => setAnomalyThreshold(Number(e.target.value))} className="w-40 accent-primary" data-testid="anomaly-threshold-slider" />
                            </div>
                          )}
                          <Button onClick={handleAnomalyDetection} size="lg" className="h-12" data-testid="run-anomaly-btn"><ShieldAlert className="h-4 w-4 mr-2" />Detect Anomalies</Button>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>

                  {anomalyResult && (<>
                    {/* Summary */}
                    <motion.div variants={fadeInUp}>
                      <div className="grid gap-4 md:grid-cols-3">
                        <Card data-testid="anomaly-count-card"><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Anomalies Found</p><p className="text-4xl font-bold text-destructive">{anomalyResult.totalAnomalies}</p></CardContent></Card>
                        <Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Normal Points</p><p className="text-4xl font-bold text-primary">{anomalyResult.totalRows - anomalyResult.totalAnomalies}</p></CardContent></Card>
                        <Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Anomaly Rate</p><p className="text-4xl font-bold">{(anomalyResult.totalAnomalies / anomalyResult.totalRows * 100).toFixed(1)}%</p></CardContent></Card>
                      </div>
                    </motion.div>

                    {/* Scatter Plot */}
                    {anomalyResult.xFeature && (
                      <motion.div variants={fadeInUp}>
                        <Card data-testid="anomaly-scatter-chart"><CardHeader><CardTitle>Normal vs Anomaly Points</CardTitle><CardDescription>{anomalyResult.xFeature} vs {anomalyResult.yFeature}</CardDescription></CardHeader>
                          <CardContent><ResponsiveContainer width="100%" height={400}>
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="x" name={anomalyResult.xFeature} type="number" label={{ value: anomalyResult.xFeature, position: 'insideBottom', offset: -5 }} />
                              <YAxis dataKey="y" name={anomalyResult.yFeature} type="number" label={{ value: anomalyResult.yFeature, angle: -90, position: 'insideLeft' }} />
                              <ZAxis range={[60, 60]} />
                              <Tooltip />
                              <Legend />
                              <Scatter name="Normal" data={anomalyResult.normalPoints} fill="hsl(var(--primary))" />
                              <Scatter name="Anomaly" data={anomalyResult.anomalyPoints} fill="hsl(var(--destructive))" />
                            </ScatterChart>
                          </ResponsiveContainer></CardContent></Card>
                      </motion.div>
                    )}

                    {/* Per-Column Anomalies */}
                    <motion.div variants={fadeInUp}>
                      <Card data-testid="anomaly-per-column"><CardHeader><CardTitle>Anomalies by Column</CardTitle></CardHeader>
                        <CardContent>
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={Object.entries(anomalyResult.anomalies).map(([col, items]) => ({ name: col, count: items.length })).filter(d => d.count > 0)}>
                              <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-30} textAnchor="end" height={80} /><YAxis /><Tooltip />
                              <Bar dataKey="count" fill="hsl(var(--destructive))" radius={[4, 4, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </CardContent></Card>
                    </motion.div>

                    {/* Anomalous Rows Table */}
                    {anomalyResult.anomalyRowIndices.length > 0 && (
                      <motion.div variants={fadeInUp}>
                        <Card data-testid="anomaly-rows-table"><CardHeader><CardTitle>Anomalous Data Points</CardTitle><CardDescription>Showing {Math.min(anomalyResult.anomalyRowIndices.length, 20)} of {anomalyResult.anomalyRowIndices.length} anomalous rows</CardDescription></CardHeader>
                          <CardContent><div className="rounded-md border overflow-auto max-h-80">
                            <table className="w-full text-sm"><thead><tr className="border-b bg-destructive/10 sticky top-0"><th className="p-2 text-left font-medium">Row #</th>
                              {dataProfile.numericColumns.slice(0, 6).map(col => <th key={col} className="p-2 text-left font-medium">{col}</th>)}
                            </tr></thead>
                            <tbody>{anomalyResult.anomalyRowIndices.slice(0, 20).map(rowIdx => (
                              <tr key={rowIdx} className="border-b last:border-0 bg-destructive/5"><td className="p-2 font-mono text-xs font-bold">{rowIdx + 1}</td>
                                {dataProfile.numericColumns.slice(0, 6).map(col => {
                                  const isAnomaly = anomalyResult.anomalies[col]?.some(a => a.index === rowIdx);
                                  return <td key={col} className={`p-2 text-xs ${isAnomaly ? 'text-destructive font-bold' : ''}`}>{typeof dataProfile.rows[rowIdx]?.[col] === 'number' ? dataProfile.rows[rowIdx][col].toFixed(2) : '-'}</td>;
                                })}
                              </tr>
                            ))}</tbody></table>
                          </div></CardContent></Card>
                      </motion.div>
                    )}
                  </>)}
                </>)}
              </motion.div>
            )}

            {/* ==================== MODELS ==================== */}
            {activeView === 'models' && (
              <motion.div key="models" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="models-view">
                <Card><CardHeader><div className="flex items-center justify-between"><div><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Model Library</CardTitle><CardDescription>View and manage your trained models</CardDescription></div><Badge variant="secondary" className="text-lg px-4 py-2" data-testid="models-count-badge">{models.length} Models</Badge></div></CardHeader>
                  <CardContent>{models.length === 0 ? (
                    <div className="text-center py-12" data-testid="empty-models"><Database className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" /><h3 className="text-lg font-medium mb-2">No Models Yet</h3><p className="text-muted-foreground mb-6">Train your first model to get started</p><Button onClick={() => setActiveView('analysis')} size="lg" data-testid="train-first-model-from-library"><Zap className="h-4 w-4 mr-2" />Train Your First Model</Button></div>
                  ) : (
                    <div className="rounded-md border"><table className="w-full" data-testid="models-table"><thead><tr className="border-b bg-muted/50"><th className="p-4 text-left text-sm font-medium">Model ID</th><th className="p-4 text-left text-sm font-medium">Algorithm</th><th className="p-4 text-left text-sm font-medium">Type</th><th className="p-4 text-left text-sm font-medium">Created</th><th className="p-4 text-left text-sm font-medium">Actions</th></tr></thead>
                      <tbody>{models.map((model, idx) => (
                        <motion.tr key={model.modelId} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }} className="border-b last:border-0 hover:bg-accent/50 transition-colors" data-testid={`model-row-${idx}`}>
                          <td className="p-4"><div className="flex items-center gap-2"><div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center"><Brain className="h-4 w-4 text-primary" /></div><code className="text-xs font-mono">{model.modelId.substring(0, 12)}...</code></div></td>
                          <td className="p-4"><Badge variant="outline">{model.algorithm}</Badge></td>
                          <td className="p-4"><span className="text-sm">{model.problemType}</span></td>
                          <td className="p-4"><span className="text-sm text-muted-foreground">{new Date(model.createdAt).toLocaleDateString()}</span></td>
                          <td className="p-4"><div className="flex gap-2">
                            <Button variant="ghost" size="sm" onClick={() => setActiveView('predict')} title="Use for predictions" data-testid={`use-model-${idx}`}><Eye className="h-4 w-4" /></Button>
                            <Button variant="ghost" size="sm" onClick={() => handleDownloadModel(model.modelId)} title="Download (JSON)" className="text-primary hover:text-primary" data-testid={`download-model-${idx}`}><Download className="h-4 w-4" /></Button>
                            <Button variant="ghost" size="sm" onClick={() => handleDeleteModel(model.modelId)} title="Delete" data-testid={`delete-model-${idx}`}><Trash2 className="h-4 w-4 text-destructive" /></Button>
                          </div></td>
                        </motion.tr>
                      ))}</tbody></table></div>
                  )}</CardContent></Card>
              </motion.div>
            )}

          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
