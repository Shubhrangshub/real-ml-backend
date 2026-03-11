import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, Sparkles, TrendingUp, Activity, Database, Zap, Settings, Upload, Play,
  Eye, Trash2, ChevronRight, ArrowUpRight, FileText, Target, Cpu, BarChart3,
  Download, AlertCircle
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import './App.css';

// ==================== ANIMATION VARIANTS ====================
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

function detectProblemType(values) {
  if (values.some(v => typeof v !== 'number')) return 'classification';
  const uniqueValues = [...new Set(values)];
  if (uniqueValues.length === 2) return 'classification';
  const uniqueRatio = uniqueValues.length / values.length;
  if (uniqueRatio < 0.05) return 'classification';
  return 'regression';
}

function prepareFeatures(rows, targetCol) {
  const allCols = Object.keys(rows[0]).filter(k => k !== targetCol);

  // Leakage detection
  const leakageKeywords = ['id', '_id', 'date', 'year', 'added', 'created', 'updated'];
  const leakageCols = [];
  const safeCols = allCols.filter(col => {
    const low = col.toLowerCase();
    const isLeak = leakageKeywords.some(kw => low.includes(kw));
    if (isLeak) leakageCols.push(col);
    return !isLeak;
  });

  // Classify columns
  const numericCols = [];
  const categoricalCols = [];
  const textCols = [];

  safeCols.forEach(col => {
    const isNum = rows.every(row => typeof row[col] === 'number');
    if (isNum) {
      numericCols.push(col);
    } else {
      const avgLen = rows.reduce((s, r) => s + String(r[col]).length, 0) / rows.length;
      if (avgLen > 20) {
        textCols.push(col);
      } else {
        categoricalCols.push(col);
      }
    }
  });

  // Build encoding map
  const encodingMap = {};
  categoricalCols.forEach(col => {
    encodingMap[col] = [...new Set(rows.map(r => String(r[col])))].sort();
  });

  // Build feature names
  const featureNames = [...numericCols];
  categoricalCols.forEach(col => {
    encodingMap[col].slice(1).forEach(val => {
      featureNames.push(`${col}_${val}`);
    });
  });

  // Build X matrix
  const X = rows.map(row => {
    const features = [];
    numericCols.forEach(col => features.push(row[col]));
    categoricalCols.forEach(col => {
      encodingMap[col].slice(1).forEach(val => {
        features.push(String(row[col]) === val ? 1 : 0);
      });
    });
    return features;
  });

  // Build y vector
  let y = rows.map(row => row[targetCol]);
  let targetEncoding = null;

  if (y.some(v => typeof v !== 'number')) {
    const uniq = [...new Set(y)].sort();
    targetEncoding = uniq;
    y = y.map(v => uniq.indexOf(String(v)));
  }

  return { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols };
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
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

function trainLinearRegression(X, y) {
  const n = X.length;
  const p = X[0].length;
  const Xa = X.map(row => [1, ...row]);
  const pa = p + 1;

  const XtX = Array(pa).fill(null).map(() => Array(pa).fill(0));
  for (let i = 0; i < pa; i++)
    for (let j = 0; j < pa; j++)
      for (let k = 0; k < n; k++)
        XtX[i][j] += Xa[k][i] * Xa[k][j];

  for (let i = 0; i < pa; i++) XtX[i][i] += 0.01;

  const Xty = Array(pa).fill(0);
  for (let i = 0; i < pa; i++)
    for (let k = 0; k < n; k++)
      Xty[i] += Xa[k][i] * y[k];

  const beta = solveLinearSystem(XtX, Xty);
  return {
    type: 'linear_regression',
    coefficients: beta,
    predict: (x) => beta[0] + x.reduce((s, v, i) => s + v * beta[i + 1], 0)
  };
}

function trainLogisticRegression(X, y) {
  const linModel = trainLinearRegression(X, y);
  const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
  return {
    type: 'logistic_regression',
    coefficients: linModel.coefficients,
    predict: (x) => sigmoid(linModel.predict(x)) >= 0.5 ? 1 : 0,
    predictProba: (x) => sigmoid(linModel.predict(x))
  };
}

function trainBaseline(y, problemType) {
  if (problemType === 'regression') {
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return { type: 'baseline', coefficients: [mean], predict: () => mean };
  }
  const counts = {};
  y.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
  const mode = Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]);
  return { type: 'baseline', coefficients: [mode], predict: () => mode };
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
  return featureNames
    .map((name, i) => ({ feature: name, importance: weights[i] / total }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);
}

function prepareInputForPrediction(inputRows, modelData) {
  const { numericCols, categoricalCols, encodingMap } = modelData;
  return inputRows.map(row => {
    const features = [];
    numericCols.forEach(col => features.push(Number(row[col]) || 0));
    categoricalCols.forEach(col => {
      encodingMap[col].slice(1).forEach(val => {
        features.push(String(row[col] || '') === val ? 1 : 0);
      });
    });
    return features;
  });
}

// ==================== APP COMPONENT ====================

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [csvText, setCsvText] = useState('');
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [models, setModels] = useState([]);
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);

  const sampleDatasets = [
    {
      name: 'Loan Approval',
      description: 'Classification problem',
      data: `age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1`
    },
    {
      name: 'House Prices',
      description: 'Regression problem',
      data: `size,bedrooms,age,location_score,price\n1200,2,5,7,250000\n1800,3,10,8,380000\n2500,4,3,9,520000\n1000,1,15,6,180000\n2200,3,7,8,450000`
    },
    {
      name: 'Insurance Costs',
      description: 'Financial regression with currency context',
      data: `age,sex,bmi,children,smoker,region,charges\n19,female,27.9,0,yes,southwest,16884.92\n18,male,33.77,1,no,southeast,1725.55\n28,male,33.0,3,no,southeast,4449.46\n33,male,22.705,0,no,northwest,21984.47\n32,male,28.88,0,no,northwest,3866.86\n31,female,25.74,0,no,southeast,3756.62\n46,female,33.44,1,no,southeast,8240.59\n37,female,27.74,3,no,northwest,7281.51\n37,male,29.83,2,no,northeast,6406.41\n60,female,25.84,0,no,northwest,28923.14\n25,male,26.22,0,no,northeast,2721.32\n62,female,26.29,0,yes,southeast,27808.73\n23,male,34.4,0,no,southwest,1826.84\n56,female,39.82,0,no,southeast,11090.72\n27,male,42.13,0,yes,southeast,39611.76\n19,male,24.6,1,no,southwest,1837.24\n52,female,30.78,1,no,northeast,10797.34\n23,female,23.845,0,no,northeast,2395.17\n56,male,40.3,0,no,southwest,10602.39\n30,male,35.3,0,yes,southwest,36837.47`
    },
    {
      name: 'TV Shows (Text Analysis)',
      description: 'Regression with text processing',
      data: `show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description\ns1,TV Show,Breaking Bad,Vince Gilligan,Bryan Cranston,United States,July 1 2020,2008,TV-MA,5 Seasons,Crime TV Shows,A high school chemistry teacher turned meth producer teams up with a former student\ns2,Movie,The Shawshank Redemption,Frank Darabont,Tim Robbins,United States,June 15 2019,1994,R,142 min,Dramas,Two imprisoned men bond over a number of years finding redemption through acts of common decency\ns3,TV Show,Stranger Things,The Duffer Brothers,Millie Bobby Brown,United States,July 15 2016,2016,TV-14,4 Seasons,Sci-Fi TV Shows,When a young boy disappears his mother and friends must confront terrifying supernatural forces\ns4,Movie,The Dark Knight,Christopher Nolan,Christian Bale,United States,January 1 2021,2008,PG-13,152 min,Action & Adventure,When the menace known as the Joker wreaks havoc on Gotham Batman must accept one of the greatest tests\ns5,TV Show,Game of Thrones,David Benioff,Emilia Clarke,United States,April 17 2019,2011,TV-MA,8 Seasons,Fantasy TV Shows,Nine noble families fight for control over the lands of Westeros while an ancient enemy returns\ns6,Movie,Inception,Christopher Nolan,Leonardo DiCaprio,United States,March 1 2020,2010,PG-13,148 min,Sci-Fi Movies,A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea\ns7,TV Show,The Crown,Peter Morgan,Claire Foy,United Kingdom,November 4 2016,2016,TV-MA,6 Seasons,British TV Shows,Follows the political rivalries and romance of Queen Elizabeth II reign and the events that shaped the second half\ns8,Movie,Pulp Fiction,Quentin Tarantino,John Travolta,United States,September 1 2020,1994,R,154 min,Crime Movies,The lives of two mob hitmen a boxer and a pair of diner bandits intertwine in four tales of violence and redemption\ns9,TV Show,The Office,Greg Daniels,Steve Carell,United States,January 1 2021,2005,TV-14,9 Seasons,TV Comedies,A mockumentary on a group of typical office workers where the workday consists of ego clashes and inappropriate behavior\ns10,Movie,The Godfather,Francis Ford Coppola,Marlon Brando,United States,August 1 2019,1972,R,175 min,Classic Movies,The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son`
    }
  ];

  // FIX #3 & #4: Dynamic stats computed from actual model data
  const stats = useMemo(() => {
    const totalModels = models.length;
    let avgMetric = 0;
    if (totalModels > 0) {
      const metrics = models.map(m => {
        if (m.problemType === 'classification') return m.metrics?.train_accuracy || 0;
        return m.metrics?.train_r2 || 0;
      });
      avgMetric = metrics.reduce((a, b) => a + b, 0) / metrics.length;
    }
    const bestModel = totalModels > 0 ? models[models.length - 1].algorithm : '--';
    return { totalModels, avgMetric, totalTrainings: totalModels, bestModel };
  }, [models]);

  const parseColumns = (csvData) => {
    const lines = csvData.trim().split('\n');
    if (lines.length > 0) {
      setColumns(lines[0].split(',').map(h => h.trim()));
    }
  };

  const handleCsvTextChange = (text) => {
    setCsvText(text);
    if (text.trim()) parseColumns(text);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => handleCsvTextChange(e.target.result);
      reader.readAsText(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      const reader = new FileReader();
      reader.onload = (event) => handleCsvTextChange(event.target.result);
      reader.readAsText(e.dataTransfer.files[0]);
    }
  };

  const loadSampleData = (sample) => handleCsvTextChange(sample.data);

  // FIX #2: Client-side training with proper model persistence
  const handleTrain = () => {
    setError('');
    setTrainingResult(null);
    if (!csvText) { setError('Please provide CSV data'); return; }
    if (!targetColumn) { setError('Please select a target column'); return; }

    setIsTraining(true);
    const startTime = performance.now();

    try {
      const { rows } = parseCSV(csvText);
      if (!rows.length) throw new Error('No data rows found');

      const prepared = prepareFeatures(rows, targetColumn);
      const { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols } = prepared;

      if (featureNames.length === 0) throw new Error('No usable features after preprocessing');
      if (X.length < 2) throw new Error('Need at least 2 data rows for training');

      const problemType = detectProblemType(y);
      const leaderboard = [];

      // Train main model
      const mainAlgo = problemType === 'regression' ? 'linear_regression' : 'logistic_regression';
      const mainModel = problemType === 'regression'
        ? trainLinearRegression(X, y)
        : trainLogisticRegression(X, y);

      const mainPredictions = X.map(x => mainModel.predict(x));
      const mainMetrics = calculateMetrics(y, mainPredictions, problemType);
      const mainFI = extractFeatureImportance(mainModel.coefficients, featureNames);
      const mainModelId = generateId();
      const mainDuration = (performance.now() - startTime) / 1000;

      leaderboard.push({
        modelId: mainModelId,
        algorithm: mainAlgo,
        status: 'ok',
        metrics: mainMetrics,
        featureImportance: mainFI,
        durationSec: mainDuration
      });

      // Train baseline
      const baselineModel = trainBaseline(y, problemType);
      const baselinePredictions = X.map(() => baselineModel.predict());
      const baselineMetrics = calculateMetrics(y, baselinePredictions, problemType);
      const baselineId = generateId();

      leaderboard.push({
        modelId: baselineId,
        algorithm: 'baseline',
        status: 'ok',
        metrics: baselineMetrics,
        featureImportance: [],
        durationSec: (performance.now() - startTime) / 1000
      });

      // Sort leaderboard by primary metric
      const metricKey = problemType === 'classification' ? 'train_accuracy' : 'train_r2';
      leaderboard.sort((a, b) => (b.metrics[metricKey] || 0) - (a.metrics[metricKey] || 0));

      const bestEntry = leaderboard[0];

      // Build regression visualizations
      let predictionsVsActual = null;
      let residuals = null;
      let residualStats = null;

      if (problemType === 'regression') {
        const bestPreds = bestEntry.modelId === mainModelId ? mainPredictions : baselinePredictions;
        predictionsVsActual = { actual: [...y], predicted: bestPreds };
        const resArray = y.map((v, i) => v - bestPreds[i]);
        residuals = resArray;
        const meanRes = resArray.reduce((a, b) => a + b, 0) / resArray.length;
        const stdRes = Math.sqrt(resArray.reduce((s, v) => s + (v - meanRes) ** 2, 0) / resArray.length);
        residualStats = {
          mean: meanRes,
          std: stdRes,
          mean_abs: resArray.reduce((s, v) => s + Math.abs(v), 0) / resArray.length,
          predictive_power: Math.abs(meanRes) < stdRes * 0.1 ? 'Good' : 'Low'
        };
      }

      const totalTime = (performance.now() - startTime) / 1000;

      // FIX #2: Store model in state with setModels(prev => [...prev, newModel])
      const newModel = {
        modelId: mainModelId,
        algorithm: mainAlgo,
        problemType,
        metrics: mainMetrics,
        featureImportance: mainFI,
        createdAt: new Date().toISOString(),
        durationSec: mainDuration,
        modelData: {
          coefficients: mainModel.coefficients,
          featureNames,
          numericCols,
          categoricalCols,
          encodingMap,
          targetEncoding,
          type: mainModel.type
        }
      };
      setModels(prev => [...prev, newModel]);

      setTrainingResult({
        status: 'success',
        problemType,
        bestModel: bestEntry,
        leaderboard,
        totalTime,
        dataInfo: {
          numSamples: rows.length,
          numFeatures: featureNames.length,
          targetColumn,
          columns: featureNames,
          removedLeakageColumns: leakageCols,
          textColumns: textCols,
          numericColumns: numericCols
        },
        predictionsVsActual,
        residuals,
        residualStats
      });
    } catch (err) {
      setError(err.message || 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  // FIX #5: Predictions use latest model
  const handlePredict = () => {
    setError('');
    setPredictionResult(null);

    const activeModel = models[models.length - 1];
    if (!activeModel) {
      setError('No trained model available. Please train a model first.');
      return;
    }
    if (!predictionInput) {
      setError('Please provide prediction data');
      return;
    }

    try {
      const rawData = JSON.parse(predictionInput);
      const items = Array.isArray(rawData) ? rawData : [rawData];
      const featureVectors = prepareInputForPrediction(items, activeModel.modelData);
      const { coefficients, type, targetEncoding } = activeModel.modelData;

      const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));

      const predictions = featureVectors.map(x => {
        const z = coefficients[0] + x.reduce((s, v, i) => s + v * coefficients[i + 1], 0);
        if (type === 'logistic_regression') {
          const classIdx = sigmoid(z) >= 0.5 ? 1 : 0;
          return targetEncoding ? targetEncoding[classIdx] : classIdx;
        }
        return z;
      });

      const probabilities = type === 'logistic_regression'
        ? featureVectors.map(x => {
            const z = coefficients[0] + x.reduce((s, v, i) => s + v * coefficients[i + 1], 0);
            const p = sigmoid(z);
            return [1 - p, p];
          })
        : null;

      setPredictionResult({
        status: 'success',
        modelId: activeModel.modelId,
        algorithm: activeModel.algorithm,
        predictions,
        probabilities,
        problemType: activeModel.problemType
      });
    } catch (err) {
      setError('Prediction failed: ' + err.message);
    }
  };

  const handleDeleteModel = (modelId) => {
    setModels(prev => prev.filter(m => m.modelId !== modelId));
  };

  const handleDownloadModel = (modelId) => {
    const model = models.find(m => m.modelId === modelId);
    if (!model) return;
    const exportData = {
      modelId: model.modelId,
      algorithm: model.algorithm,
      problemType: model.problemType,
      metrics: model.metrics,
      featureImportance: model.featureImportance,
      modelData: {
        coefficients: model.modelData.coefficients,
        featureNames: model.modelData.featureNames,
        type: model.modelData.type
      },
      createdAt: model.createdAt
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${model.algorithm}_${modelId.substring(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // FIX #4: Dynamic trend indicator - N/A when value is 0
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
                  <Badge
                    variant={metricValue > 0 ? 'default' : 'secondary'}
                    className="gap-1"
                    data-testid={`stat-trend-${title.toLowerCase().replace(/\s+/g, '-')}`}
                  >
                    {metricValue > 0 ? (
                      <><ArrowUpRight className="h-3 w-3" />{`+${metricValue}`}</>
                    ) : (
                      'N/A'
                    )}
                  </Badge>
                )}
              </div>
            </div>
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Icon className="h-6 w-6 text-primary" />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-background" data-testid="app-root">
      {/* Sidebar */}
      <motion.aside
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-sidebar"
        data-testid="app-sidebar"
      >
        <div className="flex h-full flex-col gap-2">
          <div className="flex h-16 items-center border-b border-sidebar-border px-6">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Brain className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-sidebar-foreground">AutoML</h1>
                <p className="text-xs text-sidebar-foreground/60">Master Platform</p>
              </div>
            </div>
          </div>

          <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'train', label: 'Train Models', icon: Zap },
              { id: 'predict', label: 'Predictions', icon: Sparkles },
              { id: 'models', label: 'Model Library', icon: Database },
            ].map((item) => (
              <Button
                key={item.id}
                variant={activeView === item.id ? 'secondary' : 'ghost'}
                className="w-full justify-start gap-3"
                onClick={() => setActiveView(item.id)}
                data-testid={`nav-${item.id}`}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Button>
            ))}
          </nav>

          <div className="border-t border-sidebar-border p-4">
            <Card className="bg-sidebar-accent">
              <CardContent className="p-4">
                <div className="space-y-2">
                  <p className="text-xs font-medium text-sidebar-foreground">Client-Side ML</p>
                  <p className="text-xs text-sidebar-foreground/70">All analysis runs in your browser</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <div className="pl-64">
        <motion.header
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
        >
          <div className="flex h-16 items-center justify-between px-8">
            <div>
              <h2 className="text-2xl font-bold tracking-tight" data-testid="page-title">
                {activeView === 'dashboard' && 'Dashboard'}
                {activeView === 'train' && 'Train New Model'}
                {activeView === 'predict' && 'Make Predictions'}
                {activeView === 'models' && 'Model Library'}
              </h2>
              <p className="text-sm text-muted-foreground">
                {activeView === 'dashboard' && 'Monitor your ML operations'}
                {activeView === 'train' && 'Create and train machine learning models'}
                {activeView === 'predict' && 'Generate predictions from trained models'}
                {activeView === 'models' && 'Manage and explore your models'}
              </p>
            </div>
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </motion.header>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mx-8 mt-4"
              data-testid="error-banner"
            >
              <Card className="border-destructive bg-destructive/10">
                <CardContent className="p-4">
                  <p className="text-sm text-destructive font-medium flex items-center gap-2">
                    <AlertCircle className="h-4 w-4" /> {error}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        <main className="p-8">
          <AnimatePresence mode="wait">
            {/* ==================== DASHBOARD VIEW ==================== */}
            {activeView === 'dashboard' && (
              <motion.div
                key="dashboard"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-8"
                data-testid="dashboard-view"
              >
                <motion.div variants={staggerContainer} className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                  {/* FIX #3: Total Models uses models.length */}
                  <StatCard
                    title="Total Models"
                    value={stats.totalModels}
                    metricValue={stats.totalModels}
                    icon={Database}
                  />
                  <StatCard
                    title="Avg Metric"
                    value={stats.totalModels > 0 ? `${(stats.avgMetric * 100).toFixed(0)}%` : '0%'}
                    metricValue={stats.totalModels > 0 ? `${(stats.avgMetric * 100).toFixed(0)}%` : 0}
                    icon={TrendingUp}
                  />
                  <StatCard
                    title="Total Trainings"
                    value={stats.totalTrainings}
                    metricValue={stats.totalTrainings}
                    icon={Activity}
                  />
                  <StatCard
                    title="Best Algorithm"
                    value={stats.bestModel}
                    icon={Sparkles}
                  />
                </motion.div>

                <div className="grid gap-6 lg:grid-cols-2">
                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]">
                      <CardHeader>
                        <CardTitle>Model Performance</CardTitle>
                        <CardDescription>
                          {models.length > 0 ? 'Training metrics per model' : 'Train models to see performance data'}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={280}>
                          <LineChart data={models.map((m, i) => ({
                            name: `Model ${i + 1}`,
                            metric: m.problemType === 'classification'
                              ? (m.metrics?.train_accuracy || 0)
                              : (m.metrics?.train_r2 || 0)
                          }))}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis dataKey="name" className="text-xs" />
                            <YAxis className="text-xs" />
                            <Tooltip />
                            <Line type="monotone" dataKey="metric" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ fill: 'hsl(var(--primary))' }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </motion.div>

                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]">
                      <CardHeader>
                        <CardTitle>Algorithm Distribution</CardTitle>
                        <CardDescription>
                          {models.length > 0 ? 'Usage by algorithm type' : 'Train models to see distribution'}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={280}>
                          <BarChart data={(() => {
                            const counts = {};
                            models.forEach(m => { counts[m.algorithm] = (counts[m.algorithm] || 0) + 1; });
                            return Object.entries(counts).map(([name, count]) => ({ name, count }));
                          })()}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis dataKey="name" className="text-xs" angle={-15} textAnchor="end" height={80} />
                            <YAxis className="text-xs" />
                            <Tooltip />
                            <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </motion.div>
                </div>

                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Recent Training Jobs</CardTitle>
                      <CardDescription>Latest model training activity</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {models.length === 0 ? (
                          <div className="flex flex-col items-center justify-center py-12 text-center" data-testid="empty-dashboard">
                            <Database className="h-12 w-12 text-muted-foreground/50 mb-4" />
                            <p className="text-muted-foreground">No models trained yet</p>
                            <Button className="mt-4" onClick={() => setActiveView('train')} data-testid="train-first-model-btn">
                              Train Your First Model
                            </Button>
                          </div>
                        ) : (
                          models.slice(-5).reverse().map((model, idx) => (
                            <div key={model.modelId} className="flex items-center justify-between rounded-lg border p-4 hover:bg-accent/50 transition-colors" data-testid={`recent-model-${idx}`}>
                              <div className="flex items-center gap-4">
                                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                                  <Brain className="h-5 w-5 text-primary" />
                                </div>
                                <div>
                                  <p className="font-medium">{model.algorithm}</p>
                                  <p className="text-sm text-muted-foreground">{model.problemType}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-4">
                                <Badge variant="secondary">Success</Badge>
                                <ChevronRight className="h-4 w-4 text-muted-foreground" />
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </motion.div>
            )}

            {/* ==================== TRAIN VIEW ==================== */}
            {activeView === 'train' && (
              <motion.div
                key="train"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-6"
                data-testid="train-view"
              >
                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="h-5 w-5" />
                        Quick Start with Sample Data
                      </CardTitle>
                      <CardDescription>Try our pre-loaded datasets</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4 md:grid-cols-2">
                        {sampleDatasets.map((sample, idx) => (
                          <Card
                            key={idx}
                            className="cursor-pointer hover:shadow-md transition-shadow border-2 hover:border-primary"
                            onClick={() => loadSampleData(sample)}
                            data-testid={`sample-dataset-${idx}`}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-center justify-between">
                                <div>
                                  <p className="font-medium">{sample.name}</p>
                                  <p className="text-sm text-muted-foreground">{sample.description}</p>
                                </div>
                                <ChevronRight className="h-5 w-5 text-muted-foreground" />
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Upload className="h-5 w-5" />
                        Upload Your Data
                      </CardTitle>
                      <CardDescription>Drop a CSV file or paste your data</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                        className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'} hover:border-primary hover:bg-accent/50`}
                        data-testid="csv-dropzone"
                      >
                        <input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="csv-file-input" />
                        <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-lg font-medium mb-2">Drop your CSV file here</p>
                        <p className="text-sm text-muted-foreground">or click to browse</p>
                      </div>
                      <Separator className="my-6" />
                      <div>
                        <label className="text-sm font-medium mb-2 block">Or paste CSV data:</label>
                        <textarea
                          value={csvText}
                          onChange={(e) => handleCsvTextChange(e.target.value)}
                          placeholder="Paste your CSV data here..."
                          rows={8}
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 font-mono"
                          data-testid="csv-text-input"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {columns.length > 0 && (
                  <motion.div variants={fadeInUp}>
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Target className="h-5 w-5" />
                          Model Configuration
                        </CardTitle>
                        <CardDescription>Select target variable and model type</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid gap-6 md:grid-cols-2">
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Target Variable</label>
                            <select
                              value={targetColumn}
                              onChange={(e) => setTargetColumn(e.target.value)}
                              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              data-testid="target-column-select"
                            >
                              <option value="">-- Select Target --</option>
                              {columns.map((col, idx) => (
                                <option key={idx} value={col}>{col}</option>
                              ))}
                            </select>
                          </div>
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Model Type</label>
                            <select
                              value={algorithm}
                              onChange={(e) => setAlgorithm(e.target.value)}
                              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                              data-testid="algorithm-select"
                            >
                              <option value="auto">Auto (Best Available)</option>
                              <option value="linear">Linear Regression</option>
                              <option value="logistic">Logistic Regression</option>
                            </select>
                          </div>
                        </div>
                        <Button
                          onClick={handleTrain}
                          disabled={isTraining || !targetColumn}
                          className="w-full mt-6 h-12"
                          size="lg"
                          data-testid="start-training-btn"
                        >
                          {isTraining ? (
                            <>
                              <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                              Training Models...
                            </>
                          ) : (
                            <>
                              <Play className="h-4 w-4 mr-2" />
                              Start Training
                            </>
                          )}
                        </Button>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {/* Training Results */}
                {trainingResult && (
                  <motion.div variants={fadeInUp} initial="initial" animate="animate" data-testid="training-results">
                    <Card className="border-2 border-primary">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-primary">
                          <Sparkles className="h-5 w-5" />
                          Training Complete!
                        </CardTitle>
                        <CardDescription>Your models have been trained successfully</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        <div className="grid gap-4 md:grid-cols-4">
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Problem Type</p><p className="text-2xl font-bold mt-1" data-testid="result-problem-type">{trainingResult.problemType}</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Best Model</p><p className="text-2xl font-bold mt-1" data-testid="result-best-model">{trainingResult.bestModel?.algorithm}</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Training Time</p><p className="text-2xl font-bold mt-1">{trainingResult.totalTime?.toFixed(2)}s</p></CardContent></Card>
                          <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Samples</p><p className="text-2xl font-bold mt-1">{trainingResult.dataInfo?.numSamples}</p></CardContent></Card>
                        </div>

                        {trainingResult.dataInfo?.removedLeakageColumns?.length > 0 && (
                          <Card className="border-2 border-orange-500 bg-orange-50 dark:bg-orange-950" data-testid="leakage-warning">
                            <CardContent className="p-4">
                              <div className="flex items-start gap-3">
                                <AlertCircle className="h-6 w-6 text-orange-600 mt-0.5 shrink-0" />
                                <div>
                                  <p className="font-semibold text-orange-900 dark:text-orange-100">Data Leakage Prevention</p>
                                  <p className="text-sm text-orange-800 dark:text-orange-200 mt-1">The following columns were automatically removed to prevent data leakage:</p>
                                  <div className="flex flex-wrap gap-2 mt-2">
                                    {trainingResult.dataInfo.removedLeakageColumns.map((col, idx) => (
                                      <Badge key={idx} variant="outline" className="bg-orange-100 dark:bg-orange-900">{col}</Badge>
                                    ))}
                                  </div>
                                  <p className="text-xs text-orange-700 dark:text-orange-300 mt-2">These columns (IDs, dates, years) would artificially inflate accuracy.</p>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        )}

                        <Card>
                          <CardHeader><CardTitle className="text-lg">Best Model Metrics</CardTitle></CardHeader>
                          <CardContent>
                            <div className="grid gap-3 md:grid-cols-3">
                              {trainingResult.bestModel?.metrics && Object.entries(trainingResult.bestModel.metrics).map(([key, value]) => (
                                <div key={key} className="bg-muted/50 rounded-lg p-3" data-testid={`metric-${key}`}>
                                  <p className="text-xs text-muted-foreground uppercase">{key.replace(/_/g, ' ')}</p>
                                  <p className="text-lg font-bold">
                                    {key.includes('mse') || key.includes('mae') || key.includes('rmse')
                                      ? value.toFixed(2)
                                      : (value * 100).toFixed(2) + '%'}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        {trainingResult.problemType === 'regression' && trainingResult.predictionsVsActual && (
                          <>
                            <Card>
                              <CardHeader>
                                <CardTitle className="text-lg">Predicted vs Actual</CardTitle>
                                <CardDescription>Scatter plot showing model predictions against actual values</CardDescription>
                              </CardHeader>
                              <CardContent>
                                <ResponsiveContainer width="100%" height={300}>
                                  <ScatterChart>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="actual" name="Actual" type="number" label={{ value: 'Actual Values', position: 'insideBottom', offset: -5 }} />
                                    <YAxis dataKey="predicted" name="Predicted" type="number" label={{ value: 'Predicted Values', angle: -90, position: 'insideLeft' }} />
                                    <ZAxis range={[50, 50]} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                    <Scatter
                                      name="Predictions"
                                      data={trainingResult.predictionsVsActual.actual.map((actual, idx) => ({
                                        actual, predicted: trainingResult.predictionsVsActual.predicted[idx]
                                      }))}
                                      fill="hsl(var(--primary))"
                                    />
                                  </ScatterChart>
                                </ResponsiveContainer>
                                <p className="text-xs text-muted-foreground text-center mt-2">Points closer to a diagonal line indicate better model performance.</p>
                              </CardContent>
                            </Card>

                            <Card>
                              <CardHeader>
                                <CardTitle className="text-lg">Residual Plot</CardTitle>
                                <CardDescription>Distribution of prediction errors (actual - predicted)</CardDescription>
                              </CardHeader>
                              <CardContent>
                                <ResponsiveContainer width="100%" height={300}>
                                  <ScatterChart>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="predicted" name="Predicted" type="number" label={{ value: 'Predicted Values', position: 'insideBottom', offset: -5 }} />
                                    <YAxis dataKey="residual" name="Residual" type="number" label={{ value: 'Residual (Actual - Predicted)', angle: -90, position: 'insideLeft' }} />
                                    <ZAxis range={[50, 50]} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                    <Scatter
                                      name="Residuals"
                                      data={trainingResult.predictionsVsActual.predicted.map((pred, idx) => ({
                                        predicted: pred,
                                        residual: trainingResult.predictionsVsActual.actual[idx] - pred
                                      }))}
                                      fill="hsl(var(--chart-2))"
                                    />
                                  </ScatterChart>
                                </ResponsiveContainer>
                                <p className="text-xs text-muted-foreground text-center mt-2">Residuals should be randomly scattered around zero.</p>
                                {trainingResult.residualStats?.predictive_power === 'Low' && (
                                  <div className="mt-4 p-4 bg-destructive/10 border-2 border-destructive rounded-lg" data-testid="low-power-warning">
                                    <div className="flex items-start gap-3">
                                      <AlertCircle className="h-6 w-6 text-destructive mt-0.5 shrink-0" />
                                      <div>
                                        <p className="font-semibold text-destructive">Low Predictive Power Detected</p>
                                        <p className="text-sm text-muted-foreground mt-1">The residuals show high variance. Consider adding more relevant features or collecting more data.</p>
                                        <div className="mt-2 text-xs text-muted-foreground">
                                          <p>Mean Residual: {trainingResult.residualStats.mean.toFixed(3)}</p>
                                          <p>Std Residual: {trainingResult.residualStats.std.toFixed(3)}</p>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </CardContent>
                            </Card>
                          </>
                        )}

                        {trainingResult.bestModel?.featureImportance?.length > 0 && (
                          <Card>
                            <CardHeader><CardTitle className="text-lg">Feature Importance</CardTitle></CardHeader>
                            <CardContent>
                              <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={trainingResult.bestModel.featureImportance}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                                  <YAxis />
                                  <Tooltip />
                                  <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              </ResponsiveContainer>
                            </CardContent>
                          </Card>
                        )}

                        <Card>
                          <CardHeader><CardTitle className="text-lg">All Models Leaderboard</CardTitle></CardHeader>
                          <CardContent>
                            <div className="space-y-2" data-testid="leaderboard">
                              {trainingResult.leaderboard?.map((model, idx) => (
                                <div key={idx} className="flex items-center justify-between p-3 rounded-lg border" data-testid={`leaderboard-entry-${idx}`}>
                                  <div className="flex items-center gap-3">
                                    <Badge variant={idx === 0 ? 'default' : 'secondary'}>{idx + 1}</Badge>
                                    <div>
                                      <p className="font-medium">{model.algorithm}</p>
                                      <p className="text-xs text-muted-foreground">{model.status === 'ok' ? 'Success' : 'Failed'}</p>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    {model.metrics && (
                                      <p className="font-mono text-sm">
                                        {(() => {
                                          const key = Object.keys(model.metrics)[0];
                                          const val = model.metrics[key];
                                          return key.includes('mse') || key.includes('mae') || key.includes('rmse')
                                            ? val.toFixed(2)
                                            : (val * 100).toFixed(2) + '%';
                                        })()}
                                      </p>
                                    )}
                                    <p className="text-xs text-muted-foreground">{model.durationSec ? `${model.durationSec.toFixed(2)}s` : '-'}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* ==================== PREDICT VIEW (FIX #1) ==================== */}
            {activeView === 'predict' && (
              <motion.div
                key="predict"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-6"
                data-testid="predict-view"
              >
                {models.length === 0 ? (
                  /* FIX #1: Show warning instead of redirecting */
                  <motion.div variants={fadeInUp}>
                    <Card className="border-2 border-orange-500" data-testid="no-model-warning">
                      <CardContent className="py-16 text-center">
                        <AlertCircle className="h-16 w-16 text-orange-500 mx-auto mb-6" />
                        <h3 className="text-xl font-semibold mb-3" data-testid="no-model-warning-title">
                          No trained model available
                        </h3>
                        <p className="text-muted-foreground mb-6 max-w-md mx-auto" data-testid="no-model-warning-message">
                          Please train a model in the Analysis section before making predictions.
                        </p>
                        <Button onClick={() => setActiveView('train')} size="lg" data-testid="go-to-train-btn">
                          <Zap className="h-4 w-4 mr-2" />
                          Go to Train Models
                        </Button>
                      </CardContent>
                    </Card>
                  </motion.div>
                ) : (
                  <>
                    {/* FIX #5: Show active (latest) model info */}
                    <motion.div variants={fadeInUp}>
                      <Card data-testid="active-model-card">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Cpu className="h-5 w-5" />
                            Active Model
                          </CardTitle>
                          <CardDescription>Using the most recently trained model</CardDescription>
                        </CardHeader>
                        <CardContent>
                          {(() => {
                            const activeModel = models[models.length - 1];
                            return (
                              <div className="flex items-center gap-4 p-4 rounded-lg border bg-primary/5 border-primary/20">
                                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                                  <Brain className="h-6 w-6 text-primary" />
                                </div>
                                <div className="flex-1">
                                  <p className="font-semibold" data-testid="active-model-algorithm">{activeModel.algorithm}</p>
                                  <p className="text-sm text-muted-foreground">
                                    {activeModel.problemType} &middot; ID: {activeModel.modelId.substring(0, 8)}...
                                  </p>
                                </div>
                                <Badge variant="default">Active</Badge>
                              </div>
                            );
                          })()}
                        </CardContent>
                      </Card>
                    </motion.div>

                    <motion.div variants={fadeInUp}>
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5" />
                            Input Data
                          </CardTitle>
                          <CardDescription>Provide data in JSON format for prediction</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <textarea
                            value={predictionInput}
                            onChange={(e) => setPredictionInput(e.target.value)}
                            placeholder={'[{"feature1": "value1", "feature2": "value2"}]'}
                            rows={10}
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring font-mono"
                            data-testid="prediction-input"
                          />
                          <p className="text-xs text-muted-foreground">
                            Example: {`[{"age": 30, "income": 50000, "credit_score": 700}]`}
                          </p>
                          <Button onClick={handlePredict} className="w-full h-12" size="lg" data-testid="generate-predictions-btn">
                            <Sparkles className="h-4 w-4 mr-2" />
                            Generate Predictions
                          </Button>
                        </CardContent>
                      </Card>
                    </motion.div>

                    {predictionResult && (
                      <motion.div variants={fadeInUp} initial="initial" animate="animate">
                        <Card className="border-2 border-primary" data-testid="prediction-results">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-primary">
                              <Eye className="h-5 w-5" />
                              Prediction Results
                            </CardTitle>
                            <CardDescription>Generated using {predictionResult.algorithm} ({predictionResult.problemType})</CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              {predictionResult.predictions.map((pred, idx) => (
                                <div key={idx} className="bg-muted/50 rounded-lg p-4" data-testid={`prediction-result-${idx}`}>
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium text-muted-foreground">Prediction {idx + 1}</span>
                                    <span className="text-2xl font-bold text-primary">{typeof pred === 'number' ? pred.toFixed(4) : pred}</span>
                                  </div>
                                  {predictionResult.probabilities?.[idx] && (
                                    <div className="mt-2 text-xs text-muted-foreground">
                                      Probabilities: [{predictionResult.probabilities[idx].map(p => p.toFixed(4)).join(', ')}]
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    )}
                  </>
                )}
              </motion.div>
            )}

            {/* ==================== MODELS VIEW ==================== */}
            {activeView === 'models' && (
              <motion.div
                key="models"
                variants={fadeInUp}
                initial="initial"
                animate="animate"
                exit="exit"
                data-testid="models-view"
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Model Library
                        </CardTitle>
                        <CardDescription>View and manage your trained models</CardDescription>
                      </div>
                      <Badge variant="secondary" className="text-lg px-4 py-2" data-testid="models-count-badge">
                        {models.length} Models
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {models.length === 0 ? (
                      <div className="text-center py-12" data-testid="empty-models">
                        <Database className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
                        <h3 className="text-lg font-medium mb-2">No Models Yet</h3>
                        <p className="text-muted-foreground mb-6">Train your first model to get started</p>
                        <Button onClick={() => setActiveView('train')} size="lg" data-testid="train-first-model-from-library">
                          <Zap className="h-4 w-4 mr-2" />
                          Train Your First Model
                        </Button>
                      </div>
                    ) : (
                      <div className="rounded-md border">
                        <table className="w-full" data-testid="models-table">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-4 text-left text-sm font-medium">Model ID</th>
                              <th className="p-4 text-left text-sm font-medium">Algorithm</th>
                              <th className="p-4 text-left text-sm font-medium">Type</th>
                              <th className="p-4 text-left text-sm font-medium">Created</th>
                              <th className="p-4 text-left text-sm font-medium">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {models.map((model, idx) => (
                              <motion.tr
                                key={model.modelId}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className="border-b last:border-0 hover:bg-accent/50 transition-colors"
                                data-testid={`model-row-${idx}`}
                              >
                                <td className="p-4">
                                  <div className="flex items-center gap-2">
                                    <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                                      <Brain className="h-4 w-4 text-primary" />
                                    </div>
                                    <code className="text-xs font-mono">{model.modelId.substring(0, 12)}...</code>
                                  </div>
                                </td>
                                <td className="p-4"><Badge variant="outline">{model.algorithm}</Badge></td>
                                <td className="p-4"><span className="text-sm">{model.problemType}</span></td>
                                <td className="p-4">
                                  <span className="text-sm text-muted-foreground">
                                    {new Date(model.createdAt).toLocaleDateString()}
                                  </span>
                                </td>
                                <td className="p-4">
                                  <div className="flex gap-2">
                                    <Button variant="ghost" size="sm" onClick={() => { setActiveView('predict'); }} title="Use for predictions" data-testid={`use-model-${idx}`}>
                                      <Eye className="h-4 w-4" />
                                    </Button>
                                    <Button variant="ghost" size="sm" onClick={() => handleDownloadModel(model.modelId)} title="Download model (JSON)" className="text-primary hover:text-primary" data-testid={`download-model-${idx}`}>
                                      <Download className="h-4 w-4" />
                                    </Button>
                                    <Button variant="ghost" size="sm" onClick={() => handleDeleteModel(model.modelId)} title="Delete model" data-testid={`delete-model-${idx}`}>
                                      <Trash2 className="h-4 w-4 text-destructive" />
                                    </Button>
                                  </div>
                                </td>
                              </motion.tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
