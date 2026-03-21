import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain, Sparkles, TrendingUp, Activity, Database, Zap, Upload, Play,
  Eye, Trash2, ChevronRight, ArrowUpRight, FileText, Target, Cpu, BarChart3,
  Download, AlertCircle, Layers, ShieldAlert, Table2, Info, SplitSquareVertical,
  Clock, Trophy, CheckCircle2, XCircle, Shield, Moon, Sun, FileUp, BarChart2,
  HelpCircle, BookOpen, Lightbulb, Save, History, Share2, Copy, ExternalLink, Lock, Sheet
} from 'lucide-react';
import {
  computeGlobalSHAP, computeBeeswarmData, computeLocalSHAP,
  computeDependenceData, computeLIME, computeClassProbabilities,
  buildWaterfallData, buildForceData, importanceColor, valueToColor
} from './explainableAI';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell, ReferenceLine, PieChart, Pie
} from 'recharts';
import { kmeans } from 'ml-kmeans';
import { runUnsupervisedPipeline, predictCluster } from './unsupervisedML';
import './App.css';

// ==================== CONSTANTS ====================
const CLUSTER_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#e11d48', '#4f46e5', '#059669', '#d97706'];
const fadeInUp = { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -20 } };
const staggerContainer = { animate: { transition: { staggerChildren: 0.1 } } };
const ALGO_NAMES = { linear_regression: 'Linear Regression', ridge_regression: 'Ridge Regression', logistic_regression: 'Logistic Regression', decision_tree: 'Decision Tree', random_forest: 'Random Forest', random_forest_regressor: 'Random Forest', random_forest_classifier: 'Random Forest', gradient_boosting: 'Gradient Boosting', knn: 'KNN', svm: 'SVM', naive_bayes: 'Naive Bayes', baseline: 'Baseline' };
const ALGO_COLORS = { linear_regression: '#2563eb', ridge_regression: '#7c3aed', logistic_regression: '#2563eb', decision_tree: '#16a34a', random_forest_regressor: '#059669', random_forest_classifier: '#059669', gradient_boosting: '#d97706', knn: '#dc2626', svm: '#9333ea', naive_bayes: '#0891b2', baseline: '#6b7280' };

const ALGO_DESCRIPTIONS = {
  auto: 'Trains all compatible algorithms and picks the one with the best score. Ideal when you are unsure which to use.',
  linear: 'Fits a straight line through your data. Best for simple numeric relationships between features and target.',
  ridge: 'Like Linear Regression but prevents overfitting by penalizing large coefficients. Use when you have many features.',
  logistic: 'Predicts categories (yes/no, A/B/C). Works well for binary classification with a clear decision boundary.',
  decision_tree: 'Splits data using yes/no questions. Easy to understand and visualize, but can overfit on small data.',
  random_forest: 'Combines many decision trees for more reliable predictions. Great all-rounder for both classification and regression.',
  gradient_boosting: 'Builds trees one at a time, each correcting the previous mistakes. Often the most accurate, but slower.',
  knn: 'Predicts based on the K most similar records. Simple and intuitive, works best with normalized features.',
  svm: 'Finds the best boundary between classes. Powerful for classification, especially with clear margins in data.',
  naive_bayes: 'Uses probability to classify. Very fast and works surprisingly well, especially with text or categorical data.',
};

const GUIDE_STEPS = [
  { step: 1, title: 'Upload Dataset', desc: 'Upload a CSV file or pick a sample dataset. Ensure your data has meaningful columns and enough rows (at least 10+).' },
  { step: 2, title: 'Review & Clean', desc: 'Check the Dataset Scanner for issues like missing values or outliers. Use the auto-clean tools to fix them.' },
  { step: 3, title: 'Select Target', desc: 'Pick the column you want to predict. For classification choose a categorical column; for regression pick a numeric one.' },
  { step: 4, title: 'Train Model', desc: 'Choose an algorithm (or let Auto pick the best). Click "Start Training" and wait for results.' },
  { step: 5, title: 'Explore Results', desc: 'Review accuracy metrics, the algorithm leaderboard, and confusion matrix or scatter plots.' },
  { step: 6, title: 'Make Predictions', desc: 'Go to the Predictions tab to enter new data and get predictions from your trained model.' },
  { step: 7, title: 'Explain with XAI', desc: 'Visit Explainability to understand why the model makes its predictions using SHAP and LIME.' },
];

const METRIC_EXPLANATIONS = {
  r2: { name: 'R² Score', description: 'Shows how well the model explains the data. Values closer to 1 mean better predictions.', higherBetter: true },
  mae: { name: 'Mean Absolute Error', description: 'Average prediction error. Lower values mean the model\'s predictions are closer to actual values.', higherBetter: false },
  rmse: { name: 'Root Mean Squared Error', description: 'Like MAE but penalizes larger errors more heavily. Lower is better.', higherBetter: false },
  accuracy: { name: 'Accuracy', description: 'Percentage of correct predictions. 100% means every prediction was right.', higherBetter: true },
  precision: { name: 'Precision', description: 'When the model predicts a class, how often is it correct? Higher means fewer false alarms.', higherBetter: true },
  recall: { name: 'Recall', description: 'Of all actual positives, how many did the model catch? Higher means fewer missed cases.', higherBetter: true },
  f1: { name: 'F1 Score', description: 'Balances precision and recall. Higher values mean fewer classification mistakes overall.', higherBetter: true },
  silhouette: { name: 'Silhouette Score', description: 'Measures how well data points fit within their clusters. Higher values mean clearer cluster separation.', higherBetter: true },
  daviesBouldin: { name: 'Davies-Bouldin Index', description: 'Measures how similar clusters are to each other. Lower values mean better-separated, more distinct clusters.', higherBetter: false },
  calinskiHarabasz: { name: 'Calinski-Harabasz Score', description: 'Ratio of between-cluster to within-cluster spread. Higher means denser, better-separated clusters.', higherBetter: true },
  inertia: { name: 'Inertia', description: 'Sum of distances from each point to its cluster center. Lower values mean tighter clusters, but too low may mean too many clusters.', higherBetter: false },
  cvScore: { name: 'Cross-Validation Score', description: 'Average model performance across multiple data splits. More reliable than a single test score.', higherBetter: true },
  shapValue: { name: 'SHAP Value', description: 'Shows how much a feature pushes the prediction up (positive) or down (negative) for a specific record.', higherBetter: null },
  baseValue: { name: 'Base Value', description: 'The average model prediction across the dataset. SHAP values show how each feature moves the prediction away from this baseline.', higherBetter: null },
};

// Lightweight hover tooltip for metric names — wraps any inline text
const MetricTip = ({ metricKey, children, className = '', value }) => {
  const info = METRIC_EXPLANATIONS[metricKey];
  if (!info) return <span className={className}>{children}</span>;
  const interp = value !== undefined ? interpretMetric(metricKey, value) : null;
  return (
    <span className={`group/tip relative inline-flex items-center gap-1 cursor-help ${className}`} data-testid={`metric-tip-${metricKey}`}>
      {children}
      <Info className="h-3 w-3 text-muted-foreground/50 group-hover/tip:text-foreground transition-colors shrink-0" />
      <span className="invisible group-hover/tip:visible opacity-0 group-hover/tip:opacity-100 absolute z-[60] bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-2.5 rounded-lg bg-popover border shadow-lg text-xs text-popover-foreground transition-all duration-150 pointer-events-none">
        <span className="font-semibold block mb-0.5">{info.name}</span>
        <span className="text-muted-foreground leading-relaxed block">{info.description}</span>
        {interp && <span className={`block mt-1.5 pt-1.5 border-t leading-relaxed font-medium ${interp.color}`}>{interp.text}</span>}
      </span>
    </span>
  );
};

// Generic help tooltip — shows a (?) icon with a hover explanation
const HelpTip = ({ text, children, className = '' }) => (
  <span className={`group/help relative inline-flex items-center gap-1.5 ${className}`}>
    {children}
    <span className="cursor-help" data-testid="help-tip-icon">
      <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/40 group-hover/help:text-blue-500 transition-colors shrink-0" />
    </span>
    <span className="invisible group-hover/help:visible opacity-0 group-hover/help:opacity-100 absolute z-[60] bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 rounded-lg bg-popover border shadow-lg text-xs text-popover-foreground transition-all duration-150 pointer-events-none leading-relaxed">
      {text}
    </span>
  </span>
);

function getScoreColor(score, higherBetter = true) {
  const s = higherBetter ? score : Math.max(0, 1 - score);
  if (s >= 0.9) return { bg: 'bg-emerald-50 dark:bg-emerald-950/30', border: 'border-emerald-300 dark:border-emerald-800', text: 'text-emerald-700 dark:text-emerald-400', label: 'Excellent' };
  if (s >= 0.7) return { bg: 'bg-sky-50 dark:bg-sky-950/30', border: 'border-sky-300 dark:border-sky-800', text: 'text-sky-700 dark:text-sky-400', label: 'Good' };
  if (s >= 0.5) return { bg: 'bg-amber-50 dark:bg-amber-950/30', border: 'border-amber-300 dark:border-amber-800', text: 'text-amber-700 dark:text-amber-400', label: 'Fair' };
  return { bg: 'bg-red-50 dark:bg-red-950/30', border: 'border-red-300 dark:border-red-800', text: 'text-red-700 dark:text-red-400', label: 'Needs Work' };
}

function interpretMetric(key, value) {
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

function arrayMin(arr) { let m = Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]; return m; }
function arrayMax(arr) { let m = -Infinity; for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]; return m; }
function arrayMinMax(arr) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < arr.length; i++) { const v = arr[i]; if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }

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
      headers.forEach((h, idx) => { const num = Number(values[idx]); row[h] = isNaN(num) || values[idx] === '' ? values[idx] : num; });
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
const DOMAIN_KEYWORDS = {
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

function generateDatasetSummary(profile) {
  if (!profile || !profile.columns || profile.columns.length === 0) return null;
  const colNamesLower = profile.columns.map(c => c.name.toLowerCase());
  const allNamesJoined = colNamesLower.join(' ');

  // Detect domain
  let bestDomain = 'general';
  let bestScore = 0;
  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (allNamesJoined.includes(kw)) score++;
    }
    if (score > bestScore) { bestScore = score; bestDomain = domain; }
  }
  const domainLabels = { finance: 'Financial / Credit Analysis', health: 'Healthcare / Medical', sales: 'Sales / Marketing', education: 'Education / Academic', realestate: 'Real Estate / Housing', hr: 'Human Resources / Employee', insurance: 'Insurance', ecommerce: 'E-Commerce', transportation: 'Transportation / Travel', environment: 'Environmental / Climate', general: 'General Data Analysis' };
  const domainLabel = domainLabels[bestDomain] || 'General Data Analysis';

  // Identify key variables (top 5 most interesting columns)
  const scored = profile.columns.map(c => {
    let s = 0;
    if (c.type === 'numeric' && c.std > 0) s += 3;
    if (c.uniqueCount >= 2 && c.uniqueCount <= 10 && c.type === 'categorical') s += 4;
    if (c.uniqueCount === 2) s += 2;
    if (c.missingCount === 0) s += 1;
    return { ...c, score: s };
  }).sort((a, b) => b.score - a.score);
  const keyVars = scored.slice(0, Math.min(5, scored.length));

  // Detect possible target
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

  // Build description
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

function suggestTask(profile, targetColumn) {
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

// ==================== FEATURE PREPARATION ====================

function detectProblemType(values) {
  if (values.some(v => typeof v !== 'number')) return 'classification';
  const uv = [...new Set(values)];
  if (uv.length === 2) return 'classification';
  if (uv.length / values.length < 0.05) return 'classification';
  return 'regression';
}

function prepareFeatures(rows, targetCol) {
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

function trainTestSplit(X, y, testSize = 0.2) {
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
  for (let i = 0; i < pa; i++) XtX[i][i] += 0.001; // minimal regularization for numerical stability
  return { type: 'linear_regression', coefficients: solveLinearSystem(XtX, Xty) };
}

function trainRidgeRegression(X, y, lambda = 1.0) {
  const n = X.length, p = X[0].length, pa = p + 1;
  const Xa = X.map(row => [1, ...row]);
  const { XtX, Xty } = computeXtXAndXty(Xa, y, pa, n);
  for (let i = 1; i < pa; i++) XtX[i][i] += lambda; // L2 penalty (skip intercept)
  XtX[0][0] += 0.001;
  return { type: 'ridge_regression', coefficients: solveLinearSystem(XtX, Xty) };
}

// ==================== CLASSIFICATION MODELS ====================

function trainLogisticRegression(X, y) {
  return { type: 'logistic_regression', coefficients: trainLinearRegression(X, y).coefficients };
}

// ==================== DECISION TREE ====================

function buildDecisionTree(X, y, maxDepth, minSamples, isClassification) {
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
    // Shuffle for random feature selection
    for (let i = allFeats.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [allFeats[i], allFeats[j]] = [allFeats[j], allFeats[i]]; }
    const featSubset = allFeats.slice(0, maxFeatPerSplit);

    let bestGain = 0, bestF = -1, bestT = 0, bestL = null, bestR = null;

    // Reusable scratch for sorted index pairs to avoid per-feature allocation
    const sortBuf = new Array(n);
    for (const f of featSubset) {
      // Sort indices by feature value — single allocation reused across features
      for (let i = 0; i < n; i++) sortBuf[i] = i;
      sortBuf.sort((a, b) => X[indices[a]][f] - X[indices[b]][f]);
      const step = Math.max(1, Math.floor(n / 25));
      for (let t = step; t < n; t += step) {
        const lv = X[indices[sortBuf[t - 1]]][f], rv = X[indices[sortBuf[t]]][f];
        if (rv === lv) continue;
        const thresh = (lv + rv) / 2;
        // Build left/right from pre-sorted order — avoids scanning all indices per threshold
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

function predictTree(tree, x) {
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

function standardizeFeatures(X) {
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

function predictOne(modelObj, x) {
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
    // Maintain a small sorted buffer of k nearest instead of sorting all
    const topD = new Float64Array(k).fill(Infinity);
    const topL = new Array(k);
    for (let i = 0; i < nTr; i++) {
      const xi = xt[i];
      let d = 0;
      for (let j = 0; j < xs.length; j++) { const diff = xi[j] - xs[j]; d += diff * diff; }
      if (d < topD[k - 1]) {
        // Insert into sorted buffer (small k, so linear insert is fine)
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

function predictBatch(modelObj, X) { return X.map(x => predictOne(modelObj, x)); }

// ==================== METRICS ====================

function calcRegressionMetrics(actual, predicted) {
  const n = actual.length;
  const meanA = actual.reduce((a, b) => a + b, 0) / n;
  const ssTot = actual.reduce((s, v) => s + (v - meanA) ** 2, 0);
  const ssRes = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  return { r2: ssTot > 0 ? 1 - ssRes / ssTot : 0, mae: actual.reduce((s, v, i) => s + Math.abs(v - predicted[i]), 0) / n, rmse: Math.sqrt(ssRes / n) };
}

function calcClassificationMetrics(actual, predicted) {
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

function extractImportance(modelObj, featureNames) {
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

function prepareInputForPrediction(inputRows, modelData) {
  const { numericCols, categoricalCols, encodingMap } = modelData;
  return inputRows.map(row => {
    const f = [];
    numericCols.forEach(col => f.push(Number(row[col]) || 0));
    categoricalCols.forEach(col => encodingMap[col].slice(1).forEach(val => f.push(String(row[col] || '') === val ? 1 : 0)));
    return f;
  });
}

// ==================== K-FOLD CROSS VALIDATION ====================

function kFoldCrossValidation(X, y, k, trainFn, problemType) {
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

function buildModelForAlgo(algo, X_train, y_train, problemType) {
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

// ==================== DATASET SCANNER ====================

function scanDataset(csvText, targetCol) {
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

function rebuildCsv(headers, rows) {
  return [headers.join(','), ...rows.map(r => headers.map(h => { const v = String(r[h] ?? ''); return v.includes(',') ? `"${v}"` : v; }).join(','))].join('\n');
}

function cleanRemoveDuplicates(csvText) {
  const lines = csvText.trim().split('\n');
  const header = lines[0];
  const unique = new Set(); const kept = [header]; let removed = 0;
  for (let i = 1; i < lines.length; i++) { if (!unique.has(lines[i])) { unique.add(lines[i]); kept.push(lines[i]); } else removed++; }
  return { text: kept.join('\n'), removed };
}

function cleanFillMissing(csvText) {
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

function cleanRemoveOutliers(csvText) {
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

function cleanDropConstants(csvText) {
  const { rows, headers } = parseCSV(csvText);
  const dropped = []; const kept = headers.filter(h => { if (new Set(rows.map(r => r[h])).size <= 1) { dropped.push(h); return false; } return true; });
  return { text: rebuildCsv(kept, rows), dropped };
}

function cleanNormalize(csvText) {
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

function runKMeansClustering(rows, numericCols, k) {
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

function detectAnomaliesFunc(rows, numericCols, method, threshold) {
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

// ==================== APP COMPONENT ====================

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [csvText, setCsvText] = useState('');
  const [columns, setColumns] = useState([]);
  const [dataProfile, setDataProfile] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [evalMode, setEvalMode] = useState('split');
  const [cleaningLog, setCleaningLog] = useState([]);
  const [precleanScan, setPrecleanScan] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [models, setModels] = useState([]);
  const [predictionFormData, setPredictionFormData] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [numClusters, setNumClusters] = useState(3);
  const [clusterResult, setClusterResult] = useState(null);
  const [anomalyMethod, setAnomalyMethod] = useState('zscore');
  const [anomalyThreshold, setAnomalyThreshold] = useState(3);
  const [anomalyResult, setAnomalyResult] = useState(null);
  const [unsupervisedResult, setUnsupervisedResult] = useState(null);
  const [isRunningUnsupervised, setIsRunningUnsupervised] = useState(false);
  const [clusterPredFormData, setClusterPredFormData] = useState({});
  const [clusterPredResult, setClusterPredResult] = useState(null);
  const [predictTab, setPredictTab] = useState('predict');
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [selectedModelIdx, setSelectedModelIdx] = useState(-1);
  const [corrVarX, setCorrVarX] = useState('');
  const [corrVarY, setCorrVarY] = useState('');
  const [darkMode, setDarkMode] = useState(() => {
    try { return localStorage.getItem('automl_dark_mode') === 'true'; } catch { return false; }
  });
  const [batchCsvText, setBatchCsvText] = useState('');
  const [batchResults, setBatchResults] = useState(null);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [histogramCol, setHistogramCol] = useState('');
  const [xaiTab, setXaiTab] = useState('shap');
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiRow, setXaiRow] = useState(0);
  const [xaiDepFeature, setXaiDepFeature] = useState(0);
  const [shapGlobal, setShapGlobal] = useState(null);
  const [shapBeeswarm, setShapBeeswarm] = useState(null);
  const [shapLocal, setShapLocal] = useState(null);
  const [shapDependence, setShapDependence] = useState(null);
  const [limeResult, setLimeResult] = useState(null);
  const [limeProbs, setLimeProbs] = useState(null);
  const [clusterShap, setClusterShap] = useState(null);
  const [clusterBeeswarm, setClusterBeeswarm] = useState(null);
  const [shapSummary, setShapSummary] = useState(null);
  const [featureVsPred, setFeatureVsPred] = useState(null);
  const [clusterComparison, setClusterComparison] = useState(null);
  const xaiCacheRef = useRef({});
  const [showGuide, setShowGuide] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [viewOnlyMode, setViewOnlyMode] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [shareCopyStatus, setShareCopyStatus] = useState(''); // 'copied' | 'manual' | ''

  const safeCopyToClipboard = useCallback(async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px;opacity:0';
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
      } catch { return false; }
    }
  }, []);

  // ==================== DARK MODE TOGGLE ====================
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    try { localStorage.setItem('automl_dark_mode', darkMode); } catch {}
  }, [darkMode]);

  // ==================== SESSION PERSISTENCE ====================
  const SESSION_KEY = 'automl_session';
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const saveTimerRef = useRef(null);

  // Load full session on mount
  useEffect(() => {
    try {
      const savedModels = localStorage.getItem('automl_models');
      if (savedModels) {
        const parsed = JSON.parse(savedModels);
        if (Array.isArray(parsed) && parsed.length > 0) setModels(parsed);
      }
      const raw = localStorage.getItem(SESSION_KEY);
      if (raw) {
        const s = JSON.parse(raw);
        if (s.csvText) {
          setCsvText(s.csvText);
          const p = profileDataset(s.csvText);
          setDataProfile(p);
          setColumns(p?.headers || []);
        }
        if (s.targetColumn !== undefined) setTargetColumn(s.targetColumn);
        if (s.algorithm) setAlgorithm(s.algorithm);
        if (s.evalMode) setEvalMode(s.evalMode);
        if (s.cleaningLog) setCleaningLog(s.cleaningLog);
        if (s.precleanScan) setPrecleanScan(s.precleanScan);
        if (s.trainingResult) setTrainingResult(s.trainingResult);
        if (s.predictionResult) setPredictionResult(s.predictionResult);
        if (s.predictionHistory) setPredictionHistory(s.predictionHistory);
        if (s.selectedModelIdx !== undefined) setSelectedModelIdx(s.selectedModelIdx);
        if (s.unsupervisedResult) setUnsupervisedResult(s.unsupervisedResult);
        if (s.clusterResult) setClusterResult(s.clusterResult);
        if (s.anomalyResult) setAnomalyResult(s.anomalyResult);
        if (s.batchResults) setBatchResults(s.batchResults);
        if (s.shapGlobal) setShapGlobal(s.shapGlobal);
        if (s.shapBeeswarm) setShapBeeswarm(s.shapBeeswarm);
        if (s.shapLocal) setShapLocal(s.shapLocal);
        if (s.shapDependence) setShapDependence(s.shapDependence);
        if (s.limeResult) setLimeResult(s.limeResult);
        if (s.limeProbs) setLimeProbs(s.limeProbs);
        if (s.clusterShap) setClusterShap(s.clusterShap);
        if (s.clusterBeeswarm) setClusterBeeswarm(s.clusterBeeswarm);
        if (s.shapSummary) setShapSummary(s.shapSummary);
        if (s.featureVsPred) setFeatureVsPred(s.featureVsPred);
        if (s.clusterComparison) setClusterComparison(s.clusterComparison);
        if (s.activeView) setActiveView(s.activeView);
        if (s.predictTab) setPredictTab(s.predictTab);
        if (s.numClusters !== undefined) setNumClusters(s.numClusters);
        if (s.anomalyMethod) setAnomalyMethod(s.anomalyMethod);
        if (s.anomalyThreshold !== undefined) setAnomalyThreshold(s.anomalyThreshold);
        if (s.histogramCol) setHistogramCol(s.histogramCol);
        if (s.xaiTab) setXaiTab(s.xaiTab);
        if (s.xaiRow !== undefined) setXaiRow(s.xaiRow);
        if (s.xaiDepFeature !== undefined) setXaiDepFeature(s.xaiDepFeature);
        if (s.corrVarX) setCorrVarX(s.corrVarX);
        if (s.corrVarY) setCorrVarY(s.corrVarY);
      }
    } catch (e) { console.error('Failed to load session:', e); }
    setSessionLoaded(true);
  }, []);

  // Save models separately for backward compat
  useEffect(() => {
    if (!sessionLoaded) return;
    try {
      const s = models.map(m => ({ ...m, modelData: { ...m.modelData } }));
      localStorage.setItem('automl_models', JSON.stringify(s));
    } catch (e) { console.error('Failed to save models:', e); }
  }, [models, sessionLoaded]);

  // Debounced session save (500ms after last state change)
  useEffect(() => {
    if (!sessionLoaded) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      try {
        localStorage.setItem(SESSION_KEY, JSON.stringify({
          csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
          trainingResult, predictionResult, predictionHistory, selectedModelIdx,
          unsupervisedResult, clusterResult, anomalyResult, batchResults,
          shapGlobal, shapBeeswarm, shapLocal, shapDependence,
          limeResult, limeProbs, clusterShap, clusterBeeswarm,
          shapSummary, featureVsPred, clusterComparison,
          activeView, predictTab, numClusters, anomalyMethod, anomalyThreshold,
          histogramCol, xaiTab, xaiRow, xaiDepFeature, corrVarX, corrVarY,
        }));
      } catch (e) { console.error('Session save failed:', e); }
    }, 500);
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [sessionLoaded, csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
    trainingResult, predictionResult, predictionHistory, selectedModelIdx,
    unsupervisedResult, clusterResult, anomalyResult, batchResults,
    shapGlobal, shapBeeswarm, shapLocal, shapDependence,
    limeResult, limeProbs, clusterShap, clusterBeeswarm,
    shapSummary, featureVsPred, clusterComparison,
    activeView, predictTab, numClusters, anomalyMethod, anomalyThreshold,
    histogramCol, xaiTab, xaiRow, xaiDepFeature, corrVarX, corrVarY]);

  // Clear all session data
  const handleClearSession = useCallback(() => {
    try { localStorage.removeItem(SESSION_KEY); localStorage.removeItem('automl_models'); } catch {}
    setCsvText(''); setColumns([]); setDataProfile(null); setTargetColumn('');
    setAlgorithm('auto'); setEvalMode('split'); setCleaningLog([]); setPrecleanScan(null);
    setTrainingResult(null); setModels([]); setPredictionFormData({}); setPredictionResult(null);
    setNumClusters(3); setClusterResult(null); setAnomalyMethod('zscore'); setAnomalyThreshold(3);
    setAnomalyResult(null); setUnsupervisedResult(null); setClusterPredFormData({});
    setClusterPredResult(null); setPredictTab('predict'); setPredictionHistory([]);
    setSelectedModelIdx(-1); setCorrVarX(''); setCorrVarY('');
    setBatchCsvText(''); setBatchResults(null); setHistogramCol('');
    setXaiTab('shap'); setXaiRow(0); setXaiDepFeature(0);
    setShapGlobal(null); setShapBeeswarm(null); setShapLocal(null); setShapDependence(null);
    setLimeResult(null); setLimeProbs(null); setClusterShap(null); setClusterBeeswarm(null);
    setShapSummary(null); setFeatureVsPred(null); setClusterComparison(null);
    xaiCacheRef.current = {};
    setActiveView('dashboard'); setError('');
  }, []);

  // ==================== HISTORY & SHARING ====================
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/snapshots`);
      const data = await res.json();
      setHistoryList(data.snapshots || []);
    } catch { setHistoryList([]); }
    setHistoryLoading(false);
  }, []);

  const handleSaveAnalysis = useCallback(async (customName) => {
    if (!trainingResult && !unsupervisedResult) { setError('Nothing to save — train a model first.'); return; }
    const name = customName || `${targetColumn || 'Unsupervised'} — ${new Date().toLocaleString()}`;
    const bestModel = trainingResult?.leaderboard?.[0];
    try {
      const body = {
        name,
        dataset_name: dataProfile?.fileName || 'Uploaded CSV',
        target_column: targetColumn || null,
        problem_type: trainingResult?.problemType || 'clustering',
        row_count: dataProfile?.rowCount || 0,
        col_count: dataProfile?.columns?.length || 0,
        models_summary: (trainingResult?.leaderboard || []).map(m => ({ algorithm: m.algorithm, score: m.score })),
        key_metrics: bestModel?.metrics || {},
        state: {
          csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
          trainingResult, predictionResult, predictionHistory, selectedModelIdx,
          unsupervisedResult, clusterResult, anomalyResult, batchResults,
          shapGlobal, shapBeeswarm, shapLocal, shapDependence,
          limeResult, limeProbs, clusterShap, clusterBeeswarm,
          shapSummary, featureVsPred, clusterComparison, models,
        },
      };
      const res = await fetch(`${API_URL}/api/snapshots`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await res.json();
      if (data.snapshot_id) { setError(''); fetchHistory(); return data.snapshot_id; }
      else { setError('Save failed: ' + (data.detail || 'Unknown error')); }
    } catch (e) { setError('Save failed: ' + e.message); }
    return null;
  }, [csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
    trainingResult, predictionResult, predictionHistory, selectedModelIdx,
    unsupervisedResult, clusterResult, anomalyResult, batchResults,
    shapGlobal, shapBeeswarm, shapLocal, shapDependence,
    limeResult, limeProbs, clusterShap, clusterBeeswarm,
    shapSummary, featureVsPred, clusterComparison, models, dataProfile, fetchHistory]);

  const handleLoadSnapshot = useCallback(async (snapshotId) => {
    try {
      const res = await fetch(`${API_URL}/api/snapshots/${snapshotId}`);
      const data = await res.json();
      if (!data.snapshot?.state) { setError('Snapshot not found or corrupted.'); return; }
      const s = data.snapshot.state;
      if (s.csvText) { setCsvText(s.csvText); const p = profileDataset(s.csvText); setDataProfile(p); setColumns(p?.headers || []); }
      if (s.targetColumn !== undefined) setTargetColumn(s.targetColumn);
      if (s.algorithm) setAlgorithm(s.algorithm);
      if (s.evalMode) setEvalMode(s.evalMode);
      if (s.cleaningLog) setCleaningLog(s.cleaningLog);
      if (s.precleanScan) setPrecleanScan(s.precleanScan);
      if (s.trainingResult) setTrainingResult(s.trainingResult);
      if (s.predictionResult) setPredictionResult(s.predictionResult);
      if (s.predictionHistory) setPredictionHistory(s.predictionHistory);
      if (s.selectedModelIdx !== undefined) setSelectedModelIdx(s.selectedModelIdx);
      if (s.unsupervisedResult) setUnsupervisedResult(s.unsupervisedResult);
      if (s.clusterResult) setClusterResult(s.clusterResult);
      if (s.anomalyResult) setAnomalyResult(s.anomalyResult);
      if (s.batchResults) setBatchResults(s.batchResults);
      if (s.shapGlobal) setShapGlobal(s.shapGlobal);
      if (s.shapBeeswarm) setShapBeeswarm(s.shapBeeswarm);
      if (s.shapLocal) setShapLocal(s.shapLocal);
      if (s.shapDependence) setShapDependence(s.shapDependence);
      if (s.limeResult) setLimeResult(s.limeResult);
      if (s.limeProbs) setLimeProbs(s.limeProbs);
      if (s.clusterShap) setClusterShap(s.clusterShap);
      if (s.clusterBeeswarm) setClusterBeeswarm(s.clusterBeeswarm);
      if (s.shapSummary) setShapSummary(s.shapSummary);
      if (s.featureVsPred) setFeatureVsPred(s.featureVsPred);
      if (s.clusterComparison) setClusterComparison(s.clusterComparison);
      if (s.models) setModels(s.models);
      setActiveView('dashboard');
      setError('');
    } catch (e) { setError('Failed to load snapshot: ' + e.message); }
  }, []);

  const handleDeleteSnapshot = useCallback(async (snapshotId) => {
    try {
      await fetch(`${API_URL}/api/snapshots/${snapshotId}`, { method: 'DELETE' });
      fetchHistory();
    } catch (e) { setError('Delete failed: ' + e.message); }
  }, [fetchHistory]);

  const handleShareAnalysis = useCallback(async () => {
    try {
      let sid = await handleSaveAnalysis();
      if (sid) {
        const url = `${window.location.origin}${window.location.pathname}?snapshot=${sid}`;
        setShareUrl(url);
        const ok = await safeCopyToClipboard(url);
        setShareCopyStatus(ok ? 'copied' : 'manual');
      }
    } catch { setShareCopyStatus('manual'); }
  }, [handleSaveAnalysis, safeCopyToClipboard]);

  const buildFullCSV = useCallback(() => {
    const sections = [];
    // --- Metrics Summary ---
    if (trainingResult?.metrics) {
      sections.push('--- Metrics Summary ---');
      sections.push('Metric,Value');
      Object.entries(trainingResult.metrics).forEach(([k, v]) => {
        sections.push(`${k},${typeof v === 'number' ? v.toFixed(6) : v}`);
      });
    }
    // --- Algorithm Leaderboard ---
    const lb = trainingResult?.leaderboard || [];
    if (lb.length > 0) {
      sections.push('');
      sections.push('--- Algorithm Leaderboard ---');
      sections.push('Rank,Algorithm,Score,Duration(s)');
      lb.forEach((m, i) => {
        sections.push(`${i + 1},${m.algorithm},${m.score?.toFixed(6) || 'N/A'},${m.duration?.toFixed(3) || m.durationSec?.toFixed(3) || 'N/A'}`);
      });
    }
    // --- SHAP Feature Importance ---
    if (shapGlobal?.importance?.length > 0) {
      sections.push('');
      sections.push('--- SHAP Feature Importance ---');
      sections.push('Feature,Importance');
      shapGlobal.importance.forEach(f => {
        sections.push(`${f.feature},${f.importance.toFixed(6)}`);
      });
    }
    // --- LIME Explanations ---
    if (limeResult?.contributions?.length > 0) {
      sections.push('');
      sections.push('--- LIME Explanations ---');
      sections.push('Feature,Weight,Contribution');
      limeResult.contributions.forEach(c => {
        sections.push(`${c.feature},${c.weight.toFixed(6)},${c.contribution.toFixed(6)}`);
      });
      if (limeResult.intercept != null) sections.push(`Intercept,,${limeResult.intercept.toFixed(6)}`);
      if (limeResult.prediction != null) sections.push(`Prediction,,${limeResult.prediction.toFixed(6)}`);
    }
    // --- Predictions ---
    if (predictionHistory?.length > 0) {
      sections.push('');
      sections.push('--- Predictions ---');
      const keys = Object.keys(predictionHistory[0].input || {});
      sections.push([...keys, 'Prediction', 'Algorithm', 'Timestamp'].join(','));
      predictionHistory.forEach(p => {
        const vals = keys.map(k => p.input?.[k] ?? '');
        sections.push([...vals, p.prediction ?? '', p.algorithm ?? '', p.timestamp ?? ''].join(','));
      });
    }
    return sections.join('\n');
  }, [trainingResult, shapGlobal, limeResult, predictionHistory]);

  const triggerBackendDownload = useCallback(async (csvContent, filename) => {
    try {
      const res = await fetch(`${API_URL}/api/export/prepare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_content: csvContent, filename })
      });
      const data = await res.json();
      if (data.token) {
        const downloadUrl = `${API_URL}/api/export/download/${data.token}`;
        // Use hidden iframe to trigger download — works in sandboxed iframes
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        document.body.appendChild(iframe);
        iframe.src = downloadUrl;
        setTimeout(() => { try { document.body.removeChild(iframe); } catch {} }, 30000);
      } else { setError('Export failed — could not prepare download.'); }
    } catch (e) { setError('Export failed: ' + e.message); }
  }, []);

  const handleExportCSV = useCallback(() => {
    if (!trainingResult && !shapGlobal && !limeResult && !predictionHistory?.length) return;
    const csv = buildFullCSV();
    triggerBackendDownload(csv, `analysis_data_${Date.now()}.csv`);
  }, [trainingResult, shapGlobal, limeResult, predictionHistory, buildFullCSV, triggerBackendDownload]);

  const handleExportSheets = useCallback(() => {
    if (!trainingResult && !shapGlobal && !limeResult && !predictionHistory?.length) return;
    const csv = buildFullCSV();
    triggerBackendDownload(csv, `analysis_for_google_sheets_${Date.now()}.csv`);
  }, [trainingResult, shapGlobal, limeResult, predictionHistory, buildFullCSV, triggerBackendDownload]);

  // Load shared snapshot from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const snapshotId = params.get('snapshot');
    if (snapshotId) {
      setViewOnlyMode(true);
      handleLoadSnapshot(snapshotId);
    }
  }, [handleLoadSnapshot]);

  // Fetch history when switching to history view
  useEffect(() => { if (activeView === 'history') fetchHistory(); }, [activeView, fetchHistory]);

  // ==================== SAMPLE DATASETS ====================
  const sampleDatasets = [
    { name: 'Loan Approval', description: 'Binary classification (20 rows)', data: `age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1\n22,32000,600,5000,0\n55,130000,810,80000,1\n29,48000,660,12000,0\n40,90000,750,35000,1\n33,70000,710,22000,1\n27,44000,640,11000,0\n50,110000,795,65000,1\n36,78000,730,28000,1\n24,40000,630,9000,0\n44,98000,770,45000,1` },
    { name: 'House Prices', description: 'Regression (15 rows)', data: `size,bedrooms,age,location_score,price\n1200,2,5,7,250000\n1800,3,10,8,380000\n2500,4,3,9,520000\n1000,1,15,6,180000\n2200,3,7,8,450000\n1500,2,8,7,300000\n3000,5,2,9,620000\n1100,1,20,5,170000\n1900,3,5,8,400000\n2800,4,1,10,580000\n1600,2,12,7,310000\n2100,3,6,8,430000\n1350,2,9,6,270000\n2600,4,4,9,540000\n1750,3,11,7,350000` },
    { name: 'Insurance Costs', description: 'Financial regression (20 rows)', data: `age,sex,bmi,children,smoker,region,charges\n19,female,27.9,0,yes,southwest,16884.92\n18,male,33.77,1,no,southeast,1725.55\n28,male,33.0,3,no,southeast,4449.46\n33,male,22.705,0,no,northwest,21984.47\n32,male,28.88,0,no,northwest,3866.86\n31,female,25.74,0,no,southeast,3756.62\n46,female,33.44,1,no,southeast,8240.59\n37,female,27.74,3,no,northwest,7281.51\n37,male,29.83,2,no,northeast,6406.41\n60,female,25.84,0,no,northwest,28923.14\n25,male,26.22,0,no,northeast,2721.32\n62,female,26.29,0,yes,southeast,27808.73\n23,male,34.4,0,no,southwest,1826.84\n56,female,39.82,0,no,southeast,11090.72\n27,male,42.13,0,yes,southeast,39611.76\n19,male,24.6,1,no,southwest,1837.24\n52,female,30.78,1,no,northeast,10797.34\n23,female,23.845,0,no,northeast,2395.17\n56,male,40.3,0,no,southwest,10602.39\n30,male,35.3,0,yes,southwest,36837.47` },
    { name: 'TV Shows', description: 'Text features (5 rows)', data: `show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description\ns1,TV Show,Breaking Bad,Vince Gilligan,Bryan Cranston,United States,July 1 2020,2008,TV-MA,5 Seasons,Crime TV Shows,A high school chemistry teacher turned meth producer teams up with a former student\ns2,Movie,The Shawshank Redemption,Frank Darabont,Tim Robbins,United States,June 15 2019,1994,R,142 min,Dramas,Two imprisoned men bond over a number of years finding redemption through acts of common decency\ns3,TV Show,Stranger Things,The Duffer Brothers,Millie Bobby Brown,United States,July 15 2016,2016,TV-14,4 Seasons,Sci-Fi TV Shows,When a young boy disappears his mother and friends must confront terrifying supernatural forces\ns4,Movie,The Dark Knight,Christopher Nolan,Christian Bale,United States,January 1 2021,2008,PG-13,152 min,Action & Adventure,When the menace known as the Joker wreaks havoc on Gotham Batman must accept one of the greatest tests\ns5,TV Show,Game of Thrones,David Benioff,Emilia Clarke,United States,April 17 2019,2011,TV-MA,8 Seasons,Fantasy TV Shows,Nine noble families fight for control over the lands of Westeros while an ancient enemy returns` }
  ];

  // ==================== COMPUTED STATS ====================
  const datasetScan = useMemo(() => {
    if (!csvText || !csvText.trim()) return null;
    try { return scanDataset(csvText, targetColumn); } catch { return null; }
  }, [csvText, targetColumn]);

  const stats = useMemo(() => {
    const t = models.length;
    if (t === 0) return { totalModels: 0, avgScore: 0, bestAlgoByScore: '--', mostUsedAlgo: '--', highestScore: 0, lastTraining: null, totalTrainings: 0 };
    const scores = models.map(m => m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || 0));
    const avgScore = scores.reduce((a, b) => a + b, 0) / t;
    const highestScore = Math.max(...scores);
    const bestIdx = scores.indexOf(highestScore);
    const bestAlgoByScore = ALGO_NAMES[models[bestIdx].algorithm] || models[bestIdx].algorithm;
    const algoCounts = {};
    models.forEach(m => { const name = ALGO_NAMES[m.algorithm] || m.algorithm; algoCounts[name] = (algoCounts[name] || 0) + 1; });
    const mostUsedAlgo = Object.entries(algoCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '--';
    return { totalModels: t, avgScore, bestAlgoByScore, mostUsedAlgo, highestScore, lastTraining: models[t - 1]?.createdAt, totalTrainings: t };
  }, [models]);

  const topModels = useMemo(() => {
    return [...models].sort((a, b) => {
      const aS = a.problemType === 'classification' ? (a.metrics?.accuracy || 0) : (a.metrics?.r2 || 0);
      const bS = b.problemType === 'classification' ? (b.metrics?.accuracy || 0) : (b.metrics?.r2 || 0);
      return bS - aS;
    }).slice(0, 5);
  }, [models]);

  const taskSuggestion = useMemo(() => dataProfile ? suggestTask(dataProfile, targetColumn) : null, [dataProfile, targetColumn]);

  const datasetSummary = useMemo(() => dataProfile ? generateDatasetSummary(dataProfile) : null, [dataProfile]);

  // Smart target suggestion: suggest the best target column based on data characteristics
  const suggestedTarget = useMemo(() => {
    if (!dataProfile || targetColumn) return null;
    const cols = dataProfile.columns;
    // Prefer categorical columns with 2-10 unique values (classification) or numeric columns with high variability (regression)
    let bestCol = null, bestScore = -1;
    for (const c of cols) {
      let score = 0;
      if (c.type === 'categorical' && c.uniqueCount >= 2 && c.uniqueCount <= 10) {
        score = 10 + (10 - c.uniqueCount); // fewer classes = cleaner classification
      } else if (c.type === 'numeric' && c.uniqueCount > 10) {
        score = 5 + Math.min(5, c.std / (Math.abs(c.mean) || 1)); // higher variability = better regression target
      }
      if (score > bestScore) { bestScore = score; bestCol = c; }
    }
    if (!bestCol) return null;
    const task = bestCol.type === 'categorical' ? 'classification' : 'regression';
    return { name: bestCol.name, task, reason: bestCol.type === 'categorical' ? `Categorical with ${bestCol.uniqueCount} classes` : `Numeric with high variability (std: ${bestCol.std?.toFixed(2)})` };
  }, [dataProfile, targetColumn]);

  // Smart XAI row suggestions: suggest interesting rows for explanation
  const smartRowSuggestions = useMemo(() => {
    if (!models.length || !dataProfile) return [];
    const model = models[selectedModelIdx === -1 ? models.length - 1 : selectedModelIdx];
    if (!model) return [];
    try {
      const data = prepareInputForPrediction(dataProfile.rows, model.modelData);
      if (!data.length) return [];
      const preds = data.slice(0, Math.min(200, data.length)).map((x, i) => ({ idx: i, pred: predictOne(model.modelData, x) }));
      const suggestions = [];
      // Random sample
      suggestions.push({ idx: Math.floor(Math.random() * data.length), label: 'Random sample', desc: 'A randomly selected data point' });
      // Highest prediction
      const highest = preds.reduce((a, b) => b.pred > a.pred ? b : a, preds[0]);
      suggestions.push({ idx: highest.idx, label: 'Highest prediction', desc: `Prediction: ${Number(highest.pred).toFixed(3)}` });
      // Lowest prediction
      const lowest = preds.reduce((a, b) => b.pred < a.pred ? b : a, preds[0]);
      suggestions.push({ idx: lowest.idx, label: 'Lowest prediction', desc: `Prediction: ${Number(lowest.pred).toFixed(3)}` });
      // Middle prediction (representative)
      const sorted = [...preds].sort((a, b) => a.pred - b.pred);
      const mid = sorted[Math.floor(sorted.length / 2)];
      suggestions.push({ idx: mid.idx, label: 'Representative (median)', desc: `Prediction: ${Number(mid.pred).toFixed(3)}` });
      return suggestions;
    } catch { return []; }
  }, [models, selectedModelIdx, dataProfile]);

  // ==================== XAI MODEL HELPER (moved up to avoid hoisting issues) ====================
  const getXaiModel = useCallback(() => models[selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : models.length - 1], [models, selectedModelIdx]);

  // ==================== FEATURE INSIGHTS (derived from SHAP data) ====================
  const featureInsights = useMemo(() => {
    if (!shapGlobal || !shapSummary || !shapBeeswarm || !dataProfile) return null;
    const fn = getXaiModel()?.modelData?.featureNames || [];
    if (!fn.length) return null;
    const target = targetColumn || 'target';

    // Top features ranked
    const topN = shapGlobal.importance.slice(0, 10);
    const top2 = topN.slice(0, 2);
    const top5 = topN.slice(0, 5);

    // Direction of impact for each feature (from shapSummary)
    const directions = {};
    shapSummary.forEach(s => {
      const net = s.positive + s.negative;
      directions[s.feature] = {
        positive: s.positive, negative: s.negative, net,
        direction: Math.abs(s.positive) > Math.abs(s.negative) ? 'increase' : 'decrease',
        label: Math.abs(s.positive) > Math.abs(s.negative) ? 'Higher values increase ' + target : 'Higher values decrease ' + target,
      };
    });

    // Optimal ranges: for top features, find what feature value ranges produce positive vs negative SHAP
    const optimalRanges = {};
    top5.forEach(feat => {
      const fi = fn.indexOf(feat.feature);
      if (fi === -1) return;
      const pts = shapBeeswarm.points.filter(p => p.featureIdx === fi);
      if (!pts.length) return;
      const highVal = pts.filter(p => p.normalizedValue > 0.6);
      const lowVal = pts.filter(p => p.normalizedValue < 0.4);
      const highShap = highVal.length > 0 ? highVal.reduce((s, p) => s + p.shapValue, 0) / highVal.length : 0;
      const lowShap = lowVal.length > 0 ? lowVal.reduce((s, p) => s + p.shapValue, 0) / lowVal.length : 0;
      optimalRanges[feat.feature] = {
        highShap, lowShap,
        recommendation: highShap > lowShap ? 'higher' : 'lower',
        label: highShap > lowShap ? `Higher ${feat.feature} tends to increase ${target}` : `Lower ${feat.feature} tends to increase ${target}`,
      };
    });

    // Correlation between top 2 features
    let correlation = null;
    if (top2.length === 2 && dataProfile.rows) {
      const f1 = top2[0].feature, f2 = top2[1].feature;
      const v1 = dataProfile.rows.map(r => typeof r[f1] === 'number' ? r[f1] : NaN).filter(v => !isNaN(v));
      const v2 = dataProfile.rows.map(r => typeof r[f2] === 'number' ? r[f2] : NaN).filter(v => !isNaN(v));
      const n = Math.min(v1.length, v2.length);
      if (n > 2) {
        const m1 = v1.slice(0, n).reduce((a, b) => a + b, 0) / n;
        const m2 = v2.slice(0, n).reduce((a, b) => a + b, 0) / n;
        let num = 0, d1 = 0, d2 = 0;
        for (let i = 0; i < n; i++) { const a = v1[i] - m1, b = v2[i] - m2; num += a * b; d1 += a * a; d2 += b * b; }
        const r = d1 > 0 && d2 > 0 ? num / (Math.sqrt(d1) * Math.sqrt(d2)) : 0;
        const strength = Math.abs(r) > 0.7 ? 'strong' : Math.abs(r) > 0.3 ? 'moderate' : 'weak';
        const dir = r > 0 ? 'positive' : 'negative';
        correlation = { feature1: f1, feature2: f2, r, strength, dir, label: `${strength} ${dir} correlation (r = ${r.toFixed(3)})` };
      }
    }

    // Scatter data for top 2 features
    let pairScatter = null;
    if (top2.length === 2 && dataProfile.rows && targetColumn) {
      const f1 = top2[0].feature, f2 = top2[1].feature;
      pairScatter = dataProfile.rows.slice(0, 500).map(r => ({
        x: typeof r[f1] === 'number' ? r[f1] : null,
        y: typeof r[f2] === 'number' ? r[f2] : null,
        target: r[targetColumn],
      })).filter(d => d.x !== null && d.y !== null);
    }

    // Business insights text
    const actionItems = [];
    if (top2[0]) {
      const d = directions[top2[0].feature] || {};
      const o = optimalRanges[top2[0].feature] || {};
      actionItems.push(`Focus on "${top2[0].feature}" — it has the strongest influence on ${target}. ${o.recommendation === 'higher' ? 'Increasing' : 'Decreasing'} this feature tends to improve outcomes.`);
    }
    if (top2[1]) {
      const d = directions[top2[1].feature] || {};
      const o = optimalRanges[top2[1].feature] || {};
      actionItems.push(`Optimize "${top2[1].feature}" as the second most impactful factor. ${o.recommendation === 'higher' ? 'Higher' : 'Lower'} values of this feature are associated with better ${target} outcomes.`);
    }
    if (correlation && Math.abs(correlation.r) > 0.3) {
      actionItems.push(`"${correlation.feature1}" and "${correlation.feature2}" have a ${correlation.strength} ${correlation.dir} relationship (r=${correlation.r.toFixed(2)}). Changes in one may affect the other.`);
    }

    return { topN, top2, top5, directions, optimalRanges, correlation, pairScatter, actionItems, target };
  }, [shapGlobal, shapSummary, shapBeeswarm, dataProfile, targetColumn, getXaiModel]);

  // ==================== DATA HANDLERS ====================
  const handleCsvTextChange = useCallback((text, isCleanAction) => {
    setCsvText(text); setTrainingResult(null); setClusterResult(null); setAnomalyResult(null);
    if (!isCleanAction) { setCleaningLog([]); setPrecleanScan(null); }
    if (text.trim()) { const p = profileDataset(text); setDataProfile(p); setColumns(p?.headers || []); }
    else { setDataProfile(null); setColumns([]); }
  }, []);
  const handleFileUpload = (event) => { const file = event.target.files[0]; if (file) { const reader = new FileReader(); reader.onload = (e) => handleCsvTextChange(e.target.result); reader.readAsText(file); } };
  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover'); };
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); if (e.dataTransfer.files?.[0]) { const reader = new FileReader(); reader.onload = (ev) => handleCsvTextChange(ev.target.result); reader.readAsText(e.dataTransfer.files[0]); } };

  const handleClean = (action) => {
    if (!csvText) return;
    if (!precleanScan && datasetScan) setPrecleanScan({ ...datasetScan });
    let result;
    if (action === 'duplicates') { result = cleanRemoveDuplicates(csvText); if (result.removed > 0) { handleCsvTextChange(result.text, true); setCleaningLog(prev => [...prev, `Removed ${result.removed} duplicate rows`]); } }
    else if (action === 'missing') { result = cleanFillMissing(csvText); if (result.filled > 0) { handleCsvTextChange(result.text, true); setCleaningLog(prev => [...prev, `Filled ${result.filled} missing values using median/mode`]); } }
    else if (action === 'outliers') { result = cleanRemoveOutliers(csvText); if (result.removed > 0) { handleCsvTextChange(result.text, true); setCleaningLog(prev => [...prev, `Removed ${result.removed} outlier rows using IQR method`]); } }
    else if (action === 'constants') { result = cleanDropConstants(csvText); if (result.dropped.length > 0) { handleCsvTextChange(result.text, true); setCleaningLog(prev => [...prev, `Dropped ${result.dropped.length} constant columns: ${result.dropped.join(', ')}`]); } }
    else if (action === 'normalize') { result = cleanNormalize(csvText); if (result.count > 0) { handleCsvTextChange(result.text, true); setCleaningLog(prev => [...prev, `Normalized ${result.count} numeric features (min-max scaling)`]); } }
  };

  const dataPreview = useMemo(() => {
    if (!csvText || !csvText.trim() || cleaningLog.length === 0) return null;
    try { const { rows, headers } = parseCSV(csvText); return { headers, rows: rows.slice(0, 10) }; } catch { return null; }
  }, [csvText, cleaningLog]);

  // ==================== TRAINING (ROBUST SUPERVISED LEARNING) ====================
  const handleTrain = () => {
    setError(''); setTrainingResult(null);
    if (!csvText) { setError('Please provide CSV data'); return; }
    if (!targetColumn || targetColumn === '__none__') { setError('Please select a target column'); return; }
    setIsTraining(true);

    // setTimeout allows loading spinner to render before heavy computation
    setTimeout(() => {
      try {
        const startTime = performance.now();
        // Reuse already-parsed rows from dataProfile when available to avoid re-parsing
        const rows = dataProfile?.rows || parseCSV(csvText).rows;
        if (rows.length < 4) throw new Error('Need at least 4 rows for train-test splitting');
        const prepared = prepareFeatures(rows, targetColumn);
        const { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols } = prepared;
        if (featureNames.length === 0) throw new Error('No usable features after preprocessing');
        const problemType = detectProblemType(y);

        // 80/20 Train-Test Split
        const { X_train, X_test, y_train, y_test, trainSize, testSize } = trainTestSplit(X, y, 0.2);

        const leaderboard = [];
        const trainModels = []; // raw model objects for evaluation

        // Decide which algorithms to train
        const algosToTrain = [];
        if (algorithm === 'auto') {
          if (problemType === 'regression') algosToTrain.push('linear_regression', 'ridge_regression', 'decision_tree', 'random_forest', 'gradient_boosting');
          else algosToTrain.push('logistic_regression', 'decision_tree', 'random_forest', 'knn', 'svm', 'naive_bayes');
        } else if (algorithm === 'linear') algosToTrain.push('linear_regression');
        else if (algorithm === 'ridge') algosToTrain.push('ridge_regression');
        else if (algorithm === 'logistic') algosToTrain.push('logistic_regression');
        else if (algorithm === 'decision_tree') algosToTrain.push('decision_tree');
        else if (algorithm === 'random_forest') algosToTrain.push('random_forest');
        else if (algorithm === 'gradient_boosting') algosToTrain.push('gradient_boosting');
        else if (algorithm === 'knn') algosToTrain.push('knn');
        else if (algorithm === 'svm') algosToTrain.push('svm');
        else if (algorithm === 'naive_bayes') algosToTrain.push('naive_bayes');
        algosToTrain.push('baseline');

        for (const algo of algosToTrain) {
          const t0 = performance.now();
          let modelObj;
          try { modelObj = buildModelForAlgo(algo, X_train, y_train, problemType); } catch { continue; }

          const trainPreds = predictBatch(modelObj, X_train);
          const testPreds = predictBatch(modelObj, X_test);
          const trainMetrics = problemType === 'regression' ? calcRegressionMetrics(y_train, trainPreds) : calcClassificationMetrics(y_train, trainPreds);
          const testMetrics = problemType === 'regression' ? calcRegressionMetrics(y_test, testPreds) : calcClassificationMetrics(y_test, testPreds);
          const fi = extractImportance(modelObj, featureNames);
          const dur = (performance.now() - t0) / 1000;

          // Run K-Fold Cross Validation if selected (skip baseline)
          let cvResult = null;
          if (evalMode === 'cv' && algo !== 'baseline') {
            try {
              cvResult = kFoldCrossValidation(X, y, 5, (xt, yt) => buildModelForAlgo(algo, xt, yt, problemType), problemType);
            } catch { cvResult = null; }
          }

          trainModels.push({ algo, modelObj });
          leaderboard.push({
            modelId: generateId(), algorithm: algo, status: 'ok', testMetrics, trainMetrics,
            featureImportance: fi, durationSec: dur,
            cvScore: cvResult?.cvScore ?? null, foldScores: cvResult?.foldScores ?? null
          });
        }

        // Sort by CV score (when available) or primary test metric
        const primaryKey = problemType === 'classification' ? 'accuracy' : 'r2';
        if (evalMode === 'cv') {
          leaderboard.sort((a, b) => {
            const aScore = a.cvScore ?? -Infinity;
            const bScore = b.cvScore ?? -Infinity;
            return bScore - aScore;
          });
        } else {
          leaderboard.sort((a, b) => (b.testMetrics[primaryKey] || 0) - (a.testMetrics[primaryKey] || 0));
        }
        const best = leaderboard[0];
        const bestModelObj = trainModels.find(m => m.algo === best.algorithm)?.modelObj;

        // Build regression viz from test set
        let predictionsVsActual = null, residualStats = null;
        if (problemType === 'regression' && bestModelObj) {
          const testPreds = predictBatch(bestModelObj, X_test);
          predictionsVsActual = { actual: [...y_test], predicted: testPreds };
          const res = y_test.map((v, i) => v - testPreds[i]);
          const mr = res.reduce((a, b) => a + b, 0) / res.length;
          const sr = Math.sqrt(res.reduce((s, v) => s + (v - mr) ** 2, 0) / res.length);
          residualStats = { mean: mr, std: sr, mean_abs: res.reduce((s, v) => s + Math.abs(v), 0) / res.length, predictive_power: Math.abs(mr) < sr * 0.5 ? 'Good' : 'Low' };
        }

        // Store best model for predictions
        const modelData = { featureNames, numericCols, categoricalCols, encodingMap, targetEncoding, ...bestModelObj };
        // Limit KNN storage for localStorage
        if (modelData.type === 'knn' && modelData.X_train && modelData.X_train.length > 500) {
          const si = Array.from({ length: modelData.X_train.length }, (_, i) => i);
          for (let i = si.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [si[i], si[j]] = [si[j], si[i]]; }
          const kept = si.slice(0, 500);
          modelData.X_train = kept.map(i => modelData.X_train[i]);
          modelData.y_train = kept.map(i => modelData.y_train[i]);
        }

        setModels(prev => [...prev, {
          modelId: best.modelId, algorithm: best.algorithm, problemType,
          metrics: best.testMetrics, trainMetrics: best.trainMetrics,
          featureImportance: best.featureImportance,
          createdAt: new Date().toISOString(), durationSec: best.durationSec,
          evalMode, targetColumn, modelData
        }]);

        setTrainingResult({
          status: 'success', problemType, bestModel: best, leaderboard, evalMode,
          totalTime: (performance.now() - startTime) / 1000,
          splitInfo: { trainSize, testSize, totalSize: rows.length },
          dataInfo: { numSamples: rows.length, numFeatures: featureNames.length, targetColumn, columns: featureNames, removedLeakageColumns: leakageCols, textColumns: textCols },
          predictionsVsActual, residualStats
        });
      } catch (err) { setError(err.message || 'Training failed'); }
      finally { setIsTraining(false); }
    }, 50);
  };

  // ==================== PREDICTION ====================
  const handlePredict = () => {
    setError(''); setPredictionResult(null);
    const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : models.length - 1;
    const am = models[idx];
    if (!am) { setError('No trained model available.'); return; }
    try {
      const row = {};
      am.modelData.numericCols.forEach(col => { row[col] = Number(predictionFormData[col]) || 0; });
      am.modelData.categoricalCols.forEach(col => { row[col] = predictionFormData[col] || ''; });
      const fvs = prepareInputForPrediction([row], am.modelData);
      const predictions = fvs.map(x => {
        const raw = predictOne(am.modelData, x);
        return am.modelData.targetEncoding ? am.modelData.targetEncoding[raw] : raw;
      });
      const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
      const probabilities = am.modelData.type === 'logistic_regression' ? fvs.map(x => { const z = am.modelData.coefficients[0] + x.reduce((s, v, i) => s + v * am.modelData.coefficients[i + 1], 0); const p = sigmoid(z); return [1 - p, p]; }) : null;
      setPredictionResult({ status: 'success', modelId: am.modelId, algorithm: am.algorithm, predictions, probabilities, problemType: am.problemType, inputData: row });
      setPredictionHistory(prev => [...prev, { id: Date.now(), type: 'supervised', model: ALGO_NAMES[am.algorithm] || am.algorithm, target: am.targetColumn, prediction: predictions[0], input: { ...row }, timestamp: Date.now() }]);
    } catch (err) { setError('Prediction failed: ' + err.message); }
  };

  // ==================== CLUSTERING & ANOMALY ====================
  const handleClustering = () => { setError(''); setClusterResult(null); if (!dataProfile) { setError('Please upload data first'); return; } if (dataProfile.numericColumns.length < 1) { setError('Need numeric columns'); return; } try { setClusterResult(runKMeansClustering(dataProfile.rows, dataProfile.numericColumns, numClusters)); } catch (err) { setError('Clustering failed: ' + err.message); } };
  const handleAnomalyDetection = () => { setError(''); setAnomalyResult(null); if (!dataProfile) { setError('Please upload data first'); return; } if (dataProfile.numericColumns.length < 1) { setError('Need numeric columns'); return; } try { setAnomalyResult(detectAnomaliesFunc(dataProfile.rows, dataProfile.numericColumns, anomalyMethod, anomalyThreshold)); } catch (err) { setError('Anomaly detection failed: ' + err.message); } };

  // ==================== BATCH PREDICTIONS ====================
  const handleBatchPredict = () => {
    setError(''); setBatchResults(null);
    const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : models.length - 1;
    const am = models[idx];
    if (!am) { setError('No trained model available.'); return; }
    if (!batchCsvText.trim()) { setError('Please upload or paste CSV data for batch prediction.'); return; }
    setBatchProcessing(true);
    setTimeout(() => {
      try {
        const { rows, headers } = parseCSV(batchCsvText);
        if (rows.length === 0) throw new Error('No rows found in CSV');
        const predictions = rows.map(row => {
          const inputRow = {};
          am.modelData.numericCols.forEach(col => { inputRow[col] = Number(row[col]) || 0; });
          am.modelData.categoricalCols.forEach(col => { inputRow[col] = row[col] || ''; });
          const fvs = prepareInputForPrediction([inputRow], am.modelData);
          const raw = predictOne(am.modelData, fvs[0]);
          return am.modelData.targetEncoding ? am.modelData.targetEncoding[raw] : raw;
        });
        setBatchResults({ headers, rows, predictions, algorithm: am.algorithm, targetColumn: am.targetColumn, problemType: am.problemType });
      } catch (err) { setError('Batch prediction failed: ' + err.message); }
      finally { setBatchProcessing(false); }
    }, 50);
  };

  const handleBatchCsvUpload = (event) => {
    const file = event.target.files[0];
    if (file) { const reader = new FileReader(); reader.onload = (e) => setBatchCsvText(e.target.result); reader.readAsText(file); }
  };

  const downloadBatchCsv = () => {
    if (!batchResults) return;
    const lines = [
      [...batchResults.headers, `predicted_${batchResults.targetColumn}`].join(','),
      ...batchResults.rows.map((row, i) => [...batchResults.headers.map(h => String(row[h] ?? '')), String(batchResults.predictions[i])].join(','))
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `batch_predictions_${Date.now()}.csv`;
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  };

  // ==================== MODEL IMPORT ====================
  const handleImportModel = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const model = JSON.parse(e.target.result);
        if (!model.modelId || !model.algorithm || !model.modelData) throw new Error('Invalid model file format');
        model.modelId = generateId(); // assign new ID to avoid conflicts
        model.importedAt = new Date().toISOString();
        setModels(prev => [...prev, model]);
      } catch (err) { setError('Failed to import model: ' + err.message); }
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  // ==================== XAI — SHAP & LIME ====================
  const getXaiData = (model) => {
    if (!model || !dataProfile) return null;
    let data = prepareInputForPrediction(dataProfile.rows, model.modelData);
    if (data.length > 2000) {
      // Stratified sampling: pick evenly spaced indices + random subset for representativeness
      const cap = 800;
      const step = Math.floor(data.length / cap);
      const sampled = [];
      for (let i = 0; i < data.length && sampled.length < cap; i += step) sampled.push(data[i]);
      data = sampled;
    }
    return data;
  };

  const handleRunSHAP = () => {
    setXaiLoading(true); setError('');
    setTimeout(() => {
      try {
        const model = getXaiModel();
        const data = getXaiData(model);
        if (!model || !data) { setError('Need a trained model and data.'); setXaiLoading(false); return; }
        const predictFn = (fv) => predictOne(model.modelData, fv);
        const fn = model.modelData.featureNames;
        const global = computeGlobalSHAP(predictFn, data, fn, Math.min(60, data.length), Math.min(60, data.length));
        const beeswarm = computeBeeswarmData(predictFn, data, fn, Math.min(80, data.length), Math.min(50, data.length));
        const rowIdx = Math.min(xaiRow, data.length - 1);
        const local = computeLocalSHAP(predictFn, data[rowIdx], data, Math.min(80, data.length));
        const dep = computeDependenceData(predictFn, data, fn, Math.min(xaiDepFeature, fn.length - 1), Math.min(120, data.length), Math.min(50, data.length));
        setShapGlobal(global);
        setShapBeeswarm(beeswarm);
        setShapLocal({ ...local, featureNames: fn, instance: data[rowIdx] });
        setShapDependence(dep);
        // Compute SHAP Summary (positive/negative mean per feature)
        const summaryMap = {};
        fn.forEach((f, fi) => { summaryMap[fi] = { feature: f, posSum: 0, posCount: 0, negSum: 0, negCount: 0 }; });
        beeswarm.points.forEach(p => {
          const entry = summaryMap[p.featureIdx];
          if (entry) {
            if (p.shapValue >= 0) { entry.posSum += p.shapValue; entry.posCount++; }
            else { entry.negSum += p.shapValue; entry.negCount++; }
          }
        });
        setShapSummary(fn.map((f, fi) => {
          const e = summaryMap[fi];
          return { feature: f, positive: e.posCount > 0 ? e.posSum / e.posCount : 0, negative: e.negCount > 0 ? e.negSum / e.negCount : 0 };
        }).sort((a, b) => (Math.abs(b.positive) + Math.abs(b.negative)) - (Math.abs(a.positive) + Math.abs(a.negative))));
        // Compute Feature vs Prediction scatter
        const sampleSize = Math.min(200, data.length);
        const fvpData = [];
        for (let i = 0; i < sampleSize; i++) {
          const pred = predictFn(data[i]);
          fn.forEach((f, fi) => { fvpData.push({ feature: f, featureIdx: fi, value: data[i][fi], prediction: pred }); });
        }
        setFeatureVsPred(fvpData);
      } catch (err) { setError('SHAP analysis failed: ' + err.message); }
      setXaiLoading(false);
    }, 50);
  };

  const handleRunLIME = () => {
    setXaiLoading(true); setError('');
    setTimeout(() => {
      try {
        const model = getXaiModel();
        const data = getXaiData(model);
        if (!model || !data) { setError('Need a trained model and data.'); setXaiLoading(false); return; }
        const predictFn = (fv) => predictOne(model.modelData, fv);
        const fn = model.modelData.featureNames;
        const rowIdx = Math.min(xaiRow, data.length - 1);
        const lime = computeLIME(predictFn, data[rowIdx], data, fn);
        setLimeResult(lime);
        if (model.problemType === 'classification') {
          setLimeProbs(computeClassProbabilities(predictFn, data[rowIdx], data));
        } else { setLimeProbs(null); }
      } catch (err) { setError('LIME explanation failed: ' + err.message); }
      setXaiLoading(false);
    }, 50);
  };

  const handleExplainPrediction = () => {
    setXaiLoading(true); setError('');
    setTimeout(() => {
      try {
        const model = getXaiModel();
        const data = getXaiData(model);
        if (!model || !data) { setError('Need a trained model and data.'); setXaiLoading(false); return; }
        const predictFn = (fv) => predictOne(model.modelData, fv);
        const fn = model.modelData.featureNames;
        const rowIdx = Math.min(xaiRow, data.length - 1);
        const inst = data[rowIdx];
        // SHAP
        const global = computeGlobalSHAP(predictFn, data, fn, Math.min(60, data.length), Math.min(60, data.length));
        const beeswarm = computeBeeswarmData(predictFn, data, fn, Math.min(80, data.length), Math.min(50, data.length));
        const local = computeLocalSHAP(predictFn, inst, data, Math.min(80, data.length));
        const dep = computeDependenceData(predictFn, data, fn, Math.min(xaiDepFeature, fn.length - 1), Math.min(120, data.length), Math.min(50, data.length));
        setShapGlobal(global); setShapBeeswarm(beeswarm);
        setShapLocal({ ...local, featureNames: fn, instance: inst });
        setShapDependence(dep);
        // Summary
        const summaryMap = {};
        fn.forEach((f, fi) => { summaryMap[fi] = { feature: f, posSum: 0, posCount: 0, negSum: 0, negCount: 0 }; });
        beeswarm.points.forEach(p => {
          const entry = summaryMap[p.featureIdx];
          if (entry) {
            if (p.shapValue >= 0) { entry.posSum += p.shapValue; entry.posCount++; }
            else { entry.negSum += p.shapValue; entry.negCount++; }
          }
        });
        setShapSummary(fn.map((f, fi) => {
          const e = summaryMap[fi];
          return { feature: f, positive: e.posCount > 0 ? e.posSum / e.posCount : 0, negative: e.negCount > 0 ? e.negSum / e.negCount : 0 };
        }).sort((a, b) => (Math.abs(b.positive) + Math.abs(b.negative)) - (Math.abs(a.positive) + Math.abs(a.negative))));
        // Feature vs Prediction
        const sampleSize = Math.min(200, data.length);
        const fvpData = [];
        for (let i = 0; i < sampleSize; i++) {
          const pred = predictFn(data[i]);
          fn.forEach((f, fi) => { fvpData.push({ feature: f, featureIdx: fi, value: data[i][fi], prediction: pred }); });
        }
        setFeatureVsPred(fvpData);
        // LIME
        const lime = computeLIME(predictFn, inst, data, fn);
        setLimeResult(lime);
        if (model.problemType === 'classification') { setLimeProbs(computeClassProbabilities(predictFn, inst, data)); } else { setLimeProbs(null); }
      } catch (err) { setError('Explanation failed: ' + err.message); }
      setXaiLoading(false);
    }, 50);
  };

  const handleClusterExplain = () => {
    setXaiLoading(true); setError('');
    setTimeout(() => {
      try {
        if (!unsupervisedResult || !dataProfile) { setError('No unsupervised results.'); setXaiLoading(false); return; }
        const fn = unsupervisedResult.preprocessing.featureNames;
        const { means, stds } = unsupervisedResult;
        const data = dataProfile.rows.map(row => fn.map((col, j) => {
          const v = typeof row[col] === 'number' ? row[col] : 0;
          return (v - means[j]) / (stds[j] || 1);
        }));
        const labels = unsupervisedResult.bestAlgorithm.labels;
        const k = unsupervisedResult.bestAlgorithm.k;
        const centroids = [];
        for (let c = 0; c < k; c++) {
          const pts = data.filter((_, i) => labels[i] === c);
          if (pts.length) centroids.push(pts[0].map((_, fi) => pts.reduce((s, p) => s + p[fi], 0) / pts.length));
        }
        const clusterPredict = (fv) => {
          let best = 0, minD = Infinity;
          centroids.forEach((c, i) => { let d = 0; for (let j = 0; j < fv.length; j++) d += (fv[j] - c[j]) ** 2; if (d < minD) { minD = d; best = i; } });
          return best;
        };
        const global = computeGlobalSHAP(clusterPredict, data, fn, Math.min(60, data.length), Math.min(60, data.length));
        const beeswarm = computeBeeswarmData(clusterPredict, data, fn, Math.min(80, data.length), Math.min(50, data.length));
        setClusterShap(global);
        setClusterBeeswarm(beeswarm);
        // Compute cluster comparison (per-feature raw means per cluster)
        const rawData = dataProfile.rows.map(row => fn.map(col => typeof row[col] === 'number' ? row[col] : 0));
        const comparison = fn.map((f, fi) => {
          const entry = { feature: f };
          for (let c = 0; c < k; c++) {
            const pts = rawData.filter((_, i) => labels[i] === c);
            entry[`cluster_${c}`] = pts.length > 0 ? pts.reduce((s, p) => s + p[fi], 0) / pts.length : 0;
          }
          const allVals = rawData.map(r => r[fi]);
          entry.overall = allVals.reduce((s, v) => s + v, 0) / allVals.length;
          return entry;
        });
        setClusterComparison({ data: comparison, k, featureNames: fn });
      } catch (err) { setError('Cluster explanation failed: ' + err.message); }
      setXaiLoading(false);
    }, 50);
  };

  // ==================== UNSUPERVISED LEARNING ====================
  const handleRunUnsupervised = () => {
    setError(''); setUnsupervisedResult(null); setClusterPredResult(null);
    if (!dataProfile) { setError('Please upload data first'); return; }
    if (dataProfile.numericColumns.length < 1) { setError('Need at least 1 numeric column'); return; }
    setIsRunningUnsupervised(true);
    setTimeout(() => {
      try {
        const result = runUnsupervisedPipeline(dataProfile.rows, dataProfile.numericColumns);
        setUnsupervisedResult(result);
        setActiveView('predict'); setPredictTab('results');
      } catch (err) { setError('Unsupervised analysis failed: ' + err.message); }
      finally { setIsRunningUnsupervised(false); }
    }, 50);
  };

  const handleClusterPredict = () => {
    if (!unsupervisedResult?.bestAlgorithm) return;
    const { means, stds, bestAlgorithm, interpretation } = unsupervisedResult;
    const featureNames = unsupervisedResult.preprocessing.featureNames;
    const standardized = featureNames.map((col, j) => {
      const v = Number(clusterPredFormData[col]) || 0;
      return (v - means[j]) / (stds[j] || 1);
    });
    const result = predictCluster(standardized, bestAlgorithm.centroids, interpretation?.interpretations);
    setClusterPredResult({ ...result, inputData: { ...clusterPredFormData } });
    setPredictionHistory(prev => [...prev, { id: Date.now(), type: 'cluster', model: unsupervisedResult.bestAlgorithm.name, prediction: `Cluster ${result.cluster}`, input: { ...clusterPredFormData }, timestamp: Date.now() }]);
  };

  const UNSUPERVISED_TERMS = [
    { key: 'kmeans', name: 'K-Means', desc: 'Groups data into K clusters by minimizing the distance between points and their assigned cluster center. Fast and works well for spherical clusters.' },
    { key: 'hierarchical', name: 'Hierarchical Clustering', desc: 'Builds a tree of clusters by progressively merging the closest pairs. No need to specify K in advance.' },
    { key: 'dbscan', name: 'DBSCAN', desc: 'Finds clusters based on density. Points in dense regions form clusters; sparse points are marked as noise. Discovers clusters of arbitrary shape.' },
    { key: 'gmm', name: 'Gaussian Mixture Model', desc: 'Models data as a mixture of Gaussian distributions. Soft clustering: each point has a probability of belonging to each cluster.' },
    { key: 'cluster', name: 'Cluster', desc: 'A group of similar data points. Points within a cluster are more similar to each other than to points in other clusters.' },
    { key: 'centroid', name: 'Centroid', desc: 'The center point of a cluster, calculated as the average of all points in that cluster. Represents the "typical" member.' },
    { key: 'densityClustering', name: 'Density-Based Clustering', desc: 'Clustering approach that groups together points in high-density regions separated by low-density areas. DBSCAN is the most common example.' },
    { key: 'silhouette', name: 'Silhouette Score', desc: 'Measures how similar a point is to its own cluster vs. other clusters. Ranges from -1 to 1. Higher = better-defined clusters.' },
    { key: 'daviesBouldin', name: 'Davies-Bouldin Index', desc: 'Measures average similarity between clusters. Lower values = better separation. 0 = perfect clustering.' },
    { key: 'calinskiHarabasz', name: 'Calinski-Harabasz Score', desc: 'Ratio of between-cluster to within-cluster variance. Higher values = denser, better-separated clusters.' },
    { key: 'pca', name: 'PCA (Principal Component Analysis)', desc: 'Reduces data dimensions by finding directions of maximum variance. Useful for visualizing high-dimensional data in 2D.' },
    { key: 'tsne', name: 't-SNE', desc: 'Non-linear dimensionality reduction that preserves local structure. Excellent for visualizing clusters in 2D but not for distance interpretation.' },
    { key: 'anomalyDetection', name: 'Anomaly Detection', desc: 'Identifies data points that differ significantly from the majority. These outliers may indicate errors, fraud, or unusual behavior.' },
    { key: 'isolationForest', name: 'Isolation Forest', desc: 'Detects anomalies by randomly partitioning data. Anomalies are easier to isolate and have shorter path lengths in random trees.' },
    { key: 'lof', name: 'Local Outlier Factor', desc: 'Measures how isolated a point is relative to its neighbors. Points with much lower density than their neighbors are flagged as outliers.' },
  ];

  const corrMatrix = useMemo(() => {
    if (!dataProfile) return [];
    const cols = dataProfile.numericColumns;
    const rows = dataProfile.rows;
    function corr(c1, c2) {
      const pairs = rows.filter(r => typeof r[c1] === 'number' && typeof r[c2] === 'number');
      if (pairs.length < 3) return 0;
      const n = pairs.length;
      const x = pairs.map(r => r[c1]), y = pairs.map(r => r[c2]);
      const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
      let sx = 0, sy = 0, sxy = 0;
      for (let i = 0; i < n; i++) { sx += (x[i] - mx) ** 2; sy += (y[i] - my) ** 2; sxy += (x[i] - mx) * (y[i] - my); }
      return (sx > 0 && sy > 0) ? sxy / Math.sqrt(sx * sy) : 0;
    }
    return cols.map(c1 => { const entry = { feature: c1 }; cols.forEach(c2 => { entry[c2] = corr(c1, c2); }); return entry; });
  }, [dataProfile]);

  const histogramData = useMemo(() => {
    if (!dataProfile || !histogramCol) return [];
    const vals = dataProfile.rows.map(r => r[histogramCol]).filter(v => typeof v === 'number');
    if (vals.length === 0) return [];
    const [min, max] = arrayMinMax(vals);
    const range = max - min;
    if (range === 0) return [{ bin: `${min.toFixed(1)}`, count: vals.length }];
    const nBins = Math.min(20, Math.max(5, Math.ceil(Math.sqrt(vals.length))));
    const binWidth = range / nBins;
    const bins = Array.from({ length: nBins }, (_, i) => ({
      bin: `${(min + i * binWidth).toFixed(1)}`,
      from: min + i * binWidth,
      to: min + (i + 1) * binWidth,
      count: 0,
    }));
    vals.forEach(v => {
      let idx = Math.floor((v - min) / binWidth);
      if (idx >= nBins) idx = nBins - 1;
      bins[idx].count++;
    });
    return bins;
  }, [dataProfile, histogramCol]);

  // ==================== MODEL MANAGEMENT ====================
  const handleDeleteModel = (modelId) => setModels(prev => prev.filter(m => m.modelId !== modelId));
  const handleDownloadModel = (modelId) => {
    const model = models.find(m => m.modelId === modelId); if (!model) return;
    const blob = new Blob([JSON.stringify({ ...model }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `${model.algorithm}_${modelId.substring(0, 8)}.json`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  };

  // ==================== SUB-COMPONENTS ====================
  const StatCard = ({ title, value, icon: Icon, metricValue }) => (
    <motion.div variants={fadeInUp}><Card className="hover:shadow-lg transition-shadow duration-300" data-testid={`stat-card-${title.toLowerCase().replace(/\s+/g, '-')}`}><CardContent className="p-6"><div className="flex items-center justify-between"><div className="space-y-2"><p className="text-sm font-medium text-muted-foreground">{title}</p><div className="flex items-baseline gap-2"><h3 className="text-3xl font-bold tracking-tight" data-testid={`stat-value-${title.toLowerCase().replace(/\s+/g, '-')}`}>{value}</h3>
      {metricValue !== undefined && <Badge variant={metricValue > 0 ? 'default' : 'secondary'} className="gap-1" data-testid={`stat-trend-${title.toLowerCase().replace(/\s+/g, '-')}`}>{metricValue > 0 ? <><ArrowUpRight className="h-3 w-3" />{`+${metricValue}`}</> : 'N/A'}</Badge>}
    </div></div><div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center"><Icon className="h-6 w-6 text-primary" /></div></div></CardContent></Card></motion.div>
  );

  const DataUploadMini = () => (
    <Card data-testid="data-upload-mini"><CardContent className="p-6"><div className="flex flex-col items-center justify-center py-8 text-center"><Upload className="h-12 w-12 text-muted-foreground/50 mb-4" /><h3 className="text-lg font-semibold mb-2">No Data Loaded</h3><p className="text-sm text-muted-foreground mb-4">Upload data in the Analysis tab or select a sample dataset</p>
      <div className="flex gap-2 flex-wrap justify-center">{sampleDatasets.slice(0, 3).map((ds, i) => <Button key={i} variant="outline" size="sm" onClick={() => handleCsvTextChange(ds.data)} data-testid={`mini-sample-${i}`}>{ds.name}</Button>)}</div>
    </div></CardContent></Card>
  );

  // Metric display helper with color coding and tooltips
  const MetricCard = ({ label, value, score, metricKey }) => {
    const explanation = METRIC_EXPLANATIONS[metricKey];
    const color = score !== undefined ? getScoreColor(score, explanation?.higherBetter !== false) : null;
    const interp = metricKey ? interpretMetric(metricKey, score !== undefined ? score : parseFloat(String(value))) : null;
    return (
      <div className={`rounded-xl p-4 border-2 transition-all ${color ? `${color.bg} ${color.border}` : 'bg-muted/50 border-transparent'}`} data-testid={`metric-${metricKey || label}`}>
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</p>
          {explanation && (
            <div className="group relative">
              <Info className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help hover:text-foreground transition-colors" />
              <div className="invisible group-hover:visible opacity-0 group-hover:opacity-100 absolute z-50 bottom-full right-0 mb-2 w-72 p-3 rounded-lg bg-popover border shadow-lg text-xs text-popover-foreground transition-all duration-200 pointer-events-none">
                <p className="font-semibold mb-1">{explanation.name}</p>
                <p className="text-muted-foreground leading-relaxed">{explanation.description}</p>
                {interp && <p className={`mt-2 pt-2 border-t leading-relaxed font-medium ${interp.color}`}>{interp.text}</p>}
              </div>
            </div>
          )}
        </div>
        <p className={`text-2xl font-bold ${color ? color.text : ''}`}>{value}</p>
        {color && <div className={`inline-flex items-center gap-1 mt-1 text-xs font-semibold ${color.text}`}>{color.label === 'Excellent' || color.label === 'Good' ? <CheckCircle2 className="h-3 w-3" /> : <AlertCircle className="h-3 w-3" />}{color.label}</div>}
        {interp && <p className={`text-[11px] mt-2 leading-relaxed ${interp.color}`} data-testid={`metric-interp-${metricKey}`}>{interp.text}</p>}
      </div>
    );
  };

  // ==================== RENDER ====================
  return (
    <div className="min-h-screen bg-background" data-testid="app-root">
      <motion.aside initial={{ x: -300 }} animate={{ x: 0 }} className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-sidebar" data-testid="app-sidebar">
        <div className="flex h-full flex-col gap-2">
          <div className="flex h-16 items-center border-b border-sidebar-border px-6"><div className="flex items-center gap-2"><div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground"><Brain className="h-6 w-6" /></div><div><h1 className="text-lg font-bold text-sidebar-foreground">AutoML</h1><p className="text-xs text-sidebar-foreground/60">Universal Dashboard</p></div></div></div>
          <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
            {[{ id: 'dashboard', label: 'Dashboard', icon: Activity }, { id: 'analysis', label: 'Analysis', icon: Zap }, { id: 'predict', label: 'Predictions', icon: Sparkles }, { id: 'explainability', label: 'Explainability', icon: Eye }, { id: 'explore', label: 'Data Explorer', icon: BarChart2 }, ...(targetColumn && targetColumn !== '__none__' ? [{ id: 'anomalies', label: 'Anomalies', icon: ShieldAlert }] : []), { id: 'models', label: 'Model Library', icon: Database }, { id: 'history', label: 'History', icon: History }].map((item) => (
              <Button key={item.id} variant={activeView === item.id ? 'secondary' : 'ghost'} className="w-full justify-start gap-3" onClick={() => setActiveView(item.id)} data-testid={`nav-${item.id}`}><item.icon className="h-4 w-4" />{item.label}</Button>
            ))}
          </nav>
          <div className="border-t border-sidebar-border p-4"><Card className="bg-sidebar-accent"><CardContent className="p-4"><p className="text-xs font-medium text-sidebar-foreground">Client-Side ML</p><p className="text-xs text-sidebar-foreground/70">All analysis runs in your browser</p></CardContent></Card></div>
        </div>
      </motion.aside>

      <div className="pl-64">
        <motion.header initial={{ y: -100 }} animate={{ y: 0 }} className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex h-16 items-center justify-between px-8"><div>
            <h2 className="text-2xl font-bold tracking-tight" data-testid="page-title">
              {activeView === 'dashboard' && 'Dashboard'}{activeView === 'analysis' && 'Universal Analysis'}{activeView === 'predict' && 'Predictions & Analysis'}{activeView === 'anomalies' && 'Anomaly Detection'}{activeView === 'models' && 'Model Library'}{activeView === 'explore' && 'Data Explorer'}{activeView === 'explainability' && 'Model Explainability'}{activeView === 'history' && 'Analysis History'}
            </h2>
            <p className="text-sm text-muted-foreground">{activeView === 'dashboard' && 'Monitor your ML operations and model performance'}{activeView === 'analysis' && 'Upload data, select a target variable, and train ML models'}{activeView === 'predict' && 'Make predictions, view results, and explore visualizations'}{activeView === 'anomalies' && 'Detect outliers and unusual patterns in your data'}{activeView === 'models' && 'Manage, export, and import your trained models'}{activeView === 'explore' && 'Explore data distributions, correlations, and patterns'}{activeView === 'explainability' && 'Understand why the model made its predictions using SHAP and LIME'}{activeView === 'history' && 'View, restore, and share your saved analysis sessions'}</p>
          </div><div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setShowGuide(prev => !prev)} data-testid="guide-toggle-btn" className={showGuide ? 'bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-700' : ''}><BookOpen className="h-4 w-4 mr-2" />{showGuide ? 'Hide Guide' : 'Help Guide'}</Button>
            {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button variant="outline" size="sm" onClick={() => handleSaveAnalysis()} data-testid="save-analysis-btn"><Save className="h-4 w-4 mr-2" />Save</Button>}
            {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button variant="outline" size="sm" onClick={handleShareAnalysis} data-testid="share-analysis-btn"><Share2 className="h-4 w-4 mr-2" />Share Analysis</Button>}
            {(trainingResult || shapGlobal || limeResult || predictionHistory?.length > 0) && !viewOnlyMode && <><Button variant="outline" size="sm" onClick={handleExportSheets} data-testid="export-sheets-btn" title="Export data as CSV for Google Sheets"><Sheet className="h-4 w-4 mr-2" />Export to Google Sheets</Button>
              <Button variant="outline" size="sm" onClick={handleExportCSV} data-testid="export-csv-btn"><Download className="h-4 w-4 mr-2" />Download CSV</Button></>}
            {csvText && !viewOnlyMode && <Button variant="outline" size="sm" onClick={handleClearSession} data-testid="clear-session-btn"><Trash2 className="h-4 w-4 mr-2" />Clear Session</Button>}
            <Button variant="outline" size="icon" onClick={() => setDarkMode(prev => !prev)} data-testid="dark-mode-toggle">{darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
          </div></div>
        </motion.header>

        <AnimatePresence>{error && <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="mx-8 mt-4" data-testid="error-banner"><Card className="border-destructive bg-destructive/10"><CardContent className="p-4"><p className="text-sm text-destructive font-medium flex items-center gap-2"><AlertCircle className="h-4 w-4" /> {error}</p></CardContent></Card></motion.div>}</AnimatePresence>

        {/* ==================== GUIDE PANEL ==================== */}
        <AnimatePresence>{showGuide && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="mx-8 mt-4 overflow-hidden" data-testid="guide-panel">
            <Card className="border-blue-200 dark:border-blue-800 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20">
              <CardContent className="p-5">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2"><BookOpen className="h-5 w-5 text-blue-600" /><span className="font-semibold text-blue-900 dark:text-blue-300">Getting Started Guide</span></div>
                  <Button variant="ghost" size="sm" onClick={() => setShowGuide(false)} className="text-muted-foreground h-7 text-xs">Dismiss</Button>
                </div>
                <div className="grid gap-3 md:grid-cols-4 lg:grid-cols-7">
                  {GUIDE_STEPS.map(s => {
                    const done = s.step === 1 ? !!csvText : s.step === 2 ? (cleaningLog.length > 0 || (datasetScan?.score >= 70)) : s.step === 3 ? !!targetColumn : s.step === 4 ? (!!trainingResult || !!unsupervisedResult) : s.step === 5 ? (!!trainingResult || !!unsupervisedResult) : s.step === 6 ? predictionHistory.length > 0 : (!!shapGlobal || !!limeResult);
                    return <div key={s.step} className={`rounded-lg border p-3 transition-colors ${done ? 'bg-emerald-50 dark:bg-emerald-950/20 border-emerald-300 dark:border-emerald-700' : 'bg-white dark:bg-card border-border'}`} data-testid={`guide-step-${s.step}`}>
                      <div className="flex items-center gap-2 mb-1.5">
                        {done ? <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" /> : <span className="h-5 w-5 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 text-[10px] font-bold flex items-center justify-center shrink-0">{s.step}</span>}
                        <span className="text-xs font-semibold">{s.title}</span>
                      </div>
                      <p className="text-[10px] text-muted-foreground leading-relaxed">{s.desc}</p>
                    </div>;
                  })}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}</AnimatePresence>

        {/* ==================== VIEW-ONLY BANNER ==================== */}
        {viewOnlyMode && (
          <div className="mx-8 mt-4 p-3 rounded-lg border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/20 flex items-center justify-between" data-testid="view-only-banner">
            <div className="flex items-center gap-2"><Lock className="h-4 w-4 text-amber-600" /><span className="text-sm font-medium text-amber-800 dark:text-amber-300">This is a shared analysis (view-only). Request access to edit.</span></div>
            <Button variant="outline" size="sm" onClick={() => { setViewOnlyMode(false); window.history.replaceState({}, '', window.location.pathname); }} className="text-xs border-amber-300" data-testid="exit-view-only-btn">Exit View-Only</Button>
          </div>
        )}

        {/* ==================== SHARE URL TOAST ==================== */}
        <AnimatePresence>{shareUrl && (
          <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: -20, opacity: 0 }} className="mx-8 mt-4 p-3 rounded-lg border border-emerald-300 dark:border-emerald-700 bg-emerald-50 dark:bg-emerald-950/20" data-testid="share-url-toast">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {shareCopyStatus === 'copied' ? <><CheckCircle2 className="h-4 w-4 text-emerald-600" /><span className="text-sm font-medium text-emerald-800 dark:text-emerald-300" data-testid="share-copy-success">Link copied to clipboard</span></> : <><AlertCircle className="h-4 w-4 text-amber-600" /><span className="text-sm font-medium text-amber-800 dark:text-amber-300" data-testid="share-copy-fallback">Copy not supported here. Please copy manually.</span></>}
              </div>
              <Button variant="ghost" size="sm" onClick={() => { setShareUrl(''); setShareCopyStatus(''); }} className="text-xs"><XCircle className="h-3.5 w-3.5" /></Button>
            </div>
            <div className="flex items-center gap-2">
              <input type="text" readOnly value={shareUrl} onClick={e => e.target.select()} className="flex-1 text-xs bg-white dark:bg-card border rounded px-2 py-1.5 font-mono select-all focus:outline-none focus:ring-1 focus:ring-emerald-400" data-testid="share-url-input" />
              <Button variant="outline" size="sm" onClick={async () => { const ok = await safeCopyToClipboard(shareUrl); setShareCopyStatus(ok ? 'copied' : 'manual'); }} className="text-xs shrink-0" data-testid="copy-share-url-btn"><Copy className="h-3.5 w-3.5 mr-1" />Copy</Button>
            </div>
          </motion.div>
        )}</AnimatePresence>

        <main className="p-8"><AnimatePresence mode="wait">

          {/* ==================== DASHBOARD ==================== */}
          {activeView === 'dashboard' && (
            <motion.div key="dashboard" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="dashboard-view">
              {models.length === 0 ? (
                <motion.div variants={fadeInUp} className="space-y-6">
                  <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
                    <Database className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                    <h3 className="text-lg font-semibold mb-2" data-testid="dashboard-empty-title">{dataProfile ? 'No Models Trained Yet' : 'No Dataset Uploaded Yet'}</h3>
                    <p className="text-muted-foreground mb-6 max-w-md mx-auto text-sm" data-testid="dashboard-empty-msg">{dataProfile ? 'Dataset loaded. Go to Analysis to select a target variable and train your first model.' : 'Upload a CSV file or pick a sample dataset. The system will guide you through each step.'}</p>
                    <Button size="lg" onClick={() => setActiveView('analysis')} data-testid="train-first-model-btn"><Zap className="h-4 w-4 mr-2" />{dataProfile ? 'Train Your First Model' : 'Upload Dataset'}</Button>
                    {!showGuide && <div className="mt-4"><Button variant="link" size="sm" onClick={() => setShowGuide(true)} className="text-blue-600" data-testid="show-guide-from-dashboard"><BookOpen className="h-3.5 w-3.5 mr-1.5 inline" />New here? Open the Getting Started Guide</Button></div>}
                  </CardContent></Card>
                  {datasetScan && (
                    <Card data-testid="dataset-health-empty"><CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" />Dataset Health</CardTitle></CardHeader><CardContent>
                      <div className={`p-3 rounded-lg border-2 flex items-center gap-3 ${datasetScan.ready ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20' : 'border-orange-500 bg-orange-50 dark:bg-orange-950/20'}`}>
                        {datasetScan.ready ? <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" /> : <XCircle className="h-5 w-5 text-orange-600 shrink-0" />}
                        <p className="font-medium text-sm">{datasetScan.ready ? 'Dataset Ready for Training' : 'Dataset Needs Cleaning'}</p>
                        <Badge className="ml-auto" variant={datasetScan.score >= 80 ? 'default' : datasetScan.score >= 50 ? 'secondary' : 'destructive'}>{datasetScan.score}/100</Badge>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-sm">
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Rows</p><p className="font-bold text-lg">{datasetScan.rows}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Columns</p><p className="font-bold text-lg">{datasetScan.columns}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Missing</p><p className="font-bold text-lg">{datasetScan.totalMissing}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Outliers</p><p className="font-bold text-lg">{datasetScan.totalOutliers}</p></div>
                      </div>
                    </CardContent></Card>
                  )}
                </motion.div>
              ) : (<>
                {/* Stat Cards */}
                <motion.div variants={staggerContainer} className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
                  <StatCard title="Total Models" value={stats.totalModels} metricValue={stats.totalModels} icon={Database} />
                  <StatCard title="Avg Score" value={`${(stats.avgScore * 100).toFixed(1)}%`} metricValue={`${(stats.avgScore * 100).toFixed(0)}%`} icon={TrendingUp} />
                  <StatCard title="Best Algorithm" value={stats.bestAlgoByScore} icon={Trophy} />
                  <StatCard title="Highest Score" value={`${(stats.highestScore * 100).toFixed(1)}%`} metricValue={`${(stats.highestScore * 100).toFixed(0)}%`} icon={Sparkles} />
                  <StatCard title="Last Training" value={stats.lastTraining ? new Date(stats.lastTraining).toLocaleDateString() : '--'} icon={Clock} />
                </motion.div>

                {/* Quick Insights */}
                <motion.div variants={fadeInUp} className="grid gap-4 md:grid-cols-3" data-testid="quick-insights">
                  <Card><CardContent className="p-4 flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center shrink-0"><Trophy className="h-5 w-5 text-emerald-600" /></div>
                    <div><p className="text-xs text-muted-foreground">Best Performing</p><p className="font-semibold text-sm" data-testid="insight-best">{stats.bestAlgoByScore}</p></div>
                  </CardContent></Card>
                  <Card><CardContent className="p-4 flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center shrink-0"><Activity className="h-5 w-5 text-blue-600" /></div>
                    <div><p className="text-xs text-muted-foreground">Most Used</p><p className="font-semibold text-sm" data-testid="insight-most-used">{stats.mostUsedAlgo}</p></div>
                  </CardContent></Card>
                  <Card><CardContent className="p-4 flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center shrink-0"><Sparkles className="h-5 w-5 text-amber-600" /></div>
                    <div><p className="text-xs text-muted-foreground"><MetricTip metricKey="accuracy" value={stats.highestScore}>Highest Accuracy</MetricTip></p><p className="font-semibold text-sm" data-testid="insight-highest">{(stats.highestScore * 100).toFixed(1)}%</p></div>
                  </CardContent></Card>
                </motion.div>

                {/* Dataset Health + Model Leaderboard */}
                <div className="grid gap-6 lg:grid-cols-2">
                  <motion.div variants={fadeInUp}><Card className="h-full" data-testid="dataset-health-widget"><CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" />Dataset Health</CardTitle></CardHeader><CardContent>
                    {datasetScan ? (<div className="space-y-4">
                      <div className={`p-3 rounded-lg border-2 flex items-center gap-3 ${datasetScan.ready ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20' : 'border-orange-500 bg-orange-50 dark:bg-orange-950/20'}`} data-testid="dataset-readiness">
                        {datasetScan.ready ? <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" /> : <XCircle className="h-5 w-5 text-orange-600 shrink-0" />}
                        <p className="font-medium text-sm">{datasetScan.ready ? 'Dataset Ready for Training' : 'Dataset Needs Cleaning'}</p>
                        <Badge className="ml-auto" variant={datasetScan.score >= 80 ? 'default' : datasetScan.score >= 50 ? 'secondary' : 'destructive'} data-testid="health-score">{datasetScan.score}/100</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Rows</p><p className="font-bold text-lg" data-testid="scan-rows">{datasetScan.rows}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Columns</p><p className="font-bold text-lg" data-testid="scan-cols">{datasetScan.columns}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Missing Values</p><p className="font-bold text-lg" data-testid="scan-missing">{datasetScan.totalMissing}</p></div>
                        <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Outliers</p><p className="font-bold text-lg" data-testid="scan-outliers">{datasetScan.totalOutliers}</p></div>
                      </div>
                      {datasetScan.warnings.length > 0 && <div className="space-y-1.5">{datasetScan.warnings.map((w, i) => <div key={i} className="flex items-center gap-2 text-xs text-orange-600"><AlertCircle className="h-3 w-3 shrink-0" />{w}</div>)}</div>}
                    </div>) : (
                      <div className="text-center py-8"><Shield className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" /><p className="text-sm text-muted-foreground">Load a dataset in Analysis to see health metrics</p></div>
                    )}
                  </CardContent></Card></motion.div>

                  <motion.div variants={fadeInUp}><Card className="h-full" data-testid="model-leaderboard"><CardHeader><CardTitle className="flex items-center gap-2"><Trophy className="h-5 w-5" />Top Models</CardTitle><CardDescription>Top 5 by performance score</CardDescription></CardHeader><CardContent>
                    <div className="space-y-2">{topModels.map((model, idx) => {
                      const score = model.problemType === 'classification' ? (model.metrics?.accuracy || 0) : (model.metrics?.r2 || 0);
                      return (<div key={model.modelId} className="flex items-center gap-3 p-2.5 rounded-lg border hover:bg-accent/50 transition-colors" data-testid={`top-model-${idx}`}>
                        <Badge variant={idx === 0 ? 'default' : 'secondary'} className="shrink-0 w-7 justify-center">{idx + 1}</Badge>
                        <div className="flex-1 min-w-0"><p className="font-medium text-sm truncate">{ALGO_NAMES[model.algorithm] || model.algorithm}</p><p className="text-xs text-muted-foreground">{model.problemType}{model.evalMode === 'cv' ? ' (CV)' : ''}</p></div>
                        <span className="font-mono text-sm font-semibold shrink-0" style={{color: ALGO_COLORS[model.algorithm]}}>{(score * 100).toFixed(1)}%</span>
                      </div>);
                    })}</div>
                  </CardContent></Card></motion.div>
                </div>

                {/* Charts Row */}
                <div className="grid gap-6 lg:grid-cols-2">
                  <motion.div variants={fadeInUp}><Card className="h-[400px]" data-testid="model-performance-chart"><CardHeader><CardTitle>Model Performance</CardTitle><CardDescription>Score per trained model</CardDescription></CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={270}><BarChart data={models.map((m, i) => ({
                      name: `${ALGO_NAMES[m.algorithm] || m.algorithm}`.substring(0, 12), score: +((m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || 0)) * 100).toFixed(1)
                    }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-30} textAnchor="end" height={80} tick={{fontSize: 10}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={v => `${v}%`} /><Bar dataKey="score" radius={[4, 4, 0, 0]}>{models.map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
                  </CardContent></Card></motion.div>

                  <motion.div variants={fadeInUp}><Card className="h-[400px]" data-testid="algorithm-usage-chart"><CardHeader><CardTitle>Algorithm Usage</CardTitle><CardDescription>Training frequency per algorithm</CardDescription></CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={270}><PieChart><Pie data={(() => {
                      const c = {}; models.forEach(m => { const nm = ALGO_NAMES[m.algorithm] || m.algorithm; c[nm] = (c[nm] || 0) + 1; });
                      return Object.entries(c).map(([name, value], i) => ({ name, value, fill: Object.values(ALGO_COLORS)[i % Object.values(ALGO_COLORS).length] }));
                    })()} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({name, percent}) => `${name} ${(percent*100).toFixed(0)}%`} /><Tooltip /></PieChart></ResponsiveContainer>
                  </CardContent></Card></motion.div>
                </div>

                {/* Training Timeline */}
                <motion.div variants={fadeInUp}><Card data-testid="training-timeline"><CardHeader><CardTitle>Training Timeline</CardTitle><CardDescription>Cumulative model count over time</CardDescription></CardHeader><CardContent>
                  <ResponsiveContainer width="100%" height={250}><LineChart data={(() => {
                    const dc = {}; models.forEach(m => { const d = m.createdAt ? new Date(m.createdAt).toLocaleDateString() : 'Today'; dc[d] = (dc[d] || 0) + 1; });
                    let cum = 0; return Object.entries(dc).map(([date, count]) => { cum += count; return { date, count, cumulative: cum }; });
                  })()}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" /><YAxis /><Tooltip /><Legend /><Line type="monotone" dataKey="cumulative" name="Total Models" stroke="hsl(var(--primary))" strokeWidth={2} dot={{fill:'hsl(var(--primary))'}} /><Line type="monotone" dataKey="count" name="Trained" stroke="#16a34a" strokeWidth={2} dot={{fill:'#16a34a'}} /></LineChart></ResponsiveContainer>
                </CardContent></Card></motion.div>

                {/* Recent Models Table */}
                <motion.div variants={fadeInUp}><Card data-testid="recent-models-table"><CardHeader><CardTitle>Recent Models</CardTitle><CardDescription>Latest trained models</CardDescription></CardHeader><CardContent>
                  <div className="rounded-md border overflow-auto"><table className="w-full text-sm">
                    <thead><tr className="border-b bg-muted/50">
                      <th className="p-3 text-left font-medium">Algorithm</th>
                      <th className="p-3 text-left font-medium">Problem Type</th>
                      <th className="p-3 text-left font-medium">Target</th>
                      <th className="p-3 text-left font-medium">Eval Mode</th>
                      <th className="p-3 text-right font-medium">Score</th>
                      <th className="p-3 text-right font-medium">Date</th>
                    </tr></thead>
                    <tbody>{models.slice(-10).reverse().map((model, idx) => {
                      const score = model.problemType === 'classification' ? (model.metrics?.accuracy || 0) : (model.metrics?.r2 || 0);
                      return (<tr key={model.modelId} className="border-b last:border-0 hover:bg-accent/50 transition-colors" data-testid={`recent-model-row-${idx}`}>
                        <td className="p-3"><div className="flex items-center gap-2"><div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center shrink-0"><Brain className="h-3.5 w-3.5 text-primary" /></div><span className="font-medium">{ALGO_NAMES[model.algorithm] || model.algorithm}</span></div></td>
                        <td className="p-3"><Badge variant="outline">{model.problemType || '—'}</Badge></td>
                        <td className="p-3 text-xs font-mono">{model.targetColumn || '—'}</td>
                        <td className="p-3"><Badge variant="secondary">{model.evalMode === 'cv' ? '5-Fold CV' : 'Train/Test'}</Badge></td>
                        <td className="p-3 text-right font-mono font-semibold">{(score * 100).toFixed(1)}%</td>
                        <td className="p-3 text-right text-muted-foreground text-xs">{model.createdAt ? new Date(model.createdAt).toLocaleDateString() : '—'}</td>
                      </tr>);
                    })}</tbody>
                  </table></div>
                </CardContent></Card></motion.div>
              </>)}
            </motion.div>
          )}

          {/* ==================== ANALYSIS ==================== */}
          {activeView === 'analysis' && (
            <motion.div key="analysis" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="analysis-view">
              <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Sample Data</CardTitle></CardHeader>
                <CardContent><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">{sampleDatasets.map((sample, idx) => <Card key={idx} className="cursor-pointer hover:shadow-md transition-shadow border-2 hover:border-primary" onClick={() => handleCsvTextChange(sample.data)} data-testid={`sample-dataset-${idx}`}><CardContent className="p-4"><div className="flex items-center justify-between"><div><p className="font-medium">{sample.name}</p><p className="text-sm text-muted-foreground">{sample.description}</p></div><ChevronRight className="h-5 w-5 text-muted-foreground" /></div></CardContent></Card>)}</div></CardContent></Card></motion.div>

              <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><Upload className="h-5 w-5" />Upload Data</CardTitle></CardHeader>
                <CardContent className="space-y-4">
                  <div onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'} hover:border-primary hover:bg-accent/50`} data-testid="csv-dropzone"><input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="csv-file-input" /><Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" /><p className="text-lg font-medium mb-2">Drop your CSV file here</p><p className="text-sm text-muted-foreground mb-3">or click to browse</p><p className="text-xs text-muted-foreground/60">Upload a clean dataset in CSV format. Ensure it has meaningful column headers and at least 10 rows for reliable results.</p></div>
                  <Separator className="my-6" />
                  <div><label className="text-sm font-medium mb-2 block">Or paste CSV data:</label><textarea value={csvText} onChange={(e) => handleCsvTextChange(e.target.value)} placeholder="Paste CSV data..." rows={6} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2" data-testid="csv-text-input" /></div>
                </CardContent></Card></motion.div>

              {dataProfile && <motion.div variants={fadeInUp}><Card className="border-2 border-blue-500/30" data-testid="data-profile-card"><CardHeader><CardTitle className="flex items-center gap-2"><Table2 className="h-5 w-5" />Dataset Profile</CardTitle><CardDescription>{dataProfile.rowCount} rows x {dataProfile.columnCount} columns</CardDescription></CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-3"><div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Numeric</p><p className="text-2xl font-bold text-blue-600">{dataProfile.numericColumns.length}</p></div><div className="bg-emerald-50 dark:bg-emerald-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Categorical</p><p className="text-2xl font-bold text-emerald-600">{dataProfile.categoricalColumns.length}</p></div><div className="bg-violet-50 dark:bg-violet-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Rows</p><p className="text-2xl font-bold text-violet-600">{dataProfile.rowCount}</p></div></div>
                  <div className="rounded-md border overflow-auto max-h-64"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Column</th><th className="p-2 text-left font-medium">Type</th><th className="p-2 text-left font-medium">Unique</th><th className="p-2 text-left font-medium">Range / Values</th></tr></thead><tbody>{dataProfile.columns.map((col, idx) => <tr key={idx} className="border-b last:border-0"><td className="p-2 font-mono text-xs">{col.name}</td><td className="p-2"><Badge variant={col.type === 'numeric' ? 'default' : 'secondary'} className="text-xs">{col.type}</Badge></td><td className="p-2 text-xs">{col.uniqueCount}</td><td className="p-2 text-xs text-muted-foreground">{col.type === 'numeric' ? `${col.min?.toFixed(1)} — ${col.max?.toFixed(1)} (mean: ${col.mean?.toFixed(1)})` : col.sampleValues.slice(0, 3).join(', ')}</td></tr>)}</tbody></table></div>
                  {taskSuggestion && <div className={`p-4 rounded-lg border-2 flex items-start gap-3 ${taskSuggestion.task === 'regression' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : taskSuggestion.task === 'classification' ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30' : 'border-violet-500 bg-violet-50 dark:bg-violet-950/30'}`} data-testid="task-suggestion"><Info className="h-5 w-5 mt-0.5 shrink-0" /><div><p className="font-semibold text-sm">{taskSuggestion.message}</p>{taskSuggestion.task === 'clustering' && <Button size="sm" className="mt-2" onClick={() => setActiveView('clusters')} data-testid="go-to-clusters-btn"><Layers className="h-3 w-3 mr-1" />Go to Clustering</Button>}</div></div>}
                </CardContent></Card></motion.div>}

              {/* ==================== DATASET SUMMARY ==================== */}
              {datasetSummary && <motion.div variants={fadeInUp}><Card className="border-2 border-indigo-500/30" data-testid="dataset-summary-card">
                <CardHeader><CardTitle className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-indigo-500" />Dataset Summary</CardTitle>
                  <CardDescription>Auto-generated overview of your data</CardDescription></CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="outline" className="text-xs border-indigo-300 text-indigo-700 dark:text-indigo-300" data-testid="dataset-domain-badge">{datasetSummary.domain}</Badge>
                  </div>
                  <div className="space-y-1.5" data-testid="dataset-summary-text">
                    {datasetSummary.description.map((line, i) => <p key={i} className="text-sm text-muted-foreground leading-relaxed">{line}</p>)}
                  </div>
                  <div className="p-3 rounded-lg bg-indigo-50 dark:bg-indigo-950/20 border border-indigo-200 dark:border-indigo-800" data-testid="dataset-focus-line">
                    <p className="text-sm font-medium text-indigo-800 dark:text-indigo-300">{datasetSummary.focusLine}</p>
                  </div>
                  <div>
                    <p className="text-xs font-semibold mb-2 text-muted-foreground uppercase tracking-wide">Key Variables</p>
                    <div className="flex flex-wrap gap-2" data-testid="dataset-key-variables">
                      {datasetSummary.keyVariables.map((v, i) => <div key={i} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border bg-background text-xs">
                        <span className="font-mono font-medium">{v.name}</span>
                        <Badge variant={v.type === 'numeric' ? 'default' : 'secondary'} className="text-[10px] px-1 py-0">{v.type}</Badge>
                      </div>)}
                    </div>
                  </div>
                  {datasetSummary.possibleTarget && <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800 flex items-start gap-2" data-testid="dataset-target-suggestion">
                    <Target className="h-4 w-4 text-emerald-600 mt-0.5 shrink-0" />
                    <div><p className="text-sm font-medium text-emerald-800 dark:text-emerald-300">Suggested target: <span className="font-mono">{datasetSummary.possibleTarget.name}</span></p>
                      <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-0.5">Detected as {datasetSummary.possibleTarget.reason} — suitable for {datasetSummary.possibleTarget.task}.</p></div>
                  </div>}
                </CardContent></Card></motion.div>}

              {/* ==================== DATASET SCANNER ==================== */}
              {datasetScan && <motion.div variants={fadeInUp}><Card className={`border-2 ${datasetScan.score >= 70 ? 'border-emerald-500/30' : datasetScan.score >= 50 ? 'border-amber-500/30' : 'border-red-500/30'}`} data-testid="dataset-scanner">
                <CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" />Dataset Scanner Report</CardTitle>
                  <CardDescription>Automated data quality analysis</CardDescription></CardHeader>
                <CardContent className="space-y-5">

                  {/* Health Score Banner */}
                  <div className={`p-4 rounded-lg flex items-center justify-between ${datasetScan.score >= 70 ? 'bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200' : datasetScan.score >= 50 ? 'bg-amber-50 dark:bg-amber-950/20 border border-amber-200' : 'bg-red-50 dark:bg-red-950/20 border border-red-200'}`} data-testid="scanner-health-banner">
                    <div className="flex items-center gap-3">
                      {datasetScan.score >= 70 ? <CheckCircle2 className="h-6 w-6 text-emerald-600" /> : datasetScan.score >= 50 ? <AlertCircle className="h-6 w-6 text-amber-600" /> : <XCircle className="h-6 w-6 text-red-600" />}
                      <div><p className="font-semibold">{datasetScan.score >= 70 ? 'Dataset Ready for Training' : datasetScan.score >= 50 ? 'Minor Issues Detected' : 'Dataset Needs Cleaning'}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">{datasetScan.warnings.length === 0 ? 'No issues found' : `${datasetScan.warnings.length} issue(s) detected`}</p></div>
                    </div>
                    <div className={`text-3xl font-bold ${datasetScan.score >= 70 ? 'text-emerald-600' : datasetScan.score >= 50 ? 'text-amber-600' : 'text-red-600'}`} data-testid="scanner-score">{datasetScan.score}<span className="text-sm font-normal text-muted-foreground">/100</span></div>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-muted/50 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Rows</p><p className="text-xl font-bold">{datasetScan.rows}</p></div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Columns</p><p className="text-xl font-bold">{datasetScan.columns}</p></div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Numeric</p><p className="text-xl font-bold text-blue-600">{datasetScan.numericCount}</p></div>
                    <div className="bg-muted/50 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Categorical</p><p className="text-xl font-bold text-emerald-600">{datasetScan.categoricalCount}</p></div>
                  </div>

                  {/* Issues Found */}
                  {datasetScan.warnings.length > 0 && <div className="space-y-3" data-testid="scanner-issues">
                    <p className="text-sm font-semibold">Issues Found</p>

                    {datasetScan.totalMissing > 0 && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-missing">
                      <div className="flex items-center justify-between"><p className="text-sm font-medium">Missing Values</p><Badge variant="secondary">{datasetScan.totalMissing} total</Badge></div>
                      <div className="mt-2 space-y-1">{datasetScan.missingCols.map((mc, i) => <div key={i} className="flex justify-between text-xs"><span className="font-mono">{mc.col}</span><span className="text-muted-foreground">{mc.count} ({mc.pct}%)</span></div>)}</div>
                      <p className="text-xs text-muted-foreground mt-2 italic">Recommendation: Fill with median (numeric) or mode (categorical)</p>
                    </div>}

                    {datasetScan.duplicateCount > 0 && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-duplicates">
                      <div className="flex items-center justify-between"><p className="text-sm font-medium">Duplicate Rows</p><Badge variant="secondary">{datasetScan.duplicateCount} ({(datasetScan.duplicateCount / datasetScan.rows * 100).toFixed(1)}%)</Badge></div>
                      <p className="text-xs text-muted-foreground mt-1 italic">Recommendation: Remove duplicate rows to avoid bias</p>
                    </div>}

                    {datasetScan.totalOutliers > 0 && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-outliers">
                      <div className="flex items-center justify-between"><p className="text-sm font-medium">Outliers Detected</p><Badge variant="secondary">{datasetScan.totalOutliers} total</Badge></div>
                      <div className="mt-2 space-y-1">{datasetScan.outlierCols.map((oc, i) => <div key={i} className="flex justify-between text-xs"><span className="font-mono">{oc.col}</span><span className="text-muted-foreground">{oc.count} outliers</span></div>)}</div>
                      <p className="text-xs text-muted-foreground mt-2 italic">Recommendation: Remove extreme outliers using IQR method</p>
                    </div>}

                    {datasetScan.constantCols.length > 0 && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-constants">
                      <div className="flex items-center justify-between"><p className="text-sm font-medium">Constant Columns</p><Badge variant="secondary">{datasetScan.constantCols.length}</Badge></div>
                      <p className="text-xs mt-1">Columns: <span className="font-mono">{datasetScan.constantCols.join(', ')}</span></p>
                      <p className="text-xs text-muted-foreground mt-1 italic">Recommendation: Drop columns with no variance</p>
                    </div>}

                    {datasetScan.highCorr.length > 0 && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-correlation">
                      <div className="flex items-center justify-between"><p className="text-sm font-medium">High Correlations (&gt;0.9)</p><Badge variant="secondary">{datasetScan.highCorr.length} pair(s)</Badge></div>
                      <div className="mt-2 space-y-1">{datasetScan.highCorr.map((hc, i) => <div key={i} className="text-xs"><span className="font-mono">{hc.col1}</span> &harr; <span className="font-mono">{hc.col2}</span> <span className="text-muted-foreground">(r={hc.r})</span></div>)}</div>
                      <p className="text-xs text-muted-foreground mt-2 italic">Recommendation: Consider removing one feature from each pair</p>
                    </div>}

                    {datasetScan.targetInfo?.imbalanced && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-imbalance">
                      <p className="text-sm font-medium">Class Imbalance</p>
                      <p className="text-xs mt-1">Majority class: {datasetScan.targetInfo.majorityPct}%</p>
                      <p className="text-xs text-muted-foreground mt-1 italic">Warning: Imbalanced classes may bias model predictions</p>
                    </div>}

                    {datasetScan.scaleIssue && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-scaling">
                      <p className="text-sm font-medium">Feature Scale Differences</p>
                      <p className="text-xs text-muted-foreground mt-1 italic">Recommendation: Normalize numeric features for better model performance</p>
                    </div>}

                    {datasetScan.sizeWarning && <div className="p-3 rounded-lg border bg-amber-50/50 dark:bg-amber-950/10" data-testid="issue-size">
                      <p className="text-sm font-medium">Small Dataset Warning</p>
                      <p className="text-xs text-muted-foreground mt-1 italic">Dataset has fewer than 100 rows. Results may be unreliable.</p>
                    </div>}

                    {datasetScan.targetInfo && <div className="p-3 rounded-lg border bg-blue-50/50 dark:bg-blue-950/10" data-testid="target-validation">
                      <p className="text-sm font-medium">Target Variable</p>
                      <p className="text-xs mt-1">Task: <Badge variant="outline" className="text-xs">{datasetScan.targetInfo.task}</Badge> &middot; Unique values: {datasetScan.targetInfo.uniqueValues}</p>
                    </div>}
                  </div>}

                  {/* Auto-Clean Actions */}
                  <div data-testid="auto-clean-actions">
                    <p className="text-sm font-semibold mb-3">Auto-Clean Actions</p>
                    <div className="flex flex-wrap gap-2">
                      <Button size="sm" variant="outline" onClick={() => handleClean('duplicates')} disabled={!datasetScan.duplicateCount} data-testid="clean-duplicates"><Trash2 className="h-3 w-3 mr-1" />Remove Duplicates{datasetScan.duplicateCount > 0 && ` (${datasetScan.duplicateCount})`}</Button>
                      <Button size="sm" variant="outline" onClick={() => handleClean('missing')} disabled={!datasetScan.totalMissing} data-testid="clean-missing"><Zap className="h-3 w-3 mr-1" />Fill Missing{datasetScan.totalMissing > 0 && ` (${datasetScan.totalMissing})`}</Button>
                      <Button size="sm" variant="outline" onClick={() => handleClean('outliers')} disabled={!datasetScan.totalOutliers} data-testid="clean-outliers"><ShieldAlert className="h-3 w-3 mr-1" />Remove Outliers{datasetScan.totalOutliers > 0 && ` (${datasetScan.totalOutliers})`}</Button>
                      <Button size="sm" variant="outline" onClick={() => handleClean('constants')} disabled={!datasetScan.constantCols.length} data-testid="clean-constants"><Trash2 className="h-3 w-3 mr-1" />Drop Constants{datasetScan.constantCols.length > 0 && ` (${datasetScan.constantCols.length})`}</Button>
                      <Button size="sm" variant="outline" onClick={() => handleClean('normalize')} disabled={!datasetScan.scaleIssue && datasetScan.numericCount === 0} data-testid="clean-normalize"><Activity className="h-3 w-3 mr-1" />Normalize Features</Button>
                    </div>
                  </div>

                  {/* Before vs After Comparison */}
                  {precleanScan && cleaningLog.length > 0 && <div data-testid="before-after-comparison">
                    <p className="text-sm font-semibold mb-3">Before vs After Cleaning</p>
                    <div className="rounded-md border overflow-auto"><table className="w-full text-sm">
                      <thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Metric</th><th className="p-2 text-right font-medium">Before</th><th className="p-2 text-right font-medium">After</th><th className="p-2 text-right font-medium">Change</th></tr></thead>
                      <tbody>
                        <tr className="border-b"><td className="p-2">Rows</td><td className="p-2 text-right font-mono">{precleanScan.rows}</td><td className="p-2 text-right font-mono">{datasetScan.rows}</td><td className="p-2 text-right font-mono text-xs">{datasetScan.rows - precleanScan.rows !== 0 ? `${datasetScan.rows - precleanScan.rows}` : '—'}</td></tr>
                        <tr className="border-b"><td className="p-2">Missing Values</td><td className="p-2 text-right font-mono">{precleanScan.totalMissing}</td><td className="p-2 text-right font-mono">{datasetScan.totalMissing}</td><td className="p-2 text-right font-mono text-xs text-emerald-600">{precleanScan.totalMissing - datasetScan.totalMissing > 0 ? `−${precleanScan.totalMissing - datasetScan.totalMissing}` : '—'}</td></tr>
                        <tr className="border-b"><td className="p-2">Duplicate Rows</td><td className="p-2 text-right font-mono">{precleanScan.duplicateCount}</td><td className="p-2 text-right font-mono">{datasetScan.duplicateCount}</td><td className="p-2 text-right font-mono text-xs text-emerald-600">{precleanScan.duplicateCount - datasetScan.duplicateCount > 0 ? `−${precleanScan.duplicateCount - datasetScan.duplicateCount}` : '—'}</td></tr>
                        <tr className="border-b"><td className="p-2">Outliers</td><td className="p-2 text-right font-mono">{precleanScan.totalOutliers}</td><td className="p-2 text-right font-mono">{datasetScan.totalOutliers}</td><td className="p-2 text-right font-mono text-xs text-emerald-600">{precleanScan.totalOutliers - datasetScan.totalOutliers > 0 ? `−${precleanScan.totalOutliers - datasetScan.totalOutliers}` : '—'}</td></tr>
                        <tr className="border-b"><td className="p-2">Constant Columns</td><td className="p-2 text-right font-mono">{precleanScan.constantCols.length}</td><td className="p-2 text-right font-mono">{datasetScan.constantCols.length}</td><td className="p-2 text-right font-mono text-xs text-emerald-600">{precleanScan.constantCols.length - datasetScan.constantCols.length > 0 ? `−${precleanScan.constantCols.length - datasetScan.constantCols.length}` : '—'}</td></tr>
                        <tr><td className="p-2 font-semibold">Health Score</td><td className="p-2 text-right font-mono font-bold">{precleanScan.score}</td><td className={`p-2 text-right font-mono font-bold ${datasetScan.score >= 70 ? 'text-emerald-600' : datasetScan.score >= 50 ? 'text-amber-600' : 'text-red-600'}`}>{datasetScan.score}</td><td className="p-2 text-right font-mono text-xs text-emerald-600 font-bold">{datasetScan.score - precleanScan.score > 0 ? `+${datasetScan.score - precleanScan.score}` : '—'}</td></tr>
                      </tbody>
                    </table></div>
                  </div>}

                  {/* Cleaning Action Log */}
                  {cleaningLog.length > 0 && <div data-testid="cleaning-log">
                    <p className="text-sm font-semibold mb-2">Cleaning Actions Performed</p>
                    <div className="space-y-1.5">{cleaningLog.map((entry, i) => <div key={i} className="flex items-center gap-2 text-xs p-2 rounded bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800"><CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" /><span>{entry}</span></div>)}</div>
                  </div>}

                  {/* Cleaned Data Preview */}
                  {dataPreview && <div data-testid="cleaned-preview">
                    <p className="text-sm font-semibold mb-2">Cleaned Dataset Preview (first 10 rows)</p>
                    <div className="rounded-md border overflow-auto max-h-64"><table className="w-full text-xs">
                      <thead><tr className="border-b bg-muted/50">{dataPreview.headers.map((h, i) => <th key={i} className="p-2 text-left font-medium font-mono whitespace-nowrap">{h}</th>)}</tr></thead>
                      <tbody>{dataPreview.rows.map((row, ri) => <tr key={ri} className="border-b last:border-0">{dataPreview.headers.map((h, ci) => <td key={ci} className="p-2 font-mono whitespace-nowrap">{String(row[h] ?? '').substring(0, 20)}</td>)}</tr>)}</tbody>
                    </table></div>
                  </div>}

                </CardContent>
              </Card></motion.div>}

              {columns.length > 0 && <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><Target className="h-5 w-5" />Model Configuration</CardTitle>
                <CardDescription>Configure your model by selecting a target variable and algorithm. The system will guide you based on your data.</CardDescription></CardHeader>
                <CardContent>
                  <div className="space-y-2 mb-4"><label className="text-sm font-medium"><HelpTip text="The target variable is the column you want the model to predict. Pick a categorical column for classification (e.g., yes/no) or a numeric column for regression (e.g., price).">Target Variable</HelpTip></label><select value={targetColumn} onChange={(e) => { setTargetColumn(e.target.value); setTrainingResult(null); setUnsupervisedResult(null); }} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="target-column-select"><option value="">-- Select Target --</option><option value="__none__">No target (Unsupervised Learning)</option>{columns.map((col, idx) => <option key={idx} value={col}>{col}</option>)}</select>
                    {suggestedTarget && <div className="mt-2 p-3 rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-950/20 flex items-start gap-2" data-testid="suggested-target">
                      <Lightbulb className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                      <div className="text-xs"><p className="font-medium text-blue-700 dark:text-blue-300">Suggested target: <strong>{suggestedTarget.name}</strong></p><p className="text-muted-foreground mt-0.5">{suggestedTarget.reason} — best for {suggestedTarget.task}</p>
                        <Button size="sm" variant="link" className="h-auto p-0 mt-1 text-blue-600 text-xs" onClick={() => { setTargetColumn(suggestedTarget.name); setTrainingResult(null); setUnsupervisedResult(null); }} data-testid="use-suggested-target-btn">Use this target</Button>
                      </div>
                    </div>}
                  </div>

                  {targetColumn && targetColumn !== '__none__' && <>
                  <div className="grid gap-6 md:grid-cols-2">
                    <div className="space-y-2"><label className="text-sm font-medium"><HelpTip text="Choose an algorithm or use 'Auto' to train all compatible algorithms and compare their performance. Auto mode is recommended for beginners.">Algorithm</HelpTip></label><select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="algorithm-select"><option value="auto">Auto (Train All & Compare)</option><optgroup label="Regression"><option value="linear">Linear Regression</option><option value="ridge">Ridge Regression</option><option value="gradient_boosting">Gradient Boosting</option></optgroup><optgroup label="Classification"><option value="logistic">Logistic Regression</option><option value="naive_bayes">Naive Bayes</option><option value="knn">KNN</option><option value="svm">SVM (Linear)</option></optgroup><optgroup label="Both"><option value="decision_tree">Decision Tree</option><option value="random_forest">Random Forest</option></optgroup></select>
                      {ALGO_DESCRIPTIONS[algorithm] && <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed flex items-start gap-1.5" data-testid="algo-description"><Lightbulb className="h-3.5 w-3.5 text-amber-500 mt-0.5 shrink-0" />{ALGO_DESCRIPTIONS[algorithm]}</p>}
                    </div>
                  </div>
                  <div className="mt-4 p-4 rounded-lg border bg-muted/30" data-testid="eval-mode-selector">
                    <label className="text-sm font-medium mb-3 block"><HelpTip text="Train/Test Split is faster — it trains on 80% and tests on 20%. Cross Validation is more reliable — it trains 5 times on different splits and averages the scores.">Evaluation Mode</HelpTip></label>
                    <div className="flex gap-6">
                      <label className="flex items-center gap-2 cursor-pointer"><input type="radio" name="evalMode" value="split" checked={evalMode === 'split'} onChange={() => setEvalMode('split')} className="accent-primary" data-testid="eval-mode-split" /><span className="text-sm">Train/Test Split <span className="text-muted-foreground">(Fast)</span></span></label>
                      <label className="flex items-center gap-2 cursor-pointer"><input type="radio" name="evalMode" value="cv" checked={evalMode === 'cv'} onChange={() => setEvalMode('cv')} className="accent-primary" data-testid="eval-mode-cv" /><span className="text-sm">5-Fold Cross Validation <span className="text-muted-foreground">(Recommended)</span></span></label>
                    </div>
                  </div>
                  {datasetScan && datasetScan.score < 70 && <div className="mt-4 p-3 rounded-lg border border-orange-400 bg-orange-50 dark:bg-orange-950/20 text-sm text-orange-700 dark:text-orange-400 flex items-center gap-2" data-testid="training-gate-warning"><AlertCircle className="h-4 w-4 shrink-0" />Dataset health score ({datasetScan.score}/100) is below the recommended threshold (70). Use the auto-clean tools in the Scanner above to improve data quality before training.</div>}
                  <Button onClick={handleTrain} disabled={isTraining || (datasetScan && datasetScan.score < 70)} className="w-full mt-6 h-12" size="lg" data-testid="start-training-btn">{isTraining ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Training...</> : <><Play className="h-4 w-4 mr-2" />Start Training</>}</Button>
                  </>}

                  {targetColumn === '__none__' && <>
                  <div className="mt-2 p-4 rounded-lg border-2 border-dashed border-primary/30 bg-primary/5">
                    <div className="flex items-center gap-3 mb-3"><Layers className="h-5 w-5 text-primary" /><div><p className="font-semibold text-sm">Unsupervised Learning Mode</p><p className="text-xs text-muted-foreground">Automatically runs K-Means, Hierarchical, DBSCAN, GMM, PCA, t-SNE, and anomaly detection</p></div></div>
                    <p className="text-xs text-muted-foreground mb-3">The system will detect the optimal number of clusters, evaluate all algorithms, and provide full visual analysis with explanations.</p>
                    <Button onClick={handleRunUnsupervised} disabled={isRunningUnsupervised} className="w-full h-12" size="lg" data-testid="run-unsupervised-btn">{isRunningUnsupervised ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Analyzing...</> : <><Sparkles className="h-4 w-4 mr-2" />Run Unsupervised Analysis</>}</Button>
                  </div>
                  </>}
                </CardContent></Card></motion.div>}

              {/* TRAINING RESULTS */}
              {trainingResult && <motion.div variants={fadeInUp} initial="initial" animate="animate" data-testid="training-results"><Card className="border-2 border-primary"><CardHeader><CardTitle className="flex items-center gap-2 text-primary"><Sparkles className="h-5 w-5" />Training Complete!</CardTitle>
                <CardDescription className="text-sm mt-2">Your <strong>{trainingResult.problemType}</strong> model was trained on {trainingResult.dataInfo?.numSamples} samples in {trainingResult.totalTime?.toFixed(2)}s. The best algorithm is <strong>{ALGO_NAMES[trainingResult.bestModel?.algorithm] || trainingResult.bestModel?.algorithm}</strong> with {trainingResult.problemType === 'regression' ? `an R\u00B2 of ${(trainingResult.bestModel.testMetrics.r2 * 100).toFixed(1)}%` : `${(trainingResult.bestModel.testMetrics.accuracy * 100).toFixed(1)}% accuracy`}.</CardDescription></CardHeader>
                <CardContent className="space-y-6">
                  {/* Summary Cards */}
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Problem Type</p><p className="text-2xl font-bold mt-1" data-testid="result-problem-type">{trainingResult.problemType}</p></CardContent></Card>
                    <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Best Model</p><p className="text-2xl font-bold mt-1" data-testid="result-best-model">{ALGO_NAMES[trainingResult.bestModel?.algorithm] || trainingResult.bestModel?.algorithm}</p></CardContent></Card>
                    <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Training Time</p><p className="text-2xl font-bold mt-1">{trainingResult.totalTime?.toFixed(2)}s</p></CardContent></Card>
                    <Card><CardContent className="p-4 text-center"><p className="text-sm text-muted-foreground">Samples</p><p className="text-2xl font-bold mt-1">{trainingResult.dataInfo?.numSamples}</p></CardContent></Card>
                  </div>

                  {/* Train-Test Split Info */}
                  <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 flex items-center gap-3" data-testid="split-info">
                    <SplitSquareVertical className="h-5 w-5 text-blue-600 shrink-0" />
                    <div><p className="text-sm font-semibold text-blue-800 dark:text-blue-200">Train/Test Split: {trainingResult.splitInfo?.trainSize} train / {trainingResult.splitInfo?.testSize} test (80/20){trainingResult.evalMode === 'cv' && ' + 5-Fold Cross Validation'}</p>
                    <p className="text-xs text-blue-600 dark:text-blue-400 mt-0.5">{trainingResult.evalMode === 'cv' ? 'Models ranked by 5-fold cross-validation score for reliable evaluation. Test metrics are also shown for reference.' : 'All metrics below are evaluated on the held-out test set for unbiased evaluation'}</p></div>
                  </div>

                  {trainingResult.dataInfo?.removedLeakageColumns?.length > 0 && <Card className="border-2 border-orange-500 bg-orange-50 dark:bg-orange-950" data-testid="leakage-warning"><CardContent className="p-4"><div className="flex items-start gap-3"><AlertCircle className="h-6 w-6 text-orange-600 mt-0.5 shrink-0" /><div><p className="font-semibold text-orange-900 dark:text-orange-100">Data Leakage Prevention</p><div className="flex flex-wrap gap-2 mt-2">{trainingResult.dataInfo.removedLeakageColumns.map((col, idx) => <Badge key={idx} variant="outline" className="bg-orange-100 dark:bg-orange-900">{col}</Badge>)}</div></div></div></CardContent></Card>}

                  {/* Test Metrics */}
                  <Card><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />How Well Did My Model Perform?</CardTitle><CardDescription>These metrics show how accurately {ALGO_NAMES[trainingResult.bestModel?.algorithm]} predicts on {trainingResult.splitInfo?.testSize} unseen test samples. Green = great, amber = okay, red = needs improvement.</CardDescription></CardHeader>
                    <CardContent>
                      {trainingResult.problemType === 'regression' ? (
                        <div className="grid gap-3 md:grid-cols-3" data-testid="test-metrics-grid">
                          <MetricCard label="R\u00B2 Score" value={`${(trainingResult.bestModel.testMetrics.r2 * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.r2} metricKey="r2" />
                          <MetricCard label="MAE" value={trainingResult.bestModel.testMetrics.mae.toFixed(2)} metricKey="mae" />
                          <MetricCard label="RMSE" value={trainingResult.bestModel.testMetrics.rmse.toFixed(2)} metricKey="rmse" />
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="grid gap-3 md:grid-cols-4" data-testid="test-metrics-grid">
                            <MetricCard label="Accuracy" value={`${(trainingResult.bestModel.testMetrics.accuracy * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.accuracy} metricKey="accuracy" />
                            <MetricCard label="Precision" value={`${(trainingResult.bestModel.testMetrics.precision * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.precision} metricKey="precision" />
                            <MetricCard label="Recall" value={`${(trainingResult.bestModel.testMetrics.recall * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.recall} metricKey="recall" />
                            <MetricCard label="F1 Score" value={`${(trainingResult.bestModel.testMetrics.f1 * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.f1} metricKey="f1" />
                          </div>

                          {/* Confusion Matrix Heatmap */}
                          {trainingResult.bestModel.testMetrics.confusionMatrix && (
                            <div data-testid="confusion-matrix">
                              <p className="text-sm font-medium mb-2 flex items-center gap-2"><Target className="h-4 w-4" />Confusion Matrix</p>
                              <p className="text-xs text-muted-foreground mb-4">Green cells on the diagonal = correct predictions. Red cells = misclassifications. A perfect model would only have green diagonal cells.</p>
                              <div className="overflow-auto"><table className="text-sm border-collapse">
                                <thead><tr><td className="p-2"></td><td className="p-2 text-xs text-center text-muted-foreground font-medium" colSpan={trainingResult.bestModel.testMetrics.confusionMatrix.classes.length}>Predicted</td></tr>
                                <tr><td className="p-2 text-xs text-muted-foreground font-medium">Actual</td>{trainingResult.bestModel.testMetrics.confusionMatrix.classes.map((cls, i) => <td key={i} className="p-2 text-center font-mono text-xs font-medium min-w-[60px]">{trainingResult.bestModel.testMetrics.confusionMatrix.classes.length <= 2 && trainingResult.problemType === 'classification' && models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[cls] : cls}</td>)}</tr></thead>
                                <tbody>{trainingResult.bestModel.testMetrics.confusionMatrix.classes.map((cls, i) => {
                                  const maxVal = Math.max(...trainingResult.bestModel.testMetrics.confusionMatrix.matrix.flat());
                                  return (
                                  <tr key={i}><td className="p-2 font-mono text-xs font-medium">{trainingResult.bestModel.testMetrics.confusionMatrix.classes.length <= 2 && trainingResult.problemType === 'classification' && models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[cls] : cls}</td>
                                  {trainingResult.bestModel.testMetrics.confusionMatrix.matrix[i].map((val, j) => {
                                    const intensity = maxVal > 0 ? val / maxVal : 0;
                                    const bg = i === j
                                      ? `rgba(34, 197, 94, ${0.15 + intensity * 0.55})`
                                      : val > 0 ? `rgba(239, 68, 68, ${0.1 + intensity * 0.5})` : undefined;
                                    return <td key={j} className="p-2 text-center font-mono min-w-[60px] rounded font-bold" style={{ backgroundColor: bg }}>{val}</td>;
                                  })}</tr>);
                                })}</tbody>
                              </table></div>
                            </div>
                          )}

                          {/* Per-Class Metrics */}
                          {trainingResult.bestModel.testMetrics.perClassMetrics && (
                            <div data-testid="per-class-metrics">
                              <p className="text-sm font-medium mb-3">Per-Class Metrics</p>
                              <div className="rounded-md border"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Class</th><th className="p-2 text-right font-medium"><MetricTip metricKey="precision">Precision</MetricTip></th><th className="p-2 text-right font-medium"><MetricTip metricKey="recall">Recall</MetricTip></th><th className="p-2 text-right font-medium"><MetricTip metricKey="f1">F1 Score</MetricTip></th></tr></thead>
                              <tbody>{trainingResult.bestModel.testMetrics.perClassMetrics.map((pc, i) => <tr key={i} className="border-b last:border-0"><td className="p-2 font-mono text-xs">{models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[pc.class] : pc.class}</td><td className="p-2 text-right">{(pc.precision * 100).toFixed(1)}%</td><td className="p-2 text-right">{(pc.recall * 100).toFixed(1)}%</td><td className="p-2 text-right font-medium">{(pc.f1 * 100).toFixed(1)}%</td></tr>)}</tbody></table></div>
                            </div>
                          )}
                        </div>
                      )}
                    </CardContent></Card>

                  {/* Train vs Test Comparison */}
                  <Card><CardHeader><CardTitle className="text-lg flex items-center gap-2"><SplitSquareVertical className="h-4 w-4" />Train vs Test Comparison</CardTitle><CardDescription>If training scores are much higher than test scores, the model may be overfitting (memorizing data instead of learning patterns).</CardDescription></CardHeader>
                    <CardContent>
                      <div className="rounded-md border" data-testid="train-test-comparison"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Metric</th><th className="p-3 text-right font-medium">Train</th><th className="p-3 text-right font-medium">Test</th><th className="p-3 text-right font-medium">Gap</th></tr></thead>
                      <tbody>{Object.keys(trainingResult.bestModel.testMetrics).filter(k => typeof trainingResult.bestModel.testMetrics[k] === 'number').map(k => {
                        const isPercent = k === 'accuracy' || k === 'precision' || k === 'recall' || k === 'r2' || k === 'f1';
                        const trainVal = trainingResult.bestModel.trainMetrics[k] || 0;
                        const testVal = trainingResult.bestModel.testMetrics[k] || 0;
                        const gap = isPercent ? ((trainVal - testVal) * 100) : (trainVal - testVal);
                        const isOverfit = isPercent ? gap > 15 : false;
                        return (
                        <tr key={k} className={`border-b last:border-0 ${isOverfit ? 'bg-red-50 dark:bg-red-950/10' : ''}`}><td className="p-3 text-xs uppercase font-medium"><MetricTip metricKey={k}>{k.replace(/_/g, ' ')}</MetricTip></td>
                        <td className="p-3 text-right font-mono">{isPercent ? `${(trainVal * 100).toFixed(2)}%` : trainVal?.toFixed(2)}</td>
                        <td className="p-3 text-right font-mono font-bold">{isPercent ? `${(testVal * 100).toFixed(2)}%` : testVal?.toFixed(2)}</td>
                        <td className={`p-3 text-right font-mono text-xs ${isOverfit ? 'text-red-600 font-semibold' : 'text-muted-foreground'}`}>{isPercent ? `${gap > 0 ? '+' : ''}${gap.toFixed(1)}pp` : `${gap > 0 ? '+' : ''}${gap.toFixed(2)}`}{isOverfit && ' (overfit)'}</td></tr>);
                      })}</tbody></table></div>
                    </CardContent></Card>

                  {/* Regression Visualizations */}
                  {trainingResult.problemType === 'regression' && trainingResult.predictionsVsActual && (<>
                    <Card><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Target className="h-4 w-4" />Actual vs Predicted</CardTitle><CardDescription>Points on the diagonal line represent perfect predictions. Spread away from the line indicates prediction error.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="actual" name="Actual" type="number" tick={{fontSize: 11}} /><YAxis dataKey="predicted" name="Predicted" type="number" tick={{fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip cursor={{strokeDasharray: '3 3'}} /><ReferenceLine segment={[{ x: arrayMin(trainingResult.predictionsVsActual.actual), y: arrayMin(trainingResult.predictionsVsActual.actual) }, { x: arrayMax(trainingResult.predictionsVsActual.actual), y: arrayMax(trainingResult.predictionsVsActual.actual) }]} stroke="#22c55e" strokeDasharray="5 5" strokeWidth={2} /><Scatter name="Predictions" data={trainingResult.predictionsVsActual.actual.map((a, i) => ({ actual: a, predicted: trainingResult.predictionsVsActual.predicted[i] }))} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer></CardContent></Card>
                    <Card><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Activity className="h-4 w-4" />Residual Analysis</CardTitle><CardDescription>Points near the zero line mean accurate predictions. Patterns may reveal systematic errors.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="predicted" name="Predicted" type="number" tick={{fontSize: 11}} /><YAxis dataKey="residual" name="Residual" type="number" tick={{fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip cursor={{strokeDasharray: '3 3'}} /><ReferenceLine y={0} stroke="#22c55e" strokeDasharray="5 5" strokeWidth={2} /><Scatter name="Residuals" data={trainingResult.predictionsVsActual.predicted.map((p, i) => ({ predicted: p, residual: trainingResult.predictionsVsActual.actual[i] - p }))} fill="hsl(var(--chart-2))" /></ScatterChart></ResponsiveContainer></CardContent></Card>
                  </>)}

                  {trainingResult.bestModel?.featureImportance?.length > 0 && <Card data-testid="feature-importance-chart"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><TrendingUp className="h-4 w-4" />Which Features Matter Most?</CardTitle><CardDescription>Features ranked by their influence on predictions. Taller bars = more impact on the model's decisions.</CardDescription></CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.bestModel.featureImportance}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} tick={{fontSize: 11}} /><YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} /><Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} /><Bar dataKey="importance" radius={[6, 6, 0, 0]}>{trainingResult.bestModel.featureImportance.map((_, i) => <Cell key={i} fill={`hsl(${210 + i * 15}, 70%, ${45 + i * 3}%)`} />)}</Bar></BarChart></ResponsiveContainer></CardContent></Card>}

                  {/* Model Comparison Chart */}
                  {trainingResult.leaderboard?.length > 1 && <Card data-testid="model-comparison-chart"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />Model Comparison</CardTitle><CardDescription>Side-by-side comparison of all algorithms. The best model is automatically selected for predictions.</CardDescription></CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map(m => ({
                      name: ALGO_NAMES[m.algorithm] || m.algorithm,
                      score: trainingResult.problemType === 'regression' ? +(m.testMetrics.r2 * 100).toFixed(2) : +(m.testMetrics.accuracy * 100).toFixed(2),
                      fill: ALGO_COLORS[m.algorithm] || '#6b7280'
                    }))}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" tick={{fontSize: 11}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="8 4" label={{ value: '50% baseline', position: 'right', fontSize: 10, fill: '#94a3b8' }} /><Bar dataKey="score" radius={[6, 6, 0, 0]}>{trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
                  </CardContent></Card>}

                  {/* Cross-Validation Performance Chart */}
                  {trainingResult.evalMode === 'cv' && trainingResult.leaderboard?.some(m => m.cvScore !== null) && (
                    <Card data-testid="cv-performance-chart"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Activity className="h-4 w-4" />Cross-Validation Performance</CardTitle><CardDescription>Average performance across 5 data folds. Higher CV scores indicate more reliable, consistent models.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.cvScore !== null && m.algorithm !== 'baseline').map(m => ({
                        name: ALGO_NAMES[m.algorithm] || m.algorithm,
                        cvScore: +(m.cvScore * 100).toFixed(2),
                        testScore: +(trainingResult.problemType === 'regression' ? (m.testMetrics.r2 * 100) : (m.testMetrics.accuracy * 100)).toFixed(2),
                        fill: ALGO_COLORS[m.algorithm] || '#6b7280'
                      }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><Legend /><Bar dataKey="cvScore" name="CV Score" radius={[4, 4, 0, 0]}>{trainingResult.leaderboard.filter(m => m.cvScore !== null && m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar><Bar dataKey="testScore" name="Test Score" radius={[4, 4, 0, 0]} fill="#94a3b8" opacity={0.5} /></BarChart></ResponsiveContainer>
                    </CardContent></Card>
                  )}

                  {/* Algorithm Leaderboard */}
                  <Card><CardHeader><CardTitle className="text-lg">Algorithm Leaderboard</CardTitle><CardDescription>All algorithms ranked by {trainingResult.evalMode === 'cv' ? 'cross-validation score' : 'test performance'} — the best model is automatically selected</CardDescription></CardHeader>
                    <CardContent><div className="space-y-2" data-testid="leaderboard">{trainingResult.leaderboard?.map((model, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 rounded-lg border" data-testid={`leaderboard-entry-${idx}`} style={idx === 0 ? { borderColor: ALGO_COLORS[model.algorithm], borderWidth: 2 } : {}}>
                        <div className="flex items-center gap-3"><Badge variant={idx === 0 ? 'default' : 'secondary'} style={idx === 0 ? { backgroundColor: ALGO_COLORS[model.algorithm] } : {}}>{idx + 1}</Badge><div><p className="font-medium">{ALGO_NAMES[model.algorithm] || model.algorithm}</p><p className="text-xs text-muted-foreground">{model.durationSec ? `${model.durationSec.toFixed(3)}s` : '-'}{idx === 0 && ' — Best Model'}</p></div></div>
                        <div className="text-right font-mono text-sm" data-testid={`leaderboard-score-${idx}`}>
                          {(() => { const m = model.testMetrics; const s = m.accuracy !== undefined ? m.accuracy : (m.r2 !== undefined ? m.r2 : 0); const c = getScoreColor(s); return <div className={`font-semibold ${c.text}`}>{m.accuracy !== undefined ? <MetricTip metricKey="accuracy" value={m.accuracy}>{`${(m.accuracy * 100).toFixed(2)}% acc`}</MetricTip> : m.r2 !== undefined ? <MetricTip metricKey="r2" value={m.r2}>{`${(m.r2 * 100).toFixed(2)}% R\u00B2`}</MetricTip> : '-'}</div>; })()}
                          {model.cvScore !== null && model.cvScore !== undefined && <div className="text-xs text-emerald-600 font-semibold" data-testid={`cv-score-${idx}`}><MetricTip metricKey="cvScore" value={model.cvScore}>CV: {(model.cvScore * 100).toFixed(2)}%</MetricTip></div>}
                        </div>
                      </div>
                    ))}</div></CardContent></Card>
                </CardContent></Card></motion.div>}
            </motion.div>
          )}

          {/* ==================== UNSUPERVISED SUMMARY (brief) ==================== */}
          {activeView === 'analysis' && unsupervisedResult && (
            <motion.div variants={fadeInUp} initial="initial" animate="animate" data-testid="unsupervised-summary">
              <Card className="border-2 border-primary"><CardContent className="p-6"><div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center"><Sparkles className="h-6 w-6 text-primary" /></div>
                  <div><p className="font-semibold">Unsupervised Analysis Complete</p>
                    <p className="text-sm text-muted-foreground">Best: {unsupervisedResult.bestAlgorithm?.name} (<MetricTip metricKey="silhouette" value={unsupervisedResult.bestAlgorithm?.metrics.silhouette}>Silhouette: {unsupervisedResult.bestAlgorithm?.metrics.silhouette.toFixed(3)}</MetricTip>) &middot; {unsupervisedResult.optimalK.bestK} clusters &middot; {unsupervisedResult.totalTime.toFixed(2)}s</p>
                  </div>
                </div>
                <Button onClick={() => { setActiveView('predict'); setPredictTab('results'); }} data-testid="view-unsupervised-results-btn"><Eye className="h-4 w-4 mr-2" />View Full Results</Button>
              </div></CardContent></Card>
            </motion.div>
          )}

          {/* ==================== PREDICTIONS & ANALYSIS WORKSPACE ==================== */}
          {activeView === 'predict' && (
            <motion.div key="predict" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="predict-view">

              {/* Sub-tab Navigation */}
              <div className="flex gap-1 p-1 rounded-lg bg-muted/50 w-fit" data-testid="predict-tabs">
                {[{ id: 'predict', label: 'Predict', icon: Sparkles }, { id: 'batch', label: 'Batch', icon: FileUp }, { id: 'results', label: 'Results', icon: Eye }, { id: 'visualize', label: 'Visualizations', icon: BarChart3 }, { id: 'correlation', label: 'Correlation', icon: TrendingUp }].map(tab => (
                  <Button key={tab.id} variant={predictTab === tab.id ? 'default' : 'ghost'} size="sm" onClick={() => setPredictTab(tab.id)} data-testid={`predict-tab-${tab.id}`} className="gap-1.5"><tab.icon className="h-3.5 w-3.5" />{tab.label}</Button>
                ))}
              </div>

              {/* ===== PREDICT TAB ===== */}
              {predictTab === 'predict' && (<div className="space-y-6">
                {models.length === 0 && !unsupervisedResult ? (
                  <Card className="border-2 border-dashed" data-testid="no-model-warning"><CardContent className="py-16 text-center">
                    <AlertCircle className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                    <h3 className="text-lg font-semibold mb-2" data-testid="no-model-warning-title">No Models Available</h3>
                    <p className="text-muted-foreground mb-6 text-sm" data-testid="no-model-warning-message">Train a supervised model or run unsupervised analysis in the Analysis tab first.</p>
                    <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="go-to-train-btn"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
                  </CardContent></Card>
                ) : (<>
                  {/* Model Selector (supervised) */}
                  {models.length > 0 && <Card data-testid="model-selector-card"><CardHeader><CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5" />Select Model</CardTitle><CardDescription>Choose which trained model to use for predictions</CardDescription></CardHeader>
                    <CardContent><select value={selectedModelIdx === -1 ? models.length - 1 : selectedModelIdx} onChange={e => { setSelectedModelIdx(Number(e.target.value)); setPredictionResult(null); setPredictionFormData({}); }} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="model-selector">
                      {models.map((m, i) => <option key={m.modelId} value={i}>{ALGO_NAMES[m.algorithm] || m.algorithm} — {m.problemType} — Target: {m.targetColumn}</option>)}
                    </select></CardContent></Card>}

                  {/* Supervised Prediction Form */}
                  {models.length > 0 && (() => { const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : models.length - 1; const am = models[idx]; return (
                    <Card><CardHeader><CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Enter Feature Values</CardTitle><CardDescription>Enter new data values similar to your training dataset. The model will generate a prediction based on the patterns it learned.</CardDescription></CardHeader>
                      <CardContent className="space-y-6">
                        {am.modelData.numericCols.length > 0 && <div><p className="text-sm font-medium mb-3 flex items-center gap-2"><TrendingUp className="h-4 w-4 text-blue-500" />Numeric Features</p><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{am.modelData.numericCols.map(col => <div key={col} className="space-y-1.5"><label className="text-sm font-medium text-muted-foreground">{col}</label><input type="number" step="any" value={predictionFormData[col] ?? ''} onChange={e => setPredictionFormData(prev => ({ ...prev, [col]: e.target.value }))} placeholder={`Enter ${col}`} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid={`predict-input-${col}`} /></div>)}</div></div>}
                        {am.modelData.categoricalCols.length > 0 && <div><p className="text-sm font-medium mb-3 flex items-center gap-2"><Layers className="h-4 w-4 text-emerald-500" />Categorical Features</p><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{am.modelData.categoricalCols.map(col => <div key={col} className="space-y-1.5"><label className="text-sm font-medium text-muted-foreground">{col}</label><select value={predictionFormData[col] ?? ''} onChange={e => setPredictionFormData(prev => ({ ...prev, [col]: e.target.value }))} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid={`predict-input-${col}`}><option value="">-- Select --</option>{am.modelData.encodingMap[col]?.map(val => <option key={val} value={val}>{val}</option>)}</select></div>)}</div></div>}
                        <Button onClick={handlePredict} className="w-full h-12" size="lg" data-testid="generate-predictions-btn"><Sparkles className="h-4 w-4 mr-2" />Generate Prediction</Button>
                      </CardContent></Card>
                  ); })()}

                  {/* Supervised Result */}
                  {predictionResult && <motion.div variants={fadeInUp} initial="initial" animate="animate"><Card className="border-2 border-primary" data-testid="prediction-results"><CardHeader><CardTitle className="flex items-center gap-2 text-primary"><Eye className="h-5 w-5" />Prediction Result</CardTitle><CardDescription>Generated using {ALGO_NAMES[predictionResult.algorithm] || predictionResult.algorithm}</CardDescription></CardHeader>
                    <CardContent><div className="space-y-4">{predictionResult.predictions.map((pred, idx) => {
                      const isClassification = predictionResult.problemType === 'classification';
                      return (<div key={idx} className="rounded-xl border-2 border-primary/20 bg-primary/5 p-6" data-testid={`prediction-result-${idx}`}>
                        <div className="flex items-center justify-between"><div><p className="text-sm font-medium text-muted-foreground mb-1">{isClassification ? 'Predicted Class' : 'Predicted Value'}</p><p className="text-4xl font-bold text-primary" data-testid="prediction-value">{typeof pred === 'number' ? (isClassification ? pred : pred.toFixed(4)) : pred}</p></div><div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center"><Target className="h-8 w-8 text-primary" /></div></div>
                        {predictionResult.probabilities?.[idx] && <div className="mt-4 pt-4 border-t"><p className="text-xs font-medium text-muted-foreground mb-2">Class Probabilities</p><div className="space-y-2">{predictionResult.probabilities[idx].map((p, ci) => <div key={ci} className="flex items-center gap-3"><span className="text-xs font-mono w-16">Class {ci}</span><div className="flex-1 h-3 bg-muted rounded-full overflow-hidden"><div className="h-full bg-primary rounded-full transition-all" style={{ width: `${(p * 100).toFixed(1)}%` }} /></div><span className="text-xs font-mono w-14 text-right">{(p * 100).toFixed(1)}%</span></div>)}</div></div>}
                        {predictionResult.inputData && <div className="mt-4 pt-4 border-t"><p className="text-xs font-medium text-muted-foreground mb-2">Input Summary</p><div className="flex flex-wrap gap-2">{Object.entries(predictionResult.inputData).filter(([,v]) => v !== '' && v !== 0).map(([k, v]) => <Badge key={k} variant="secondary" className="text-xs">{k}: {v}</Badge>)}</div></div>}
                      </div>);
                    })}</div></CardContent></Card></motion.div>}

                  {/* Unsupervised Cluster Prediction */}
                  {unsupervisedResult && <Card data-testid="cluster-prediction"><CardHeader><CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Predict Cluster</CardTitle><CardDescription>Assign a new data point to the nearest cluster using {unsupervisedResult.bestAlgorithm?.name}</CardDescription></CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{unsupervisedResult.preprocessing.featureNames.map(col => <div key={col} className="space-y-1.5"><label className="text-sm font-medium text-muted-foreground">{col}</label><input type="number" step="any" value={clusterPredFormData[col] ?? ''} onChange={e => setClusterPredFormData(prev => ({ ...prev, [col]: e.target.value }))} placeholder={`Enter ${col}`} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid={`cluster-pred-input-${col}`} /></div>)}</div>
                      <Button onClick={handleClusterPredict} className="w-full h-12" size="lg" data-testid="predict-cluster-btn"><Target className="h-4 w-4 mr-2" />Predict Cluster</Button>
                      {clusterPredResult && <div className="rounded-xl border-2 p-6 mt-4" style={{borderColor: CLUSTER_COLORS[clusterPredResult.cluster % CLUSTER_COLORS.length], backgroundColor: CLUSTER_COLORS[clusterPredResult.cluster % CLUSTER_COLORS.length] + '10'}} data-testid="cluster-pred-result">
                        <div className="flex items-center justify-between mb-4"><div><p className="text-sm font-medium text-muted-foreground mb-1">Assigned Cluster</p><p className="text-4xl font-bold" style={{color: CLUSTER_COLORS[clusterPredResult.cluster % CLUSTER_COLORS.length]}}>Cluster {clusterPredResult.cluster}</p></div><div className="h-16 w-16 rounded-full flex items-center justify-center" style={{backgroundColor: CLUSTER_COLORS[clusterPredResult.cluster % CLUSTER_COLORS.length] + '20'}}><Target className="h-8 w-8" style={{color: CLUSTER_COLORS[clusterPredResult.cluster % CLUSTER_COLORS.length]}} /></div></div>
                        <p className="text-sm text-muted-foreground mb-2">Distance to centroid: {clusterPredResult.distance.toFixed(4)}</p>
                        {clusterPredResult.interpretation && <p className="text-sm">{clusterPredResult.interpretation.interpretation}</p>}
                        {clusterPredResult.inputData && <div className="mt-3 pt-3 border-t"><p className="text-xs font-medium text-muted-foreground mb-2">Input Summary</p><div className="flex flex-wrap gap-2">{Object.entries(clusterPredResult.inputData).filter(([,v]) => v !== '').map(([k, v]) => <Badge key={k} variant="secondary" className="text-xs">{k}: {v}</Badge>)}</div></div>}
                      </div>}
                    </CardContent></Card>}
                </>)}
              </div>)}

              {/* ===== BATCH TAB ===== */}
              {predictTab === 'batch' && (<div className="space-y-6">
                {models.length === 0 ? (
                  <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
                    <AlertCircle className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                    <h3 className="text-lg font-semibold mb-2">No Models Available</h3>
                    <p className="text-muted-foreground text-sm mb-4">Train a model first to use batch predictions.</p>
                    <Button onClick={() => setActiveView('analysis')} size="lg"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
                  </CardContent></Card>
                ) : (<>
                  <Card data-testid="batch-model-selector"><CardHeader><CardTitle className="flex items-center gap-2"><Cpu className="h-5 w-5" />Select Model for Batch</CardTitle></CardHeader>
                    <CardContent><select value={selectedModelIdx === -1 ? models.length - 1 : selectedModelIdx} onChange={e => { setSelectedModelIdx(Number(e.target.value)); setBatchResults(null); }} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="batch-model-select">
                      {models.map((m, i) => <option key={m.modelId} value={i}>{ALGO_NAMES[m.algorithm] || m.algorithm} — {m.problemType} — Target: {m.targetColumn}</option>)}
                    </select></CardContent></Card>

                  <Card data-testid="batch-upload-card"><CardHeader><CardTitle className="flex items-center gap-2"><FileUp className="h-5 w-5" />Upload CSV for Batch Prediction</CardTitle><CardDescription>Upload a CSV file with the same features as your training data. Predictions will be generated for every row.</CardDescription></CardHeader>
                    <CardContent className="space-y-4">
                      <div className="relative border-2 border-dashed rounded-lg p-8 text-center transition-all border-muted-foreground/25 hover:border-primary hover:bg-accent/50">
                        <input type="file" accept=".csv" onChange={handleBatchCsvUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="batch-csv-file-input" />
                        <FileUp className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
                        <p className="text-sm font-medium">Drop CSV file or click to browse</p>
                      </div>
                      <Separator />
                      <div><label className="text-sm font-medium mb-2 block">Or paste CSV data:</label>
                        <textarea value={batchCsvText} onChange={e => setBatchCsvText(e.target.value)} placeholder="Paste CSV data for batch predictions..." rows={5} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring" data-testid="batch-csv-text-input" />
                      </div>
                      <Button onClick={handleBatchPredict} disabled={batchProcessing || !batchCsvText.trim()} className="w-full h-12" size="lg" data-testid="run-batch-predict-btn">
                        {batchProcessing ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Processing...</> : <><Play className="h-4 w-4 mr-2" />Run Batch Predictions</>}
                      </Button>
                    </CardContent></Card>

                  {batchResults && <Card data-testid="batch-results-card"><CardHeader>
                    <div className="flex items-center justify-between">
                      <div><CardTitle className="flex items-center gap-2"><CheckCircle2 className="h-5 w-5 text-emerald-600" />Batch Results</CardTitle>
                        <CardDescription>{batchResults.predictions.length} predictions using {ALGO_NAMES[batchResults.algorithm] || batchResults.algorithm}</CardDescription></div>
                      <Button variant="outline" size="sm" onClick={downloadBatchCsv} data-testid="download-batch-csv-btn"><Download className="h-4 w-4 mr-2" />Download CSV</Button>
                    </div>
                  </CardHeader><CardContent>
                    <div className="rounded-md border overflow-auto max-h-96">
                      <table className="w-full text-sm" data-testid="batch-results-table">
                        <thead><tr className="border-b bg-muted/50 sticky top-0">
                          <th className="p-2 text-left font-medium text-xs">#</th>
                          {batchResults.headers.slice(0, 6).map(h => <th key={h} className="p-2 text-left font-medium text-xs font-mono">{h}</th>)}
                          {batchResults.headers.length > 6 && <th className="p-2 text-center text-xs">...</th>}
                          <th className="p-2 text-right font-medium text-xs bg-primary/10">Prediction</th>
                        </tr></thead>
                        <tbody>{batchResults.rows.slice(0, 100).map((row, ri) => (
                          <tr key={ri} className="border-b last:border-0 hover:bg-accent/50" data-testid={`batch-row-${ri}`}>
                            <td className="p-2 text-xs text-muted-foreground">{ri + 1}</td>
                            {batchResults.headers.slice(0, 6).map(h => <td key={h} className="p-2 text-xs font-mono">{String(row[h] ?? '').substring(0, 15)}</td>)}
                            {batchResults.headers.length > 6 && <td className="p-2 text-center text-xs">...</td>}
                            <td className="p-2 text-right font-mono font-bold text-primary" data-testid={`batch-pred-${ri}`}>
                              {typeof batchResults.predictions[ri] === 'number' ? (batchResults.problemType === 'classification' ? batchResults.predictions[ri] : batchResults.predictions[ri].toFixed(4)) : batchResults.predictions[ri]}
                            </td>
                          </tr>
                        ))}</tbody>
                      </table>
                    </div>
                    {batchResults.rows.length > 100 && <p className="text-xs text-muted-foreground mt-2 text-center">Showing first 100 of {batchResults.rows.length} rows</p>}
                  </CardContent></Card>}
                </>)}
              </div>)}

              {/* ===== RESULTS TAB ===== */}
              {predictTab === 'results' && (<div className="space-y-6">

                {/* Prediction History */}
                {predictionHistory.length > 0 && <Card data-testid="prediction-history"><CardHeader><CardTitle className="flex items-center gap-2"><Clock className="h-5 w-5" />Prediction History</CardTitle><CardDescription>{predictionHistory.length} predictions made this session</CardDescription></CardHeader>
                  <CardContent><div className="rounded-md border overflow-auto"><table className="w-full text-sm">
                    <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">#</th><th className="p-3 text-left font-medium">Type</th><th className="p-3 text-left font-medium">Model</th><th className="p-3 text-left font-medium">Prediction</th><th className="p-3 text-left font-medium">Input</th><th className="p-3 text-right font-medium">Time</th></tr></thead>
                    <tbody>{[...predictionHistory].reverse().map((h, idx) => (
                      <tr key={h.id} className="border-b last:border-0" data-testid={`history-row-${idx}`}>
                        <td className="p-3 text-muted-foreground">{predictionHistory.length - idx}</td>
                        <td className="p-3"><Badge variant={h.type === 'supervised' ? 'default' : 'secondary'}>{h.type === 'supervised' ? 'Supervised' : 'Cluster'}</Badge></td>
                        <td className="p-3 font-medium">{h.model}</td>
                        <td className="p-3 font-mono font-bold text-primary">{typeof h.prediction === 'number' ? h.prediction.toFixed(4) : h.prediction}</td>
                        <td className="p-3 text-xs"><div className="flex flex-wrap gap-1">{Object.entries(h.input).filter(([,v]) => v !== '').slice(0, 3).map(([k, v]) => <Badge key={k} variant="outline" className="text-xs">{k}:{typeof v === 'number' ? Number(v).toFixed(1) : v}</Badge>)}{Object.keys(h.input).length > 3 && <span className="text-muted-foreground">+{Object.keys(h.input).length - 3}</span>}</div></td>
                        <td className="p-3 text-right text-xs text-muted-foreground">{new Date(h.timestamp).toLocaleTimeString()}</td>
                      </tr>
                    ))}</tbody>
                  </table></div></CardContent></Card>}

                {/* Unsupervised Results */}
                {unsupervisedResult && (<>
                  <Card data-testid="unsupervised-full-results"><CardHeader>
                    <CardTitle className="flex items-center gap-2 text-primary"><Sparkles className="h-5 w-5" />Clustering Results</CardTitle>
                    <CardDescription>Analyzed {unsupervisedResult.preprocessing.n} samples with {unsupervisedResult.preprocessing.p} features. Best: {unsupervisedResult.bestAlgorithm?.name} ({unsupervisedResult.optimalK.bestK} clusters).</CardDescription>
                  </CardHeader><CardContent className="space-y-6">

                    {/* Preprocessing Summary */}
                    <div data-testid="preprocessing-summary"><p className="text-sm font-semibold mb-3 flex items-center gap-2"><Database className="h-4 w-4" />Preprocessing</p>
                      <div className="grid gap-3 md:grid-cols-4">
                        <div className="rounded-xl p-3 border bg-muted/50"><p className="text-xs text-muted-foreground">Dataset Size</p><p className="text-lg font-bold">{unsupervisedResult.preprocessing.n} rows</p></div>
                        <div className="rounded-xl p-3 border bg-muted/50"><p className="text-xs text-muted-foreground">Numeric Features</p><p className="text-lg font-bold">{unsupervisedResult.preprocessing.p}</p></div>
                        <div className="rounded-xl p-3 border bg-muted/50"><p className="text-xs text-muted-foreground">Missing Filled</p><p className="text-lg font-bold">{unsupervisedResult.preprocessing.missingFilled}</p></div>
                        <div className="rounded-xl p-3 border bg-muted/50"><p className="text-xs text-muted-foreground">Scaling</p><p className="text-lg font-bold text-sm">{unsupervisedResult.preprocessing.scalingApplied}</p></div>
                      </div>
                    </div>

                    {/* Algorithm Leaderboard */}
                    <div data-testid="algorithm-leaderboard"><p className="text-sm font-semibold mb-3 flex items-center gap-2"><Trophy className="h-4 w-4" />Algorithm Leaderboard</p>
                      <div className="rounded-md border overflow-auto"><table className="w-full text-sm">
                        <thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Rank</th><th className="p-3 text-left font-medium">Algorithm</th><th className="p-3 text-right font-medium">Clusters</th><th className="p-3 text-right font-medium"><MetricTip metricKey="silhouette">Silhouette</MetricTip></th><th className="p-3 text-right font-medium"><MetricTip metricKey="daviesBouldin">Davies-Bouldin</MetricTip></th><th className="p-3 text-right font-medium"><MetricTip metricKey="calinskiHarabasz">Calinski-Harabasz</MetricTip></th><th className="p-3 text-right font-medium">Runtime</th></tr></thead>
                        <tbody>{unsupervisedResult.algorithms.map((algo, idx) => (
                          <tr key={algo.key} className={`border-b last:border-0 ${idx === 0 ? 'bg-primary/5' : ''}`} data-testid={`leaderboard-row-${algo.key}`}>
                            <td className="p-3">{idx === 0 ? <Badge variant="default" className="gap-1"><Trophy className="h-3 w-3" />Best</Badge> : <span className="text-muted-foreground">#{idx + 1}</span>}</td>
                            <td className="p-3 font-semibold">{algo.name}{algo.noiseCount > 0 && <span className="text-xs text-muted-foreground ml-1">({algo.noiseCount} noise)</span>}</td>
                            <td className="p-3 text-right font-mono">{algo.k}</td>
                            <td className={`p-3 text-right font-mono font-bold ${algo.metrics.silhouette >= 0.5 ? 'text-emerald-600' : algo.metrics.silhouette >= 0.25 ? 'text-amber-600' : 'text-red-600'}`}>{algo.metrics.silhouette.toFixed(3)}</td>
                            <td className="p-3 text-right font-mono">{algo.metrics.daviesBouldin === Infinity ? '-' : algo.metrics.daviesBouldin.toFixed(3)}</td>
                            <td className="p-3 text-right font-mono">{algo.metrics.calinskiHarabasz.toFixed(1)}</td>
                            <td className="p-3 text-right font-mono text-xs">{algo.runtime.toFixed(3)}s</td>
                          </tr>
                        ))}</tbody>
                      </table></div>
                    </div>

                    {/* Best Model Metrics */}
                    {unsupervisedResult.bestAlgorithm && <div data-testid="best-model-summary"><p className="text-sm font-semibold mb-3 flex items-center gap-2"><Cpu className="h-4 w-4" />Best Model Metrics</p>
                      <div className="grid gap-3 md:grid-cols-4">
                        <MetricCard label="Silhouette" value={unsupervisedResult.bestAlgorithm.metrics.silhouette.toFixed(3)} score={unsupervisedResult.bestAlgorithm.metrics.silhouette} metricKey="silhouette" />
                        <MetricCard label="Davies-Bouldin" value={unsupervisedResult.bestAlgorithm.metrics.daviesBouldin === Infinity ? '-' : unsupervisedResult.bestAlgorithm.metrics.daviesBouldin.toFixed(3)} metricKey="daviesBouldin" />
                        <MetricCard label="Calinski-Harabasz" value={unsupervisedResult.bestAlgorithm.metrics.calinskiHarabasz.toFixed(1)} metricKey="calinskiHarabasz" />
                        <MetricCard label="Clusters" value={unsupervisedResult.bestAlgorithm.k} />
                      </div>
                    </div>}

                    {/* Cluster Insights */}
                    {unsupervisedResult.interpretation && <div data-testid="cluster-insights"><p className="text-sm font-semibold mb-3 flex items-center gap-2"><Info className="h-4 w-4" />Cluster Insights</p>
                      <div className="grid gap-4 md:grid-cols-2">{unsupervisedResult.interpretation.interpretations.map(ci => (
                        <Card key={ci.clusterId} className="border-l-4" style={{borderLeftColor: CLUSTER_COLORS[ci.clusterId % CLUSTER_COLORS.length]}} data-testid={`cluster-insight-${ci.clusterId}`}><CardContent className="p-4">
                          <div className="flex items-center justify-between mb-2"><Badge style={{backgroundColor: CLUSTER_COLORS[ci.clusterId % CLUSTER_COLORS.length]}}>Cluster {ci.clusterId}</Badge><span className="text-sm font-mono">{ci.size} points</span></div>
                          <p className="text-sm text-muted-foreground mb-3">{ci.interpretation}</p>
                          {ci.keyFeatures.length > 0 && <div className="space-y-1"><p className="text-xs font-semibold">Key Features:</p>{ci.keyFeatures.filter(f => Math.abs(f.deviation) > 5).map(f => <div key={f.feature} className="flex items-center gap-2 text-xs"><span className={`inline-block w-2 h-2 rounded-full ${f.direction === 'higher' ? 'bg-emerald-500' : 'bg-red-500'}`} /><span className="font-medium">{f.feature}:</span><span className="text-muted-foreground">{f.clusterAvg.toFixed(1)} ({f.direction}, {Math.abs(f.deviation).toFixed(0)}%)</span></div>)}</div>}
                        </CardContent></Card>
                      ))}</div>
                    </div>}

                    {/* Terminology */}
                    <div data-testid="terminology-section"><p className="text-sm font-semibold mb-3 flex items-center gap-2"><Info className="h-4 w-4" />ML Terminology Guide</p>
                      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">{UNSUPERVISED_TERMS.map(term => <div key={term.key} className="rounded-lg border p-3 bg-muted/30 hover:bg-muted/50 transition-colors" data-testid={`term-${term.key}`}><p className="text-sm font-semibold mb-1">{term.name}</p><p className="text-xs text-muted-foreground leading-relaxed">{term.desc}</p></div>)}</div>
                    </div>

                  </CardContent></Card>
                </>)}

                {/* Supervised Training Summary */}
                {trainingResult && !unsupervisedResult && <Card data-testid="training-summary"><CardHeader>
                  <CardTitle className="flex items-center gap-2"><Sparkles className="h-5 w-5" />Latest Training Results</CardTitle>
                  <CardDescription>Best model: {ALGO_NAMES[trainingResult.bestModel?.algorithm]} ({trainingResult.problemType})</CardDescription>
                </CardHeader><CardContent>
                  <div className="grid gap-3 md:grid-cols-4">
                    {trainingResult.problemType === 'regression' ? (<>
                      <MetricCard label="R\u00B2" value={`${(trainingResult.bestModel.testMetrics.r2 * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.r2} metricKey="r2" />
                      <MetricCard label="MAE" value={trainingResult.bestModel.testMetrics.mae.toFixed(2)} metricKey="mae" />
                    </>) : (<>
                      <MetricCard label="Accuracy" value={`${(trainingResult.bestModel.testMetrics.accuracy * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.accuracy} metricKey="accuracy" />
                      <MetricCard label="F1" value={`${(trainingResult.bestModel.testMetrics.f1 * 100).toFixed(2)}%`} score={trainingResult.bestModel.testMetrics.f1} metricKey="f1" />
                    </>)}
                  </div>
                </CardContent></Card>}

                {!unsupervisedResult && !trainingResult && predictionHistory.length === 0 && <Card className="border-2 border-dashed"><CardContent className="py-16 text-center"><Eye className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" /><h3 className="text-lg font-semibold mb-2">No Results Yet</h3><p className="text-muted-foreground text-sm">Train a model or run unsupervised analysis to see results here.</p></CardContent></Card>}
              </div>)}

              {/* ===== VISUALIZATIONS TAB ===== */}
              {predictTab === 'visualize' && (<div className="space-y-6">
                {!trainingResult && !unsupervisedResult ? (
                  <Card className="border-2 border-dashed"><CardContent className="py-16 text-center"><BarChart3 className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" /><h3 className="text-lg font-semibold mb-2">No Visualizations Available</h3><p className="text-muted-foreground text-sm">Train a model or run unsupervised analysis to generate charts.</p></CardContent></Card>
                ) : (<>
                  {/* Supervised Visualizations */}
                  {trainingResult && (<>
                    {trainingResult.predictionsVsActual && <Card data-testid="viz-actual-predicted"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Target className="h-4 w-4" />Actual vs Predicted</CardTitle><CardDescription>Points near the diagonal line indicate accurate predictions.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="actual" name="Actual" type="number" tick={{fontSize: 11}} /><YAxis dataKey="predicted" name="Predicted" type="number" tick={{fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip cursor={{strokeDasharray: '3 3'}} /><ReferenceLine segment={[{ x: arrayMin(trainingResult.predictionsVsActual.actual), y: arrayMin(trainingResult.predictionsVsActual.actual) }, { x: arrayMax(trainingResult.predictionsVsActual.actual), y: arrayMax(trainingResult.predictionsVsActual.actual) }]} stroke="#22c55e" strokeDasharray="5 5" strokeWidth={2} /><Scatter name="Predictions" data={trainingResult.predictionsVsActual.actual.map((a, i) => ({ actual: a, predicted: trainingResult.predictionsVsActual.predicted[i] }))} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer>
                    </CardContent></Card>}

                    {trainingResult.bestModel?.featureImportance?.length > 0 && <Card data-testid="viz-feature-importance"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><TrendingUp className="h-4 w-4" />Feature Importance</CardTitle><CardDescription>Features ranked by their influence on predictions.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.bestModel.featureImportance}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} tick={{fontSize: 11}} /><YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} /><Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} /><Bar dataKey="importance" radius={[6, 6, 0, 0]}>{trainingResult.bestModel.featureImportance.map((_, i) => <Cell key={i} fill={`hsl(${210 + i * 15}, 70%, ${45 + i * 3}%)`} />)}</Bar></BarChart></ResponsiveContainer>
                    </CardContent></Card>}

                    {trainingResult.leaderboard?.length > 1 && <Card data-testid="viz-model-comparison"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />Prediction Distribution by Model</CardTitle><CardDescription>Performance comparison across all trained algorithms.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map(m => ({ name: ALGO_NAMES[m.algorithm] || m.algorithm, score: trainingResult.problemType === 'regression' ? +(m.testMetrics.r2 * 100).toFixed(2) : +(m.testMetrics.accuracy * 100).toFixed(2), fill: ALGO_COLORS[m.algorithm] || '#6b7280' }))}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" tick={{fontSize: 11}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="8 4" /><Bar dataKey="score" radius={[6, 6, 0, 0]}>{trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
                    </CardContent></Card>}
                  </>)}

                  {/* Unsupervised Visualizations */}
                  {unsupervisedResult && (<>
                    <Card data-testid="viz-pca-scatter"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Target className="h-4 w-4" />Cluster Scatter Plot (PCA 2D)</CardTitle><CardDescription>Data reduced to 2D using PCA. Colors represent clusters from {unsupervisedResult.bestAlgorithm?.name}.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={350}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="x" name="PC1" type="number" tick={{fontSize: 11}} label={{value: `PC1 (${(unsupervisedResult.pca.explainedVariance[0] * 100).toFixed(1)}%)`, position: 'bottom', fontSize: 11}} /><YAxis dataKey="y" name="PC2" type="number" tick={{fontSize: 11}} label={{value: `PC2 (${(unsupervisedResult.pca.explainedVariance[1] * 100).toFixed(1)}%)`, angle: -90, position: 'left', fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip content={({active, payload}) => active && payload?.length ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p>Cluster: {payload[0].payload.cluster}</p><p>PC1: {payload[0].payload.x.toFixed(3)}</p><p>PC2: {payload[0].payload.y.toFixed(3)}</p></div> : null} /><Legend />{[...new Set(unsupervisedResult.pca.points.map(p => p.cluster))].sort((a,b)=>a-b).map(c => <Scatter key={c} name={`Cluster ${c}`} data={unsupervisedResult.pca.points.filter(p => p.cluster === c)} fill={CLUSTER_COLORS[c % CLUSTER_COLORS.length]} />)}</ScatterChart></ResponsiveContainer>
                    </CardContent></Card>

                    {unsupervisedResult.bestAlgorithm && <Card data-testid="viz-cluster-distribution"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />Cluster Distribution</CardTitle><CardDescription>Points per cluster from {unsupervisedResult.bestAlgorithm.name}.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={250}><BarChart data={(() => { const c = {}; unsupervisedResult.bestAlgorithm.labels.forEach(l => { if (l >= 0) c[l] = (c[l] || 0) + 1; }); return Object.entries(c).map(([k, v]) => ({ name: `Cluster ${k}`, count: v })); })()}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" tick={{fontSize: 11}} /><YAxis tick={{fontSize: 11}} /><Tooltip /><Bar dataKey="count" radius={[6, 6, 0, 0]}>{(() => { const c = {}; unsupervisedResult.bestAlgorithm.labels.forEach(l => { if (l >= 0) c[l] = (c[l] || 0) + 1; }); return Object.keys(c).map((k, i) => <Cell key={i} fill={CLUSTER_COLORS[Number(k) % CLUSTER_COLORS.length]} />); })()}</Bar></BarChart></ResponsiveContainer>
                    </CardContent></Card>}

                    <div className="grid gap-6 md:grid-cols-2">
                      <Card data-testid="viz-elbow"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><TrendingUp className="h-4 w-4" />Elbow Method</CardTitle><CardDescription>The "elbow" suggests optimal K.</CardDescription></CardHeader><CardContent>
                        <ResponsiveContainer width="100%" height={250}><LineChart data={unsupervisedResult.optimalK.results}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="k" tick={{fontSize: 11}} /><YAxis tick={{fontSize: 11}} /><Tooltip /><Line type="monotone" dataKey="inertia" stroke="#2563eb" strokeWidth={2} dot={{fill: '#2563eb', r: 4}} /><ReferenceLine x={unsupervisedResult.optimalK.bestK} stroke="#22c55e" strokeDasharray="5 5" label={{value: `K=${unsupervisedResult.optimalK.bestK}`, position: 'top', fontSize: 10, fill: '#22c55e'}} /></LineChart></ResponsiveContainer>
                      </CardContent></Card>
                      <Card data-testid="viz-silhouette"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Activity className="h-4 w-4" />Silhouette by K</CardTitle><CardDescription>Higher = better clusters.</CardDescription></CardHeader><CardContent>
                        <ResponsiveContainer width="100%" height={250}><LineChart data={unsupervisedResult.optimalK.results}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="k" tick={{fontSize: 11}} /><YAxis tick={{fontSize: 11}} domain={[0, 1]} /><Tooltip /><Line type="monotone" dataKey="silhouette" stroke="#16a34a" strokeWidth={2} dot={{fill: '#16a34a', r: 4}} /><ReferenceLine x={unsupervisedResult.optimalK.bestK} stroke="#22c55e" strokeDasharray="5 5" /></LineChart></ResponsiveContainer>
                      </CardContent></Card>
                    </div>

                    {unsupervisedResult.anomalyDetection?.isolationForest && <Card data-testid="viz-anomaly"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><ShieldAlert className="h-4 w-4" />Anomaly Detection</CardTitle><CardDescription>Isolation Forest: {unsupervisedResult.anomalyDetection.isolationForest.nAnomalies} anomalies ({((unsupervisedResult.anomalyDetection.isolationForest.nAnomalies / unsupervisedResult.preprocessing.n) * 100).toFixed(1)}%)</CardDescription></CardHeader><CardContent>
                      {unsupervisedResult.anomalyDetection.points && <ResponsiveContainer width="100%" height={300}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="x" name="PC1" type="number" tick={{fontSize: 11}} /><YAxis dataKey="y" name="PC2" type="number" tick={{fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip content={({active, payload}) => active && payload?.length ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p>{payload[0].payload.anomaly ? 'ANOMALY' : 'Normal'}</p><p>Score: {payload[0].payload.score?.toFixed(3)}</p></div> : null} /><Legend /><Scatter name="Normal" data={unsupervisedResult.anomalyDetection.points.filter(p => !p.anomaly)} fill="#2563eb" /><Scatter name="Anomaly" data={unsupervisedResult.anomalyDetection.points.filter(p => p.anomaly)} fill="#dc2626" /></ScatterChart></ResponsiveContainer>}
                    </CardContent></Card>}

                    {unsupervisedResult.interpretation && <Card data-testid="viz-cluster-profile"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />Cluster Profiles</CardTitle><CardDescription>Feature averages per cluster vs overall.</CardDescription></CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={300}><BarChart data={unsupervisedResult.preprocessing.featureNames.map((fname, fi) => { const entry = { feature: fname }; unsupervisedResult.interpretation.interpretations.forEach(ci => { entry[`Cluster ${ci.clusterId}`] = ci.featureAverages[fi]?.value || 0; }); entry['Overall'] = unsupervisedResult.interpretation.overallAvg[fi]?.value || 0; return entry; })}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="feature" angle={-45} textAnchor="end" height={80} tick={{fontSize: 10}} /><YAxis tick={{fontSize: 11}} /><Tooltip /><Legend />{unsupervisedResult.interpretation.interpretations.map(ci => <Bar key={ci.clusterId} dataKey={`Cluster ${ci.clusterId}`} fill={CLUSTER_COLORS[ci.clusterId % CLUSTER_COLORS.length]} radius={[4, 4, 0, 0]} />)}<Bar dataKey="Overall" fill="#6b7280" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer>
                    </CardContent></Card>}
                  </>)}
                </>)}
              </div>)}

              {/* ===== CORRELATION TAB ===== */}
              {predictTab === 'correlation' && (<div className="space-y-6">
                {!dataProfile ? (
                  <Card className="border-2 border-dashed"><CardContent className="py-16 text-center"><TrendingUp className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" /><h3 className="text-lg font-semibold mb-2">No Data Loaded</h3><p className="text-muted-foreground text-sm">Upload a dataset in the Analysis tab to explore correlations.</p></CardContent></Card>
                ) : (<>
                  {/* Variable Selector */}
                  <Card data-testid="correlation-scatter"><CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" />Correlation Explorer</CardTitle><CardDescription>Select two numeric variables to visualize their relationship.</CardDescription></CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-1.5"><label className="text-sm font-medium">X Variable</label><select value={corrVarX} onChange={e => setCorrVarX(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="corr-var-x"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                        <div className="space-y-1.5"><label className="text-sm font-medium">Y Variable</label><select value={corrVarY} onChange={e => setCorrVarY(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="corr-var-y"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                      </div>
                      {corrVarX && corrVarY && (() => {
                        const pairs = dataProfile.rows.filter(r => typeof r[corrVarX] === 'number' && typeof r[corrVarY] === 'number').map(r => ({ x: r[corrVarX], y: r[corrVarY] }));
                        const n = pairs.length; if (n < 3) return <p className="text-sm text-muted-foreground">Insufficient numeric data for these variables.</p>;
                        const mx = pairs.reduce((s, p) => s + p.x, 0) / n, my = pairs.reduce((s, p) => s + p.y, 0) / n;
                        let sx = 0, sy = 0, sxy = 0; pairs.forEach(p => { sx += (p.x - mx) ** 2; sy += (p.y - my) ** 2; sxy += (p.x - mx) * (p.y - my); });
                        const r = (sx > 0 && sy > 0) ? sxy / Math.sqrt(sx * sy) : 0;
                        const abs = Math.abs(r); const dir = r >= 0 ? 'Positive' : 'Negative';
                        const str = abs >= 0.8 ? 'Strong' : abs >= 0.5 ? 'Moderate' : abs >= 0.3 ? 'Weak' : 'Very Weak';
                        return (<>
                          <div className={`p-4 rounded-lg border-2 ${abs >= 0.5 ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20' : abs >= 0.3 ? 'border-amber-300 bg-amber-50 dark:bg-amber-950/20' : 'border-muted bg-muted/30'}`} data-testid="corr-result">
                            <div className="flex items-center justify-between"><div><p className="text-sm font-medium">Correlation Coefficient</p><p className="text-3xl font-bold">{r.toFixed(4)}</p></div><Badge variant={abs >= 0.5 ? 'default' : 'secondary'}>{str} {dir}</Badge></div>
                          </div>
                          <ResponsiveContainer width="100%" height={350}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="x" name={corrVarX} type="number" tick={{fontSize: 11}} label={{value: corrVarX, position: 'bottom', fontSize: 11}} /><YAxis dataKey="y" name={corrVarY} type="number" tick={{fontSize: 11}} label={{value: corrVarY, angle: -90, position: 'left', fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip /><Scatter name="Data" data={pairs} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer>
                        </>);
                      })()}
                    </CardContent></Card>

                  {/* Correlation Heatmap */}
                  {corrMatrix.length > 0 && <Card data-testid="correlation-heatmap"><CardHeader><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Correlation Heatmap</CardTitle><CardDescription>Pairwise correlations between all numeric variables. Blue = positive, red = negative.</CardDescription></CardHeader>
                    <CardContent><div className="overflow-auto">
                      <div className="inline-grid gap-px" style={{gridTemplateColumns: `120px repeat(${dataProfile.numericColumns.length}, minmax(60px, 1fr))`}}>
                        <div />
                        {dataProfile.numericColumns.map(col => <div key={col} className="text-xs font-mono p-2 text-center truncate" title={col}>{col}</div>)}
                        {corrMatrix.map((row, ri) => (<React.Fragment key={ri}>
                          <div className="text-xs font-mono p-2 text-right truncate" title={row.feature}>{row.feature}</div>
                          {dataProfile.numericColumns.map((col, ci) => { const val = row[col] || 0; const bg = val > 0 ? `rgba(37, 99, 235, ${Math.abs(val) * 0.8})` : `rgba(220, 38, 38, ${Math.abs(val) * 0.8})`; return <div key={ci} className="p-2 text-center text-xs font-mono rounded-sm border border-muted/30" style={{backgroundColor: bg, color: Math.abs(val) > 0.4 ? 'white' : 'inherit'}} title={`${row.feature} vs ${col}: ${val.toFixed(3)}`}>{val.toFixed(2)}</div>; })}
                        </React.Fragment>))}
                      </div>
                    </div></CardContent></Card>}
                </>)}
              </div>)}

            </motion.div>
          )}

          {/* ==================== EXPLAINABILITY ==================== */}
          {activeView === 'explainability' && (
            <motion.div key="explainability" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="explainability-view">
              {(models.length === 0 && !unsupervisedResult) ? (
                <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
                  <Eye className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                  <h3 className="text-lg font-semibold mb-2">No Model to Explain</h3>
                  <p className="text-muted-foreground text-sm mb-4">Train a model first in the Analysis tab.</p>
                  <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="xai-go-analysis"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
                </CardContent></Card>
              ) : (<>
                {/* Tab selector */}
                <motion.div variants={fadeInUp}>
                  <div className="flex gap-1 p-1 rounded-xl bg-muted/50 w-fit border border-border/50" data-testid="xai-tabs">
                    {[{ id: 'shap', label: 'SHAP Analysis' }, { id: 'lime', label: 'LIME Explanation' }, { id: 'insights', label: 'Feature Insights' }, ...(unsupervisedResult ? [{ id: 'clusters', label: 'Cluster Explanation' }] : [])].map(tab => (
                      <Button key={tab.id} variant={xaiTab === tab.id ? 'default' : 'ghost'} size="sm" onClick={() => setXaiTab(tab.id)} data-testid={`xai-tab-${tab.id}`}
                        className={xaiTab === tab.id ? 'shadow-md' : ''}>{tab.label}</Button>
                    ))}
                  </div>
                </motion.div>

                {/* Controls */}
                <motion.div variants={fadeInUp}><Card data-testid="xai-controls" className="border-border/60 shadow-sm"><CardHeader>
                  <CardTitle className="flex items-center gap-2"><Eye className="h-5 w-5 text-violet-500" />Configuration</CardTitle>
                  <CardDescription>Select a model and a data record to generate explanations. SHAP measures global and local feature importance. LIME builds a local linear model to explain individual predictions.</CardDescription>
                </CardHeader><CardContent className="space-y-4">
                  {models.length > 0 && xaiTab !== 'clusters' && <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-1.5"><label className="text-sm font-medium"><HelpTip text="Select the trained model you want to explain. If you trained multiple algorithms, you can compare their explanations.">Model</HelpTip></label>
                      <select value={selectedModelIdx === -1 ? models.length - 1 : selectedModelIdx} onChange={e => setSelectedModelIdx(Number(e.target.value))} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="xai-model-select">
                        {models.map((m, i) => <option key={m.modelId} value={i}>{ALGO_NAMES[m.algorithm] || m.algorithm} — {m.problemType}</option>)}
                      </select></div>
                    <div className="space-y-1.5"><label className="text-sm font-medium"><HelpTip text="Select a row (data point) from your dataset to understand why the model made that specific prediction. Try different rows to see how explanations change.">Record (row) to explain</HelpTip></label>
                      <input type="number" min={0} max={dataProfile ? dataProfile.rowCount - 1 : 0} value={xaiRow} onChange={e => setXaiRow(Math.max(0, Number(e.target.value)))} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="xai-row-select" />
                      {smartRowSuggestions.length > 0 && <div className="mt-2 flex flex-wrap gap-1.5" data-testid="smart-row-suggestions">
                        {smartRowSuggestions.map((s, i) => (
                          <button key={i} onClick={() => setXaiRow(s.idx)} className="group/sr inline-flex items-center gap-1 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[10px] font-medium hover:bg-blue-50 dark:hover:bg-blue-950/30 hover:border-blue-300 dark:hover:border-blue-700 transition-colors" data-testid={`smart-row-${i}`} title={s.desc}>
                            <Lightbulb className="h-3 w-3 text-amber-400 group-hover/sr:text-amber-500" />{s.label}
                          </button>
                        ))}
                      </div>}
                    </div>
                  </div>}
                  <div className="flex flex-wrap gap-3">
                    {xaiTab === 'shap' && models.length > 0 && <Button onClick={handleRunSHAP} disabled={xaiLoading} size="lg" className="h-11 bg-gradient-to-r from-violet-600 to-pink-500 hover:from-violet-700 hover:to-pink-600 text-white shadow-md" data-testid="run-shap-btn">
                      {xaiLoading ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Computing...</> : <><Sparkles className="h-4 w-4 mr-2" />Run SHAP Analysis</>}
                    </Button>}
                    {xaiTab === 'lime' && models.length > 0 && <Button onClick={handleRunLIME} disabled={xaiLoading} size="lg" className="h-11 bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-700 hover:to-teal-600 text-white shadow-md" data-testid="run-lime-btn">
                      {xaiLoading ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Computing...</> : <><Target className="h-4 w-4 mr-2" />Explain with LIME</>}
                    </Button>}
                    {xaiTab === 'clusters' && unsupervisedResult && <Button onClick={handleClusterExplain} disabled={xaiLoading} size="lg" className="h-11 bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-700 hover:to-cyan-600 text-white shadow-md" data-testid="run-cluster-explain-btn">
                      {xaiLoading ? <><div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />Computing...</> : <><Layers className="h-4 w-4 mr-2" />Explain Clusters</>}
                    </Button>}
                    {xaiTab !== 'clusters' && models.length > 0 && <Button variant="outline" onClick={handleExplainPrediction} disabled={xaiLoading} size="lg" className="h-11 border-violet-300 dark:border-violet-700 hover:bg-violet-50 dark:hover:bg-violet-950/30" data-testid="explain-prediction-btn">
                      <Eye className="h-4 w-4 mr-2" />Explain This Prediction (SHAP + LIME)
                    </Button>}
                  </div>
                  {xaiTab === 'shap' && !shapGlobal && !xaiLoading && <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1.5" data-testid="shap-help-text"><Lightbulb className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />SHAP (SHapley Additive exPlanations) shows how each feature contributes to the model's prediction. It reveals both global patterns across the dataset and local explanations for individual records.</p>}
                  {xaiTab === 'lime' && !limeResult && !xaiLoading && <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1.5" data-testid="lime-help-text"><Lightbulb className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by building a simpler local model around a single data point. It shows which features most influenced that specific prediction.</p>}
                  {xaiTab === 'clusters' && !clusterShap && !xaiLoading && unsupervisedResult && <p className="text-xs text-muted-foreground mt-2 flex items-start gap-1.5" data-testid="cluster-help-text"><Lightbulb className="h-3.5 w-3.5 text-amber-400 mt-0.5 shrink-0" />Cluster explanation uses SHAP to show which features are most important in determining how data points are grouped into clusters. This helps you understand the characteristics that define each cluster.</p>}
                </CardContent></Card></motion.div>

                {/* ───── SHAP TAB ───── */}
                {xaiTab === 'shap' && (<>
                  {/* Section: Global Explainability */}
                  {shapGlobal && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-2"><div className="h-px flex-1 bg-gradient-to-r from-violet-500/50 to-transparent" /><span className="text-sm font-semibold text-violet-600 dark:text-violet-400 uppercase tracking-wider">Global Explainability</span><div className="h-px flex-1 bg-gradient-to-l from-violet-500/50 to-transparent" /></div>
                  </motion.div>}

                  {/* Global Feature Importance */}
                  {shapGlobal && <motion.div variants={fadeInUp}><Card data-testid="shap-global-importance" className="border-violet-200/50 dark:border-violet-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-violet-500" /><MetricTip metricKey="shapValue">Global Feature Importance (SHAP)</MetricTip></CardTitle>
                    <CardDescription>This chart ranks features by their average absolute SHAP value across the entire dataset. Features at the top have the strongest overall influence on predictions. Higher values indicate features the model relies on most heavily.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(200, shapGlobal.importance.length * 40)}>
                      <BarChart layout="vertical" data={shapGlobal.importance} margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                        <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Mean |SHAP Value|', position: 'bottom', fontSize: 11 }} />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                        <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0].payload.feature}</p><p>Mean |SHAP|: <strong className="text-violet-600">{payload[0].payload.importance.toFixed(4)}</strong></p><p className="text-muted-foreground mt-1">Rank #{shapGlobal.importance.indexOf(payload[0].payload) + 1} of {shapGlobal.importance.length}</p></div> : null} />
                        <Bar dataKey="importance" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                          {shapGlobal.importance.map((_, i) => <Cell key={i} fill={`hsl(${270 - (i / shapGlobal.importance.length) * 60}, 75%, ${50 + i * 2}%)`} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* SHAP Summary Plot (positive/negative) */}
                  {shapSummary && <motion.div variants={fadeInUp}><Card data-testid="shap-summary-plot" className="border-pink-200/50 dark:border-pink-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-pink-500" />SHAP Summary Plot</CardTitle>
                    <CardDescription>This plot shows the average direction of each feature's impact. Pink bars show the average positive push (increasing prediction), while cyan bars show the average negative pull (decreasing prediction). Together they reveal whether a feature typically pushes predictions up or down.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(200, shapSummary.length * 40)}>
                      <BarChart layout="vertical" data={shapSummary} margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                        <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Mean SHAP Value', position: 'bottom', fontSize: 11 }} />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                        <Tooltip content={({ active, payload }) => active && payload?.length ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0]?.payload?.feature}</p><p className="text-pink-500">Positive push: +{payload[0]?.payload?.positive?.toFixed(4)}</p><p className="text-cyan-500">Negative pull: {payload[0]?.payload?.negative?.toFixed(4)}</p></div> : null} />
                        <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" />
                        <Bar dataKey="positive" stackId="a" fill="#ec4899" radius={[0, 4, 4, 0]} name="Positive push" isAnimationActive={false} />
                        <Bar dataKey="negative" stackId="a" fill="#06b6d4" radius={[4, 0, 0, 4]} name="Negative pull" isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                    <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#ec4899' }} /> Increases prediction</span>
                      <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#06b6d4' }} /> Decreases prediction</span>
                    </div>
                  </CardContent></Card></motion.div>}

                  {/* Beeswarm Plot */}
                  {shapBeeswarm && <motion.div variants={fadeInUp}><Card data-testid="shap-beeswarm" className="border-purple-200/50 dark:border-purple-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5 text-purple-500" />SHAP Beeswarm Plot</CardTitle>
                    <CardDescription>Each dot represents one data point for one feature. Horizontal position shows the SHAP value (impact on prediction). Color encodes the original feature value — blue means low, violet is medium, and pink means high. Clusters of dots reveal patterns: for example, a streak of pink dots on the right means high feature values push predictions up.</CardDescription>
                  </CardHeader><CardContent>
                    {(() => {
                      const fn = getXaiModel()?.modelData?.featureNames || [];
                      const chartData = shapBeeswarm.points.map(p => ({ x: p.shapValue, y: p.featureIdx + p.jitter, fill: p.color }));
                      return (<>
                        <ResponsiveContainer width="100%" height={Math.max(250, fn.length * 44)}>
                          <ScatterChart margin={{ left: 20, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis type="number" dataKey="x" name="SHAP Value" tick={{ fontSize: 11 }} label={{ value: 'SHAP Value (impact on prediction)', position: 'bottom', fontSize: 11 }} />
                            <YAxis type="number" dataKey="y" domain={[-0.5, fn.length - 0.5]} ticks={fn.map((_, i) => i)} tickFormatter={v => fn[Math.round(v)] || ''} tick={{ fontSize: 11 }} width={120} />
                            <ZAxis range={[24, 24]} />
                            <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{fn[Math.round(payload[0].payload.y)]}</p><p>SHAP: <strong>{payload[0].payload.x.toFixed(4)}</strong></p></div> : null} />
                            <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" strokeWidth={1.5} />
                            <Scatter data={chartData} isAnimationActive={false}>
                              {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                            </Scatter>
                          </ScatterChart>
                        </ResponsiveContainer>
                        <div className="flex items-center justify-center gap-3 mt-4 py-2 px-4 rounded-lg bg-muted/30">
                          <span className="text-xs font-medium text-blue-500">Low</span>
                          <div className="h-4 w-44 rounded-full shadow-inner" style={{ background: 'linear-gradient(to right, #3b82f6, #7c3aed, #ec4899)' }} />
                          <span className="text-xs font-medium text-pink-500">High</span>
                          <span className="text-xs text-muted-foreground ml-1">(Feature Value)</span>
                        </div>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>}

                  {/* Section: Local Explainability */}
                  {shapLocal && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-pink-500/50 to-transparent" /><span className="text-sm font-semibold text-pink-600 dark:text-pink-400 uppercase tracking-wider">Local Explainability — Record #{xaiRow}</span><div className="h-px flex-1 bg-gradient-to-l from-pink-500/50 to-transparent" /></div>
                  </motion.div>}

                  {/* Instance Details Panel */}
                  {shapLocal && <motion.div variants={fadeInUp}><Card data-testid="shap-instance-panel" className="border-pink-200/50 dark:border-pink-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Table2 className="h-5 w-5 text-pink-500" />Instance Feature Values — Record #{xaiRow}</CardTitle>
                    <CardDescription>This table shows the actual feature values for the selected record, along with each feature's SHAP contribution. It helps you understand why this specific prediction was made by linking raw data values to their model impact.</CardDescription>
                  </CardHeader><CardContent>
                    <div className="rounded-lg border overflow-auto max-h-64">
                      <table className="w-full text-sm">
                        <thead><tr className="border-b bg-gradient-to-r from-pink-50 to-violet-50 dark:from-pink-950/20 dark:to-violet-950/20 sticky top-0">
                          <th className="p-3 text-left font-medium">Feature</th>
                          <th className="p-3 text-right font-medium">Value</th>
                          <th className="p-3 text-right font-medium">SHAP Contribution</th>
                          <th className="p-3 text-left font-medium w-32">Impact</th>
                        </tr></thead>
                        <tbody>{shapLocal.featureNames.map((f, i) => {
                          const val = shapLocal.shapValues[i];
                          const maxAbs = Math.max(...shapLocal.shapValues.map(Math.abs)) || 1;
                          const pct = (Math.abs(val) / maxAbs) * 100;
                          return <tr key={i} className="border-b last:border-0 hover:bg-accent/30 transition-colors">
                            <td className="p-3 font-mono text-xs">{f}</td>
                            <td className="p-3 text-right text-xs font-medium">{shapLocal.instance[i]?.toFixed(3)}</td>
                            <td className="p-3 text-right"><span className={`text-xs font-bold ${val >= 0 ? 'text-pink-600' : 'text-cyan-600'}`}>{val >= 0 ? '+' : ''}{val.toFixed(4)}</span></td>
                            <td className="p-3"><div className="w-full h-2.5 bg-muted rounded-full overflow-hidden"><div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: val >= 0 ? '#ec4899' : '#06b6d4' }} /></div></td>
                          </tr>;
                        })}</tbody>
                      </table>
                    </div>
                    <div className="flex items-center gap-6 mt-3 text-sm">
                      <span className="text-muted-foreground"><MetricTip metricKey="baseValue">Base value:</MetricTip> <strong>{shapLocal.basePred.toFixed(4)}</strong></span>
                      <span>Prediction: <strong className="text-primary text-base">{shapLocal.instancePred.toFixed(4)}</strong></span>
                    </div>
                  </CardContent></Card></motion.div>}

                  {/* Waterfall Plot */}
                  {shapLocal && <motion.div variants={fadeInUp}><Card data-testid="shap-waterfall" className="border-rose-200/50 dark:border-rose-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5 text-rose-500" />Waterfall Plot — Record #{xaiRow}</CardTitle>
                    <CardDescription>This chart breaks down the prediction step-by-step. Starting from the base value (average prediction), each bar shows how a single feature pushes the prediction up (pink) or down (cyan). The cumulative effect of all features leads to the final prediction.</CardDescription>
                  </CardHeader><CardContent>
                    {(() => {
                      const wf = buildWaterfallData(shapLocal.shapValues, shapLocal.featureNames, shapLocal.basePred);
                      return (<>
                        <ResponsiveContainer width="100%" height={Math.max(200, wf.length * 36)}>
                          <BarChart layout="vertical" data={wf} margin={{ left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis type="number" tick={{ fontSize: 11 }} />
                            <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                            <Tooltip content={({ active, payload }) => active && payload?.[1] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[1].payload.feature}</p><p>Contribution: <strong style={{ color: payload[1].payload.contribution >= 0 ? '#ec4899' : '#06b6d4' }}>{payload[1].payload.contribution >= 0 ? '+' : ''}{payload[1].payload.contribution.toFixed(4)}</strong></p><p className="text-muted-foreground">Running total: {(payload[1].payload.offset + payload[1].payload.width).toFixed(4)}</p></div> : null} />
                            <Bar dataKey="offset" stackId="a" fill="transparent" isAnimationActive={false} />
                            <Bar dataKey="width" stackId="a" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                              {wf.map((d, i) => <Cell key={i} fill={d.contribution >= 0 ? '#ec4899' : '#06b6d4'} />)}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                        <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#ec4899' }} /> Increases prediction</span>
                          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#06b6d4' }} /> Decreases prediction</span>
                        </div>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>}

                  {/* Force Plot */}
                  {shapLocal && <motion.div variants={fadeInUp}><Card data-testid="shap-force" className="border-amber-200/50 dark:border-amber-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Sparkles className="h-5 w-5 text-amber-500" />Force Plot — Record #{xaiRow}</CardTitle>
                    <CardDescription>This interactive visualization shows how features collectively push the prediction from the base value to the final output. Hover over each colored segment to see which feature it represents and how much it contributes. Wider segments have stronger influence.</CardDescription>
                  </CardHeader><CardContent>
                    {(() => {
                      const force = buildForceData(shapLocal.shapValues, shapLocal.featureNames, shapLocal.basePred, shapLocal.instancePred);
                      const maxMag = Math.max(force.totalPos, force.totalNeg) || 1;
                      return (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between text-sm font-medium px-1 py-2 rounded-lg bg-muted/30">
                            <span><MetricTip metricKey="baseValue">Base: {force.basePred.toFixed(3)}</MetricTip></span>
                            <span className="flex items-center gap-2"><ChevronRight className="h-4 w-4 text-muted-foreground" /><span className="text-primary font-bold text-base">Output: {force.instancePred.toFixed(3)}</span></span>
                          </div>
                          {force.positive.length > 0 && <div>
                            <p className="text-xs font-semibold mb-2 text-pink-600 dark:text-pink-400 flex items-center gap-1.5"><TrendingUp className="h-3.5 w-3.5" />Positive contributions (push prediction higher)</p>
                            <div className="flex gap-0.5 h-10 rounded-lg overflow-hidden shadow-inner">{force.positive.map((f, i) => {
                              const colors = ['#ec4899', '#f472b6', '#db2777', '#be185d', '#f9a8d4'];
                              return <div key={i} className="group/fp relative h-full rounded-sm transition-all hover:brightness-110 hover:scale-y-110" style={{ backgroundColor: colors[i % colors.length], flex: `${(f.value / maxMag) * 100}` }} data-testid={`force-pos-${i}`}>
                                <span className="invisible group-hover/fp:visible absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap bg-popover border shadow-lg rounded-lg px-3 py-2 text-xs"><strong>{f.feature}</strong>: <span className="text-pink-600">+{f.value.toFixed(4)}</span></span>
                              </div>;
                            })}</div>
                          </div>}
                          {force.negative.length > 0 && <div>
                            <p className="text-xs font-semibold mb-2 text-cyan-600 dark:text-cyan-400 flex items-center gap-1.5"><TrendingUp className="h-3.5 w-3.5 rotate-180" />Negative contributions (push prediction lower)</p>
                            <div className="flex gap-0.5 h-10 rounded-lg overflow-hidden shadow-inner">{force.negative.map((f, i) => {
                              const colors = ['#06b6d4', '#22d3ee', '#0891b2', '#0e7490', '#67e8f9'];
                              return <div key={i} className="group/fn relative h-full rounded-sm transition-all hover:brightness-110 hover:scale-y-110" style={{ backgroundColor: colors[i % colors.length], flex: `${(Math.abs(f.value) / maxMag) * 100}` }} data-testid={`force-neg-${i}`}>
                                <span className="invisible group-hover/fn:visible absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1 whitespace-nowrap bg-popover border shadow-lg rounded-lg px-3 py-2 text-xs"><strong>{f.feature}</strong>: <span className="text-cyan-600">{f.value.toFixed(4)}</span></span>
                              </div>;
                            })}</div>
                          </div>}
                          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 mt-3">{shapLocal.featureNames.map((f, i) => (
                            <div key={i} className={`rounded-lg border p-2.5 text-xs transition-colors hover:shadow-sm ${shapLocal.shapValues[i] >= 0 ? 'border-pink-200/60 dark:border-pink-800/30 hover:bg-pink-50/50 dark:hover:bg-pink-950/10' : 'border-cyan-200/60 dark:border-cyan-800/30 hover:bg-cyan-50/50 dark:hover:bg-cyan-950/10'}`} data-testid={`force-feat-${i}`}>
                              <span className="text-muted-foreground">{f}</span>
                              <span className={`block font-bold ${shapLocal.shapValues[i] >= 0 ? 'text-pink-600' : 'text-cyan-600'}`}>{shapLocal.shapValues[i] >= 0 ? '+' : ''}{shapLocal.shapValues[i].toFixed(4)}</span>
                            </div>
                          ))}</div>
                        </div>
                      );
                    })()}
                  </CardContent></Card></motion.div>}

                  {/* Section: Feature Analysis */}
                  {shapDependence && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-indigo-500/50 to-transparent" /><span className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">Feature Analysis</span><div className="h-px flex-1 bg-gradient-to-l from-indigo-500/50 to-transparent" /></div>
                  </motion.div>}

                  {/* Dependence Plot */}
                  {shapDependence && <motion.div variants={fadeInUp}><Card data-testid="shap-dependence" className="border-indigo-200/50 dark:border-indigo-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5 text-indigo-500" />SHAP Dependence Plot</CardTitle>
                    <CardDescription>This scatter plot reveals the relationship between a feature's actual value and its SHAP contribution. Use the selector to explore different features. Trends in the dots indicate whether increasing a feature's value consistently increases or decreases the model's prediction.</CardDescription>
                  </CardHeader><CardContent className="space-y-4">
                    <select value={xaiDepFeature} onChange={e => { setXaiDepFeature(Number(e.target.value)); }} className="rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="xai-dep-feature-select">
                      {(getXaiModel()?.modelData?.featureNames || []).map((f, i) => <option key={i} value={i}>{f}</option>)}
                    </select>
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart margin={{ bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis type="number" dataKey="featureValue" name="Feature Value" tick={{ fontSize: 11 }} label={{ value: (getXaiModel()?.modelData?.featureNames || [])[xaiDepFeature] || 'Feature', position: 'bottom', fontSize: 11 }} />
                        <YAxis type="number" dataKey="shapValue" name="SHAP Value" tick={{ fontSize: 11 }} label={{ value: 'SHAP Value', angle: -90, position: 'left', fontSize: 11 }} />
                        <ZAxis range={[50, 50]} />
                        <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p>Feature Value: <strong>{payload[0].payload.featureValue.toFixed(3)}</strong></p><p>SHAP Value: <strong className="text-indigo-600">{payload[0].payload.shapValue.toFixed(4)}</strong></p></div> : null} />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
                        <Scatter data={shapDependence} fill="#7c3aed" isAnimationActive={false} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* Feature vs Prediction Scatter */}
                  {featureVsPred && featureVsPred.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="shap-feature-vs-pred" className="border-blue-200/50 dark:border-blue-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-blue-500" />Feature vs Prediction</CardTitle>
                    <CardDescription>This chart shows how a feature's raw value relates to the model's prediction. Select a feature to see if there's a trend — for example, whether higher values of a feature lead to higher or lower predictions. This is a direct view of the feature-outcome relationship.</CardDescription>
                  </CardHeader><CardContent className="space-y-4">
                    {(() => {
                      const fn = getXaiModel()?.modelData?.featureNames || [];
                      const selFeat = fn[xaiDepFeature] || fn[0];
                      const fData = featureVsPred.filter(d => d.feature === selFeat);
                      return (<>
                        <select value={xaiDepFeature} onChange={e => setXaiDepFeature(Number(e.target.value))} className="rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="fvp-feature-select">
                          {fn.map((f, i) => <option key={i} value={i}>{f}</option>)}
                        </select>
                        <ResponsiveContainer width="100%" height={300}>
                          <ScatterChart margin={{ bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis type="number" dataKey="value" name={selFeat} tick={{ fontSize: 11 }} label={{ value: selFeat, position: 'bottom', fontSize: 11 }} />
                            <YAxis type="number" dataKey="prediction" name="Prediction" tick={{ fontSize: 11 }} label={{ value: 'Prediction', angle: -90, position: 'left', fontSize: 11 }} />
                            <ZAxis range={[50, 50]} />
                            <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p>{selFeat}: <strong>{payload[0].payload.value.toFixed(3)}</strong></p><p>Prediction: <strong className="text-blue-600">{payload[0].payload.prediction.toFixed(4)}</strong></p></div> : null} />
                            <Scatter data={fData} fill="#3b82f6" isAnimationActive={false} />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>}
                </>)}

                {/* ───── LIME TAB ───── */}
                {xaiTab === 'lime' && (<>
                  {limeResult && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-2"><div className="h-px flex-1 bg-gradient-to-r from-emerald-500/50 to-transparent" /><span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">Local Explanation — Record #{xaiRow}</span><div className="h-px flex-1 bg-gradient-to-l from-emerald-500/50 to-transparent" /></div>
                  </motion.div>}

                  {/* Instance Explanation Panel */}
                  {limeResult && <motion.div variants={fadeInUp}><Card data-testid="lime-instance-panel" className="border-emerald-200/50 dark:border-emerald-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Table2 className="h-5 w-5 text-emerald-500" />Instance Explanation Panel — Record #{xaiRow}</CardTitle>
                    <CardDescription>LIME builds a simple local model around this data point to explain the prediction. The table below shows each feature's value, its learned weight, and its contribution (value multiplied by weight). This tells you why the model made this specific prediction in a way that's easy to interpret.</CardDescription>
                  </CardHeader><CardContent>
                    <div className="flex items-center gap-6 mb-4 text-sm py-2 px-4 rounded-lg bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/20 dark:to-teal-950/20 border border-emerald-200/50 dark:border-emerald-800/30">
                      <span>Prediction: <strong className="text-emerald-600 text-base">{limeResult.prediction.toFixed(4)}</strong></span>
                      <span className="text-muted-foreground">Intercept: <strong>{limeResult.intercept.toFixed(4)}</strong></span>
                      <span className="text-muted-foreground">R² fit: <strong>{(limeResult.r2 || 0).toFixed(3)}</strong></span>
                    </div>
                    <div className="rounded-lg border overflow-auto max-h-64">
                      <table className="w-full text-sm">
                        <thead><tr className="border-b bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/20 dark:to-teal-950/20 sticky top-0">
                          <th className="p-3 text-left font-medium">Feature</th>
                          <th className="p-3 text-right font-medium">Weight</th>
                          <th className="p-3 text-right font-medium">Contribution</th>
                          <th className="p-3 text-left font-medium">Direction</th>
                        </tr></thead>
                        <tbody>{limeResult.contributions.slice(0, 15).map((c, i) => {
                          const maxAbs = Math.max(...limeResult.contributions.slice(0, 15).map(x => Math.abs(x.contribution))) || 1;
                          const pct = (Math.abs(c.contribution) / maxAbs) * 100;
                          return <tr key={i} className="border-b last:border-0 hover:bg-accent/30 transition-colors">
                            <td className="p-3 font-mono text-xs">{c.feature}</td>
                            <td className="p-3 text-right text-xs">{c.weight.toFixed(4)}</td>
                            <td className="p-3 text-right"><span className={`text-xs font-bold ${c.contribution >= 0 ? 'text-emerald-600' : 'text-rose-500'}`}>{c.contribution >= 0 ? '+' : ''}{c.contribution.toFixed(4)}</span></td>
                            <td className="p-3"><div className="w-24 h-2.5 bg-muted rounded-full overflow-hidden"><div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: c.contribution >= 0 ? '#10b981' : '#f43f5e' }} /></div></td>
                          </tr>;
                        })}</tbody>
                      </table>
                    </div>
                  </CardContent></Card></motion.div>}

                  {/* LIME Feature Contribution Bar Chart */}
                  {limeResult && <motion.div variants={fadeInUp}><Card data-testid="lime-contribution" className="border-teal-200/50 dark:border-teal-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-teal-500" />Feature Contribution Chart</CardTitle>
                    <CardDescription>This horizontal bar chart ranks features by their contribution to this prediction. Green bars support (increase) the prediction, while rose/pink bars contradict (decrease) it. Longer bars indicate stronger influence. The chart is sorted by absolute impact.</CardDescription>
                  </CardHeader><CardContent>
                    {(() => {
                      const data = limeResult.contributions.slice(0, 15);
                      return (<>
                        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
                          <BarChart layout="vertical" data={data} margin={{ left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Contribution Weight', position: 'bottom', fontSize: 11 }} />
                            <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                            <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0].payload.feature}</p><p>Weight: {payload[0].payload.weight.toFixed(4)}</p><p>Contribution: <strong className={payload[0].payload.contribution >= 0 ? 'text-emerald-600' : 'text-rose-500'}>{payload[0].payload.contribution >= 0 ? '+' : ''}{payload[0].payload.contribution.toFixed(4)}</strong></p></div> : null} />
                            <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" />
                            <Bar dataKey="contribution" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                              {data.map((d, i) => <Cell key={i} fill={d.contribution >= 0 ? '#10b981' : '#f43f5e'} />)}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                        <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#10b981' }} /> Supports prediction</span>
                          <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#f43f5e' }} /> Contradicts prediction</span>
                        </div>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>}

                  {/* Positive vs Negative Feature Impact */}
                  {limeResult && <motion.div variants={fadeInUp}><Card data-testid="lime-pos-neg" className="border-orange-200/50 dark:border-orange-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><SplitSquareVertical className="h-5 w-5 text-orange-500" />Positive vs Negative Feature Impact</CardTitle>
                    <CardDescription>Features are split into two groups: those that push the prediction higher (positive impact) and those that push it lower (negative impact). This makes it easy to see at a glance which factors are working for or against the prediction.</CardDescription>
                  </CardHeader><CardContent>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="rounded-xl border border-emerald-200/50 dark:border-emerald-800/30 p-4 bg-emerald-50/30 dark:bg-emerald-950/10">
                        <p className="text-sm font-semibold mb-3 text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5"><TrendingUp className="h-4 w-4" />Positive Impact</p>
                        <div className="space-y-2">{limeResult.contributions.filter(c => c.contribution >= 0).slice(0, 8).map((c, i) => {
                          const maxAbs = Math.max(...limeResult.contributions.filter(x => x.contribution >= 0).map(x => x.contribution)) || 1;
                          return <div key={i} className="flex items-center gap-2">
                            <span className="text-xs w-24 truncate text-right text-muted-foreground" title={c.feature}>{c.feature}</span>
                            <div className="flex-1 h-5 bg-emerald-100 dark:bg-emerald-900/30 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full transition-all" style={{ width: `${(c.contribution / maxAbs) * 100}%` }} /></div>
                            <span className="text-xs font-mono w-14 text-right font-bold text-emerald-600">+{c.contribution.toFixed(3)}</span>
                          </div>;
                        })}{limeResult.contributions.filter(c => c.contribution >= 0).length === 0 && <p className="text-xs text-muted-foreground italic">No positive contributions</p>}</div>
                      </div>
                      <div className="rounded-xl border border-rose-200/50 dark:border-rose-800/30 p-4 bg-rose-50/30 dark:bg-rose-950/10">
                        <p className="text-sm font-semibold mb-3 text-rose-500 dark:text-rose-400 flex items-center gap-1.5"><TrendingUp className="h-4 w-4 rotate-180" />Negative Impact</p>
                        <div className="space-y-2">{limeResult.contributions.filter(c => c.contribution < 0).slice(0, 8).map((c, i) => {
                          const maxAbs = Math.max(...limeResult.contributions.filter(x => x.contribution < 0).map(x => Math.abs(x.contribution))) || 1;
                          return <div key={i} className="flex items-center gap-2">
                            <span className="text-xs w-24 truncate text-right text-muted-foreground" title={c.feature}>{c.feature}</span>
                            <div className="flex-1 h-5 bg-rose-100 dark:bg-rose-900/30 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-rose-400 to-rose-500 rounded-full transition-all" style={{ width: `${(Math.abs(c.contribution) / maxAbs) * 100}%` }} /></div>
                            <span className="text-xs font-mono w-14 text-right font-bold text-rose-500">{c.contribution.toFixed(3)}</span>
                          </div>;
                        })}{limeResult.contributions.filter(c => c.contribution < 0).length === 0 && <p className="text-xs text-muted-foreground italic">No negative contributions</p>}</div>
                      </div>
                    </div>
                  </CardContent></Card></motion.div>}

                  {/* LIME Probability Chart (classification only) */}
                  {limeProbs && limeProbs.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="lime-probability" className="border-sky-200/50 dark:border-sky-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-sky-500" />Prediction Probability Distribution</CardTitle>
                    <CardDescription>For classification models, this chart shows the probability assigned to each possible class for the selected record. The tallest bar is the predicted class. This helps you understand how confident the model is in its prediction.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={limeProbs}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis dataKey="class" tick={{ fontSize: 11 }} label={{ value: 'Class', position: 'bottom', fontSize: 11 }} />
                        <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                        <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
                        <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                          {limeProbs.map((_, i) => <Cell key={i} fill={['#ec4899', '#3b82f6', '#a78bfa', '#06b6d4', '#f97316', '#10b981', '#8b5cf6', '#14b8a6'][i % 8]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* Side-by-side SHAP vs LIME comparison */}
                  {limeResult && shapLocal && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-amber-500/50 to-transparent" /><span className="text-sm font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider">Comparison</span><div className="h-px flex-1 bg-gradient-to-l from-amber-500/50 to-transparent" /></div>
                  </motion.div>}
                  {limeResult && shapLocal && <motion.div variants={fadeInUp}><Card data-testid="xai-side-by-side" className="border-amber-200/50 dark:border-amber-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><SplitSquareVertical className="h-5 w-5 text-amber-500" />SHAP vs LIME — Record #{xaiRow}</CardTitle>
                    <CardDescription>SHAP and LIME use fundamentally different approaches to explain predictions: SHAP is based on game theory (Shapley values), while LIME fits a local linear model. When both methods agree on a feature's importance, you can be more confident in that explanation. Disagreements highlight features where the explanation is model-dependent.</CardDescription>
                  </CardHeader><CardContent>
                    <div className="grid gap-6 md:grid-cols-2">
                      <div className="rounded-xl border border-violet-200/50 dark:border-violet-800/30 p-4 bg-violet-50/30 dark:bg-violet-950/10">
                        <p className="text-sm font-semibold mb-3 text-violet-600 dark:text-violet-400">SHAP Contributions</p>
                        <div className="space-y-1.5">{shapLocal.featureNames.map((f, i) => {
                          const val = shapLocal.shapValues[i]; const maxAbs = Math.max(...shapLocal.shapValues.map(Math.abs)) || 1;
                          return <div key={i} className="flex items-center gap-2"><span className="text-xs w-24 truncate text-right text-muted-foreground" title={f}>{f}</span><div className="flex-1 h-4 bg-muted rounded-full overflow-hidden relative"><div className="absolute inset-y-0" style={{ left: '50%', width: `${(Math.abs(val) / maxAbs) * 50}%`, marginLeft: val < 0 ? `${-(Math.abs(val) / maxAbs) * 50}%` : 0, backgroundColor: val >= 0 ? '#ec4899' : '#06b6d4', borderRadius: '4px' }} /></div><span className="text-xs font-mono w-16 text-right">{val >= 0 ? '+' : ''}{val.toFixed(3)}</span></div>;
                        })}</div>
                      </div>
                      <div className="rounded-xl border border-emerald-200/50 dark:border-emerald-800/30 p-4 bg-emerald-50/30 dark:bg-emerald-950/10">
                        <p className="text-sm font-semibold mb-3 text-emerald-600 dark:text-emerald-400">LIME Contributions</p>
                        <div className="space-y-1.5">{limeResult.contributions.slice(0, shapLocal.featureNames.length).map((c, i) => {
                          const maxAbs = Math.max(...limeResult.contributions.map(x => Math.abs(x.contribution))) || 1;
                          return <div key={i} className="flex items-center gap-2"><span className="text-xs w-24 truncate text-right text-muted-foreground" title={c.feature}>{c.feature}</span><div className="flex-1 h-4 bg-muted rounded-full overflow-hidden relative"><div className="absolute inset-y-0" style={{ left: '50%', width: `${(Math.abs(c.contribution) / maxAbs) * 50}%`, marginLeft: c.contribution < 0 ? `${-(Math.abs(c.contribution) / maxAbs) * 50}%` : 0, backgroundColor: c.contribution >= 0 ? '#10b981' : '#f43f5e', borderRadius: '4px' }} /></div><span className="text-xs font-mono w-16 text-right">{c.contribution >= 0 ? '+' : ''}{c.contribution.toFixed(3)}</span></div>;
                        })}</div>
                      </div>
                    </div>
                  </CardContent></Card></motion.div>}
                </>)}

                {/* ───── INSIGHTS TAB ───── */}
                {xaiTab === 'insights' && (<>
                  {!featureInsights ? (
                    <motion.div variants={fadeInUp}><Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
                      <Lightbulb className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                      <h3 className="text-lg font-semibold mb-2">Run SHAP Analysis First</h3>
                      <p className="text-muted-foreground text-sm mb-4">Switch to the SHAP tab and click "Run SHAP Analysis" to generate data for feature insights.</p>
                      <Button onClick={() => setXaiTab('shap')} size="lg" data-testid="insights-go-shap"><Sparkles className="h-4 w-4 mr-2" />Go to SHAP Analysis</Button>
                    </CardContent></Card></motion.div>
                  ) : (<>
                    {/* Key Business Insights Summary */}
                    <motion.div variants={fadeInUp}>
                      <div className="flex items-center gap-3 mb-4 mt-2"><div className="h-px flex-1 bg-gradient-to-r from-amber-500/50 to-transparent" /><span className="text-sm font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider">Key Business Insights</span><div className="h-px flex-1 bg-gradient-to-l from-amber-500/50 to-transparent" /></div>
                    </motion.div>

                    <motion.div variants={fadeInUp}><Card data-testid="key-insights-card" className="border-amber-200/50 dark:border-amber-800/30 shadow-lg bg-gradient-to-br from-amber-50/50 to-orange-50/30 dark:from-amber-950/10 dark:to-orange-950/10"><CardHeader>
                      <CardTitle className="flex items-center gap-2"><Trophy className="h-5 w-5 text-amber-500" />What Should You Do?</CardTitle>
                      <CardDescription>These are the most important findings from your model. Use these recommendations to focus your efforts on what matters most.</CardDescription>
                    </CardHeader><CardContent className="space-y-4">
                      <div className="p-4 rounded-xl border-2 border-amber-300 dark:border-amber-700 bg-white dark:bg-card" data-testid="top-2-recommendation">
                        <div className="flex items-center gap-2 mb-3"><span className="h-6 w-6 rounded-full bg-amber-500 text-white text-xs font-bold flex items-center justify-center">!</span><span className="text-sm font-bold">Top 2 Features to Focus On</span></div>
                        <p className="text-xs text-muted-foreground mb-3">These 2 features have the highest impact on <strong className="text-foreground">{featureInsights.target}</strong> and should be prioritized.</p>
                        <div className="grid gap-3 md:grid-cols-2">
                          {featureInsights.top2.map((feat, i) => {
                            const dir = featureInsights.directions[feat.feature] || {};
                            const opt = featureInsights.optimalRanges[feat.feature] || {};
                            return <div key={i} className={`rounded-lg p-4 border-2 ${i === 0 ? 'border-pink-300 dark:border-pink-700 bg-pink-50/50 dark:bg-pink-950/10' : 'border-violet-300 dark:border-violet-700 bg-violet-50/50 dark:bg-violet-950/10'}`} data-testid={`top-feature-${i}`}>
                              <div className="flex items-center gap-2 mb-2">
                                <span className={`h-7 w-7 rounded-full text-white text-xs font-bold flex items-center justify-center ${i === 0 ? 'bg-pink-500' : 'bg-violet-500'}`}>#{i + 1}</span>
                                <span className="font-bold text-sm">{feat.feature}</span>
                              </div>
                              <p className="text-xs text-muted-foreground mb-2">SHAP importance: <strong className="text-foreground">{feat.importance.toFixed(4)}</strong></p>
                              <div className={`flex items-center gap-1.5 text-xs font-semibold ${opt.recommendation === 'higher' ? 'text-emerald-600 dark:text-emerald-400' : 'text-blue-600 dark:text-blue-400'}`}>
                                <TrendingUp className={`h-3.5 w-3.5 ${opt.recommendation !== 'higher' ? 'rotate-180' : ''}`} />
                                {dir.label || (opt.recommendation === 'higher' ? `Higher values increase ${featureInsights.target}` : `Higher values decrease ${featureInsights.target}`)}
                              </div>
                              {opt.label && <p className="text-[11px] text-muted-foreground mt-1.5 italic">{opt.label}</p>}
                            </div>;
                          })}
                        </div>
                      </div>
                      {/* Action Items */}
                      <div className="space-y-2" data-testid="action-items">
                        {featureInsights.actionItems.map((item, i) => (
                          <div key={i} className="flex items-start gap-3 p-3 rounded-lg border bg-white dark:bg-card" data-testid={`action-item-${i}`}>
                            <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5 shrink-0" />
                            <p className="text-sm leading-relaxed">{item}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent></Card></motion.div>

                    {/* Top Features Ranking */}
                    <motion.div variants={fadeInUp}>
                      <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-violet-500/50 to-transparent" /><span className="text-sm font-semibold text-violet-600 dark:text-violet-400 uppercase tracking-wider">Feature Importance Ranking</span><div className="h-px flex-1 bg-gradient-to-l from-violet-500/50 to-transparent" /></div>
                    </motion.div>

                    <motion.div variants={fadeInUp}><Card data-testid="feature-ranking-card" className="border-violet-200/50 dark:border-violet-800/30 shadow-sm"><CardHeader>
                      <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-violet-500" />Top {featureInsights.topN.length} Features by SHAP Importance</CardTitle>
                      <CardDescription>Features are ranked by their average absolute SHAP value — how much they influence the model's predictions. The bars show impact direction: pink means the feature tends to push predictions up, cyan means it tends to push them down.</CardDescription>
                    </CardHeader><CardContent>
                      <div className="space-y-3">
                        {featureInsights.topN.map((feat, i) => {
                          const dir = featureInsights.directions[feat.feature] || {};
                          const opt = featureInsights.optimalRanges[feat.feature] || {};
                          const maxImp = featureInsights.topN[0]?.importance || 1;
                          const pct = (feat.importance / maxImp) * 100;
                          const isTop2 = i < 2;
                          return <div key={i} className={`rounded-lg border p-3 transition-colors ${isTop2 ? 'border-amber-300/60 dark:border-amber-700/40 bg-amber-50/30 dark:bg-amber-950/10' : 'hover:bg-accent/30'}`} data-testid={`ranked-feature-${i}`}>
                            <div className="flex items-center gap-3 mb-2">
                              <span className={`h-6 w-6 rounded-full text-xs font-bold flex items-center justify-center shrink-0 ${isTop2 ? 'bg-amber-500 text-white' : 'bg-muted text-muted-foreground'}`}>{i + 1}</span>
                              <span className="font-semibold text-sm flex-1">{feat.feature}</span>
                              <span className="text-xs font-mono text-muted-foreground">{feat.importance.toFixed(4)}</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="flex-1 h-4 bg-muted rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: dir.net >= 0 ? 'linear-gradient(to right, #ec4899, #f472b6)' : 'linear-gradient(to right, #06b6d4, #22d3ee)' }} />
                              </div>
                              <div className={`text-[10px] font-semibold shrink-0 ${dir.net >= 0 ? 'text-pink-600 dark:text-pink-400' : 'text-cyan-600 dark:text-cyan-400'}`}>
                                {dir.direction === 'increase' ? 'Pushes up' : 'Pushes down'}
                              </div>
                            </div>
                            {opt.label && <p className="text-[10px] text-muted-foreground mt-1.5">{opt.label}</p>}
                          </div>;
                        })}
                      </div>
                    </CardContent></Card></motion.div>

                    {/* SHAP Direction Summary Chart */}
                    {shapSummary && <motion.div variants={fadeInUp}><Card data-testid="insights-direction-chart" className="border-pink-200/50 dark:border-pink-800/30 shadow-sm"><CardHeader>
                      <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5 text-pink-500" />Impact Direction by Feature</CardTitle>
                      <CardDescription>This chart shows whether each feature generally pushes the {featureInsights.target} up (pink) or down (cyan). Use this to understand the direction in which you should adjust each feature to improve outcomes.</CardDescription>
                    </CardHeader><CardContent>
                      <ResponsiveContainer width="100%" height={Math.max(200, featureInsights.top5.length * 44)}>
                        <BarChart layout="vertical" data={featureInsights.top5.map(f => ({ feature: f.feature, ...(featureInsights.directions[f.feature] || {}) }))} margin={{ left: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                          <XAxis type="number" tick={{ fontSize: 11 }} />
                          <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                          <Tooltip content={({ active, payload }) => active && payload?.length ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0]?.payload?.feature}</p><p className="text-pink-500">Positive push: +{payload[0]?.payload?.positive?.toFixed(4)}</p><p className="text-cyan-500">Negative pull: {payload[0]?.payload?.negative?.toFixed(4)}</p></div> : null} />
                          <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" />
                          <Bar dataKey="positive" stackId="a" fill="#ec4899" radius={[0, 4, 4, 0]} name="Increases target" isAnimationActive={false} />
                          <Bar dataKey="negative" stackId="a" fill="#06b6d4" radius={[4, 0, 0, 4]} name="Decreases target" isAnimationActive={false} />
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="flex items-center gap-4 mt-3 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#ec4899' }} /> Increases {featureInsights.target}</span>
                        <span className="flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ backgroundColor: '#06b6d4' }} /> Decreases {featureInsights.target}</span>
                      </div>
                    </CardContent></Card></motion.div>}

                    {/* Feature Pair Analysis */}
                    {featureInsights.top2.length === 2 && (<>
                      <motion.div variants={fadeInUp}>
                        <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-blue-500/50 to-transparent" /><span className="text-sm font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">Feature Pair Analysis — Top 2</span><div className="h-px flex-1 bg-gradient-to-l from-blue-500/50 to-transparent" /></div>
                      </motion.div>

                      {/* Correlation */}
                      {featureInsights.correlation && <motion.div variants={fadeInUp}><Card data-testid="pair-correlation-card" className="border-blue-200/50 dark:border-blue-800/30 shadow-sm"><CardHeader>
                        <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5 text-blue-500" />Correlation: {featureInsights.correlation.feature1} vs {featureInsights.correlation.feature2}</CardTitle>
                        <CardDescription>This shows how the top 2 features relate to each other. A strong correlation means changes in one feature tend to accompany changes in the other. This matters because adjusting one may indirectly affect the other.</CardDescription>
                      </CardHeader><CardContent>
                        <div className="flex items-center gap-6 p-4 rounded-xl border bg-muted/30">
                          <div className="text-center">
                            <p className="text-3xl font-bold" style={{ color: Math.abs(featureInsights.correlation.r) > 0.7 ? '#ef4444' : Math.abs(featureInsights.correlation.r) > 0.3 ? '#f59e0b' : '#22c55e' }}>{featureInsights.correlation.r.toFixed(3)}</p>
                            <p className="text-xs text-muted-foreground mt-1">Correlation (r)</p>
                          </div>
                          <div className="flex-1">
                            <p className="text-sm font-medium capitalize">{featureInsights.correlation.label}</p>
                            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                              {featureInsights.correlation.strength === 'strong' ? `These features are strongly ${featureInsights.correlation.dir === 'positive' ? 'linked — they tend to increase together' : 'opposed — when one increases, the other decreases'}. Consider their joint effect when making decisions.` :
                               featureInsights.correlation.strength === 'moderate' ? `There is a moderate ${featureInsights.correlation.dir} relationship. They somewhat influence each other but can be adjusted somewhat independently.` :
                               'These features are largely independent. You can adjust one without significantly affecting the other.'}
                            </p>
                          </div>
                        </div>
                      </CardContent></Card></motion.div>}

                      {/* Scatter Plot */}
                      {featureInsights.pairScatter && featureInsights.pairScatter.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="pair-scatter-card" className="border-indigo-200/50 dark:border-indigo-800/30 shadow-sm"><CardHeader>
                        <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-indigo-500" />{featureInsights.top2[0].feature} vs {featureInsights.top2[1].feature}</CardTitle>
                        <CardDescription>Each dot is a data point. The x-axis is {featureInsights.top2[0].feature}, the y-axis is {featureInsights.top2[1].feature}, and the color represents the {featureInsights.target} value. Look for patterns: if one corner is consistently one color, that region represents optimal conditions.</CardDescription>
                      </CardHeader><CardContent>
                        {(() => {
                          const data = featureInsights.pairScatter;
                          const targets = data.map(d => typeof d.target === 'number' ? d.target : 0);
                          const tMin = arrayMin(targets), tMax = arrayMax(targets), tRange = tMax - tMin || 1;
                          return (<>
                            <ResponsiveContainer width="100%" height={380}>
                              <ScatterChart margin={{ bottom: 25, left: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                <XAxis type="number" dataKey="x" name={featureInsights.top2[0].feature} tick={{ fontSize: 11 }} label={{ value: featureInsights.top2[0].feature, position: 'bottom', fontSize: 11 }} />
                                <YAxis type="number" dataKey="y" name={featureInsights.top2[1].feature} tick={{ fontSize: 11 }} label={{ value: featureInsights.top2[1].feature, angle: -90, position: 'left', fontSize: 11 }} />
                                <ZAxis range={[50, 50]} />
                                <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p>{featureInsights.top2[0].feature}: <strong>{payload[0].payload.x?.toFixed(3)}</strong></p><p>{featureInsights.top2[1].feature}: <strong>{payload[0].payload.y?.toFixed(3)}</strong></p><p>{featureInsights.target}: <strong className="text-primary">{String(payload[0].payload.target)}</strong></p></div> : null} />
                                <Scatter data={data} isAnimationActive={false}>
                                  {data.map((d, i) => {
                                    const norm = typeof d.target === 'number' ? (d.target - tMin) / tRange : 0.5;
                                    return <Cell key={i} fill={valueToColor(norm)} />;
                                  })}
                                </Scatter>
                              </ScatterChart>
                            </ResponsiveContainer>
                            <div className="flex items-center justify-center gap-3 mt-3 py-2 px-4 rounded-lg bg-muted/30">
                              <span className="text-xs font-medium text-blue-500">Low {featureInsights.target}</span>
                              <div className="h-4 w-44 rounded-full shadow-inner" style={{ background: 'linear-gradient(to right, #3b82f6, #7c3aed, #ec4899)' }} />
                              <span className="text-xs font-medium text-pink-500">High {featureInsights.target}</span>
                            </div>
                          </>);
                        })()}
                      </CardContent></Card></motion.div>}

                      {/* Joint Insight */}
                      <motion.div variants={fadeInUp}><Card data-testid="pair-insight-card" className="border-emerald-200/50 dark:border-emerald-800/30 shadow-sm"><CardHeader>
                        <CardTitle className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-emerald-500" />Combined Feature Insight</CardTitle>
                        <CardDescription>How these two features together influence {featureInsights.target}.</CardDescription>
                      </CardHeader><CardContent>
                        <div className="space-y-3">
                          {(() => {
                            const f1 = featureInsights.top2[0].feature, f2 = featureInsights.top2[1].feature;
                            const o1 = featureInsights.optimalRanges[f1] || {};
                            const o2 = featureInsights.optimalRanges[f2] || {};
                            const rec = `${o1.recommendation === 'higher' ? 'Higher' : 'Lower'} ${f1} and ${o2.recommendation === 'higher' ? 'higher' : 'lower'} ${f2} tend to result in ${o1.recommendation === 'higher' || o2.recommendation === 'higher' ? 'higher' : 'better'} ${featureInsights.target}.`;
                            return <>
                              <div className="p-4 rounded-xl border-2 border-emerald-300 dark:border-emerald-700 bg-emerald-50/30 dark:bg-emerald-950/10" data-testid="best-value-suggestion">
                                <p className="text-sm font-semibold mb-1 text-emerald-700 dark:text-emerald-400 flex items-center gap-1.5"><Trophy className="h-4 w-4" />Best Value Suggestion</p>
                                <p className="text-sm leading-relaxed">{rec}</p>
                              </div>
                              <div className="grid gap-3 md:grid-cols-2">
                                <div className="p-3 rounded-lg border bg-muted/20">
                                  <p className="text-xs font-semibold mb-1">{f1}</p>
                                  <p className="text-xs text-muted-foreground">Avg SHAP when high: <strong className={o1.highShap > 0 ? 'text-pink-600' : 'text-cyan-600'}>{o1.highShap?.toFixed(4)}</strong></p>
                                  <p className="text-xs text-muted-foreground">Avg SHAP when low: <strong className={o1.lowShap > 0 ? 'text-pink-600' : 'text-cyan-600'}>{o1.lowShap?.toFixed(4)}</strong></p>
                                </div>
                                <div className="p-3 rounded-lg border bg-muted/20">
                                  <p className="text-xs font-semibold mb-1">{f2}</p>
                                  <p className="text-xs text-muted-foreground">Avg SHAP when high: <strong className={o2.highShap > 0 ? 'text-pink-600' : 'text-cyan-600'}>{o2.highShap?.toFixed(4)}</strong></p>
                                  <p className="text-xs text-muted-foreground">Avg SHAP when low: <strong className={o2.lowShap > 0 ? 'text-pink-600' : 'text-cyan-600'}>{o2.lowShap?.toFixed(4)}</strong></p>
                                </div>
                              </div>
                            </>;
                          })()}
                        </div>
                      </CardContent></Card></motion.div>
                    </>)}

                    {/* Optimal Range Recommendations */}
                    <motion.div variants={fadeInUp}>
                      <div className="flex items-center gap-3 mb-4 mt-6"><div className="h-px flex-1 bg-gradient-to-r from-emerald-500/50 to-transparent" /><span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">Optimization Recommendations</span><div className="h-px flex-1 bg-gradient-to-l from-emerald-500/50 to-transparent" /></div>
                    </motion.div>

                    <motion.div variants={fadeInUp}><Card data-testid="optimization-card" className="border-emerald-200/50 dark:border-emerald-800/30 shadow-sm"><CardHeader>
                      <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-emerald-500" />Feature Optimization Guide</CardTitle>
                      <CardDescription>Based on SHAP analysis, here is what tends to work best for each important feature. "Higher is better" means increasing this feature generally improves {featureInsights.target}. Use these as directional guidance, not absolute rules.</CardDescription>
                    </CardHeader><CardContent>
                      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                        {featureInsights.top5.map((feat, i) => {
                          const opt = featureInsights.optimalRanges[feat.feature] || {};
                          const dir = featureInsights.directions[feat.feature] || {};
                          const isUp = opt.recommendation === 'higher';
                          return <div key={i} className={`rounded-lg border p-4 transition-colors ${i < 2 ? 'border-amber-200/60 dark:border-amber-700/40 bg-amber-50/20 dark:bg-amber-950/5' : ''}`} data-testid={`opt-feature-${i}`}>
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-semibold text-sm">{feat.feature}</span>
                              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${isUp ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'}`}>
                                {isUp ? 'Higher is better' : 'Lower is better'}
                              </span>
                            </div>
                            <div className="flex items-center gap-2 mb-2">
                              <TrendingUp className={`h-4 w-4 ${isUp ? 'text-emerald-500' : 'text-blue-500 rotate-180'}`} />
                              <span className="text-xs text-muted-foreground">{opt.label || dir.label || ''}</span>
                            </div>
                            <div className="flex items-center gap-2 text-[10px]">
                              <span className="text-muted-foreground">High val SHAP:</span>
                              <span className={opt.highShap >= 0 ? 'text-pink-600 font-mono' : 'text-cyan-600 font-mono'}>{opt.highShap?.toFixed(3)}</span>
                              <span className="text-muted-foreground ml-2">Low val SHAP:</span>
                              <span className={opt.lowShap >= 0 ? 'text-pink-600 font-mono' : 'text-cyan-600 font-mono'}>{opt.lowShap?.toFixed(3)}</span>
                            </div>
                          </div>;
                        })}
                      </div>
                    </CardContent></Card></motion.div>
                  </>)}
                </>)}

                {/* ───── CLUSTER TAB ───── */}
                {xaiTab === 'clusters' && (<>
                  {clusterShap && <motion.div variants={fadeInUp}>
                    <div className="flex items-center gap-3 mb-4 mt-2"><div className="h-px flex-1 bg-gradient-to-r from-blue-500/50 to-transparent" /><span className="text-sm font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">Cluster Explainability</span><div className="h-px flex-1 bg-gradient-to-l from-blue-500/50 to-transparent" /></div>
                  </motion.div>}

                  {/* Cluster Feature Influence */}
                  {clusterShap && <motion.div variants={fadeInUp}><Card data-testid="cluster-feature-influence" className="border-blue-200/50 dark:border-blue-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5 text-blue-500" />Cluster Feature Influence</CardTitle>
                    <CardDescription>This chart shows which features most strongly influence how data points are assigned to clusters. Features with higher values are the primary drivers of cluster separation. This helps you understand what makes each cluster distinct from others.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(200, clusterShap.importance.length * 40)}>
                      <BarChart layout="vertical" data={clusterShap.importance} margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Mean |SHAP Value| for Cluster Assignment', position: 'bottom', fontSize: 11 }} />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                        <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0].payload.feature}</p><p>Cluster SHAP: <strong className="text-blue-600">{payload[0].payload.importance.toFixed(4)}</strong></p></div> : null} />
                        <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                          {clusterShap.importance.map((_, i) => <Cell key={i} fill={`hsl(${210 + (i / clusterShap.importance.length) * 50}, 75%, ${45 + i * 3}%)`} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* Cluster Comparison Bar Charts */}
                  {clusterComparison && <motion.div variants={fadeInUp}><Card data-testid="cluster-comparison-chart" className="border-teal-200/50 dark:border-teal-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-teal-500" />Cluster Comparison</CardTitle>
                    <CardDescription>This grouped bar chart compares feature averages across all clusters. Each group of bars represents one feature, and the colors represent different clusters. Significant differences between clusters for a feature indicate that feature helps distinguish clusters. The gray bar shows the overall average for reference.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={Math.max(250, clusterComparison.data.length * 30)}>
                      <BarChart data={clusterComparison.data}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis dataKey="feature" angle={-35} textAnchor="end" height={80} tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend />
                        {Array.from({ length: clusterComparison.k }, (_, c) => (
                          <Bar key={c} dataKey={`cluster_${c}`} name={`Cluster ${c}`} fill={CLUSTER_COLORS[c % CLUSTER_COLORS.length]} radius={[3, 3, 0, 0]} />
                        ))}
                        <Bar dataKey="overall" name="Overall" fill="#6b7280" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* PCA Scatter Plot (color-coded clusters) */}
                  {unsupervisedResult?.pca && <motion.div variants={fadeInUp}><Card data-testid="cluster-pca-scatter" className="border-cyan-200/50 dark:border-cyan-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5 text-cyan-500" />Cluster PCA Scatter Plot</CardTitle>
                    <CardDescription>Data points projected into 2D using Principal Component Analysis. Each dot is colored by its cluster assignment. Well-separated clusters in this view indicate the model successfully found distinct groups. Overlapping regions may suggest ambiguous cluster boundaries.</CardDescription>
                  </CardHeader><CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                        <XAxis dataKey="x" name="PC1" type="number" tick={{ fontSize: 11 }} label={{ value: `PC1 (${(unsupervisedResult.pca.explainedVariance[0] * 100).toFixed(1)}% variance)`, position: 'bottom', fontSize: 11 }} />
                        <YAxis dataKey="y" name="PC2" type="number" tick={{ fontSize: 11 }} label={{ value: `PC2 (${(unsupervisedResult.pca.explainedVariance[1] * 100).toFixed(1)}% variance)`, angle: -90, position: 'left', fontSize: 11 }} />
                        <ZAxis range={[60, 60]} />
                        <Tooltip content={({ active, payload }) => active && payload?.length ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold" style={{ color: CLUSTER_COLORS[payload[0].payload.cluster % CLUSTER_COLORS.length] }}>Cluster {payload[0].payload.cluster}</p><p>PC1: {payload[0].payload.x.toFixed(3)}</p><p>PC2: {payload[0].payload.y.toFixed(3)}</p></div> : null} />
                        <Legend />
                        {[...new Set(unsupervisedResult.pca.points.map(p => p.cluster))].sort((a, b) => a - b).map(c => (
                          <Scatter key={c} name={`Cluster ${c}`} data={unsupervisedResult.pca.points.filter(p => p.cluster === c)} fill={CLUSTER_COLORS[c % CLUSTER_COLORS.length]} />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </CardContent></Card></motion.div>}

                  {/* Feature Distribution per Cluster */}
                  {clusterComparison && <motion.div variants={fadeInUp}><Card data-testid="cluster-feature-distribution" className="border-purple-200/50 dark:border-purple-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><BarChart2 className="h-5 w-5 text-purple-500" />Feature Distribution per Cluster</CardTitle>
                    <CardDescription>Each mini-chart shows how a feature's average value differs across clusters. This helps identify the defining characteristics of each cluster. For example, if one cluster has much higher values for a feature, that feature is a key distinguishing trait.</CardDescription>
                  </CardHeader><CardContent>
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                      {clusterComparison.data.map((feat, fi) => (
                        <div key={fi} className="rounded-lg border p-3 bg-muted/10 hover:bg-muted/20 transition-colors">
                          <p className="text-xs font-semibold mb-2 truncate" title={feat.feature}>{feat.feature}</p>
                          <div className="space-y-1.5">
                            {Array.from({ length: clusterComparison.k }, (_, c) => {
                              const val = feat[`cluster_${c}`] || 0;
                              const maxVal = Math.max(...Array.from({ length: clusterComparison.k }, (_, ci) => Math.abs(feat[`cluster_${ci}`] || 0)), Math.abs(feat.overall || 0)) || 1;
                              return <div key={c} className="flex items-center gap-2">
                                <span className="text-[10px] w-8 text-right font-medium" style={{ color: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }}>C{c}</span>
                                <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                                  <div className="h-full rounded-full transition-all" style={{ width: `${(Math.abs(val) / maxVal) * 100}%`, backgroundColor: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                                </div>
                                <span className="text-[10px] font-mono w-12 text-right">{val.toFixed(2)}</span>
                              </div>;
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent></Card></motion.div>}

                  {/* Cluster SHAP Distribution (Beeswarm) */}
                  {clusterBeeswarm && <motion.div variants={fadeInUp}><Card data-testid="cluster-shap-distribution" className="border-indigo-200/50 dark:border-indigo-800/30 shadow-sm"><CardHeader>
                    <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5 text-indigo-500" />Cluster SHAP Distribution</CardTitle>
                    <CardDescription>This beeswarm plot shows how each feature contributes to cluster assignments across all data points. Like the supervised beeswarm, color encodes the feature value (blue = low, pink = high). Spread along the X-axis shows the range of SHAP values, indicating varying influence across data points.</CardDescription>
                  </CardHeader><CardContent>
                    {(() => {
                      const fn = unsupervisedResult?.preprocessing?.featureNames || [];
                      const chartData = clusterBeeswarm.points.map(p => ({ x: p.shapValue, y: p.featureIdx + p.jitter, fill: p.color }));
                      return (<>
                        <ResponsiveContainer width="100%" height={Math.max(250, fn.length * 44)}>
                          <ScatterChart margin={{ left: 20, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                            <XAxis type="number" dataKey="x" name="SHAP Value" tick={{ fontSize: 11 }} label={{ value: 'SHAP Value (cluster influence)', position: 'bottom', fontSize: 11 }} />
                            <YAxis type="number" dataKey="y" domain={[-0.5, fn.length - 0.5]} ticks={fn.map((_, i) => i)} tickFormatter={v => fn[Math.round(v)] || ''} tick={{ fontSize: 11 }} width={120} />
                            <ZAxis range={[24, 24]} />
                            <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{fn[Math.round(payload[0].payload.y)]}</p><p>SHAP: <strong>{payload[0].payload.x.toFixed(4)}</strong></p></div> : null} />
                            <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 4" />
                            <Scatter data={chartData} isAnimationActive={false}>
                              {chartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                            </Scatter>
                          </ScatterChart>
                        </ResponsiveContainer>
                        <div className="flex items-center justify-center gap-3 mt-4 py-2 px-4 rounded-lg bg-muted/30">
                          <span className="text-xs font-medium text-blue-500">Low</span>
                          <div className="h-4 w-44 rounded-full shadow-inner" style={{ background: 'linear-gradient(to right, #3b82f6, #7c3aed, #ec4899)' }} />
                          <span className="text-xs font-medium text-pink-500">High</span>
                          <span className="text-xs text-muted-foreground ml-1">(Feature Value)</span>
                        </div>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>}
                </>)}
              </>)}
            </motion.div>
          )}

          {/* ==================== DATA EXPLORER ==================== */}
          {activeView === 'explore' && (
            <motion.div key="explore" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="explore-view">
              {!dataProfile ? (
                <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
                  <BarChart2 className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
                  <h3 className="text-lg font-semibold mb-2">No Data Loaded</h3>
                  <p className="text-muted-foreground text-sm mb-4">Please upload a dataset in the Analysis tab first, then come back to explore your data visually.</p>
                  <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="explore-go-analysis"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
                </CardContent></Card>
              ) : (<>
                {/* Histogram */}
                <motion.div variants={fadeInUp}>
                <Card data-testid="histogram-card"><CardHeader>
                  <CardTitle className="flex items-center gap-2"><BarChart2 className="h-5 w-5" /><HelpTip text="Histograms show the distribution of values in a numeric column. Each bar represents a range, and the bar height shows how many data points fall in that range. Look for skewed distributions, outliers, or multimodal patterns.">Feature Histograms</HelpTip></CardTitle>
                  <CardDescription>Select a numeric feature to see its distribution.</CardDescription>
                </CardHeader><CardContent className="space-y-4">
                  <select value={histogramCol} onChange={e => setHistogramCol(e.target.value)} className="w-full max-w-xs rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="histogram-col-select">
                    <option value="">-- Select Column --</option>
                    {dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                  {histogramData.length > 0 && (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={histogramData}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="bin" angle={-30} textAnchor="end" height={60} tick={{fontSize: 10}} />
                        <YAxis tick={{fontSize: 11}} />
                        <Tooltip formatter={(v) => [`${v} rows`, 'Count']} />
                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                  {histogramCol && histogramData.length > 0 && (() => {
                    const vals = dataProfile.rows.map(r => r[histogramCol]).filter(v => typeof v === 'number');
                    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                    const sorted = [...vals].sort((a, b) => a - b);
                    const median = sorted.length % 2 === 0 ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 : sorted[Math.floor(sorted.length / 2)];
                    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
                    return (
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Count</p><p className="text-lg font-bold">{vals.length}</p></div>
                        <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Mean</p><p className="text-lg font-bold">{mean.toFixed(2)}</p></div>
                        <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Median</p><p className="text-lg font-bold">{median.toFixed(2)}</p></div>
                        <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Std Dev</p><p className="text-lg font-bold">{std.toFixed(2)}</p></div>
                        <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Range</p><p className="text-lg font-bold">{arrayMin(vals).toFixed(1)} – {arrayMax(vals).toFixed(1)}</p></div>
                      </div>
                    );
                  })()}
                </CardContent></Card>
                </motion.div>

                {/* Correlation Heatmap */}
                {corrMatrix.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="explore-correlation-heatmap"><CardHeader><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Correlation Heatmap</CardTitle><CardDescription>Pairwise correlations between all numeric variables. Blue = positive, red = negative.</CardDescription></CardHeader>
                  <CardContent><div className="overflow-auto">
                    <div className="inline-grid gap-px" style={{gridTemplateColumns: `120px repeat(${dataProfile.numericColumns.length}, minmax(60px, 1fr))`}}>
                      <div />
                      {dataProfile.numericColumns.map(col => <div key={col} className="text-xs font-mono p-2 text-center truncate" title={col}>{col}</div>)}
                      {corrMatrix.map((row, ri) => (<React.Fragment key={ri}>
                        <div className="text-xs font-mono p-2 text-right truncate" title={row.feature}>{row.feature}</div>
                        {dataProfile.numericColumns.map((col, ci) => { const val = row[col] || 0; const bg = val > 0 ? `rgba(37, 99, 235, ${Math.abs(val) * 0.8})` : `rgba(220, 38, 38, ${Math.abs(val) * 0.8})`; return <div key={ci} className="p-2 text-center text-xs font-mono rounded-sm border border-muted/30" style={{backgroundColor: bg, color: Math.abs(val) > 0.4 ? 'white' : 'inherit'}} title={`${row.feature} vs ${col}: ${val.toFixed(3)}`}>{val.toFixed(2)}</div>; })}
                      </React.Fragment>))}
                    </div>
                  </div></CardContent></Card></motion.div>}

                {/* Scatter Plot */}
                <motion.div variants={fadeInUp}><Card data-testid="explore-scatter"><CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" />Scatter Plot</CardTitle><CardDescription>Select two numeric variables to visualize their relationship.</CardDescription></CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="space-y-1.5"><label className="text-sm font-medium">X Variable</label><select value={corrVarX} onChange={e => setCorrVarX(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="explore-var-x"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                      <div className="space-y-1.5"><label className="text-sm font-medium">Y Variable</label><select value={corrVarY} onChange={e => setCorrVarY(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="explore-var-y"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                    </div>
                    {corrVarX && corrVarY && (() => {
                      const pairs = dataProfile.rows.filter(r => typeof r[corrVarX] === 'number' && typeof r[corrVarY] === 'number').map(r => ({ x: r[corrVarX], y: r[corrVarY] }));
                      const n = pairs.length; if (n < 3) return <p className="text-sm text-muted-foreground">Insufficient data for these variables.</p>;
                      const mx = pairs.reduce((s, p) => s + p.x, 0) / n, my = pairs.reduce((s, p) => s + p.y, 0) / n;
                      let sx = 0, sy = 0, sxy = 0; pairs.forEach(p => { sx += (p.x - mx) ** 2; sy += (p.y - my) ** 2; sxy += (p.x - mx) * (p.y - my); });
                      const r = (sx > 0 && sy > 0) ? sxy / Math.sqrt(sx * sy) : 0;
                      const abs = Math.abs(r); const dir = r >= 0 ? 'Positive' : 'Negative';
                      const str = abs >= 0.8 ? 'Strong' : abs >= 0.5 ? 'Moderate' : abs >= 0.3 ? 'Weak' : 'Very Weak';
                      return (<>
                        <div className={`p-4 rounded-lg border-2 ${abs >= 0.5 ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20' : abs >= 0.3 ? 'border-amber-300 bg-amber-50 dark:bg-amber-950/20' : 'border-muted bg-muted/30'}`} data-testid="explore-corr-result">
                          <div className="flex items-center justify-between"><div><p className="text-sm font-medium">Correlation Coefficient</p><p className="text-3xl font-bold">{r.toFixed(4)}</p></div><Badge variant={abs >= 0.5 ? 'default' : 'secondary'}>{str} {dir}</Badge></div>
                        </div>
                        <ResponsiveContainer width="100%" height={350}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="x" name={corrVarX} type="number" tick={{fontSize: 11}} label={{value: corrVarX, position: 'bottom', fontSize: 11}} /><YAxis dataKey="y" name={corrVarY} type="number" tick={{fontSize: 11}} label={{value: corrVarY, angle: -90, position: 'left', fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip /><Scatter name="Data" data={pairs} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer>
                      </>);
                    })()}
                  </CardContent></Card></motion.div>
              </>)}
            </motion.div>
          )}

          {/* ==================== CLUSTERS ==================== */}
          {activeView === 'clusters' && (
            <motion.div key="clusters" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="clusters-view">
              {!dataProfile ? <DataUploadMini /> : (<>
                <motion.div variants={fadeInUp}><Card data-testid="cluster-config-card"><CardHeader><CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Configuration</CardTitle><CardDescription>{dataProfile.numericColumns.length} numeric features, {dataProfile.rowCount} rows</CardDescription></CardHeader>
                  <CardContent><div className="flex items-center gap-6 flex-wrap">
                    <div className="space-y-2 flex-1 min-w-[200px]"><label className="text-sm font-medium">Clusters (k): <span className="text-primary font-bold">{numClusters}</span></label><input type="range" min={2} max={Math.min(10, dataProfile.rowCount - 1)} value={numClusters} onChange={(e) => setNumClusters(Number(e.target.value))} className="w-full accent-primary" data-testid="cluster-k-slider" /></div>
                    <Button onClick={handleClustering} size="lg" className="h-12" data-testid="run-clustering-btn"><Layers className="h-4 w-4 mr-2" />Run K-Means</Button>
                  </div></CardContent></Card></motion.div>
                {clusterResult && (<>
                  <motion.div variants={fadeInUp}><div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5">{clusterResult.clusterStats.map((cs) => <Card key={cs.clusterId} data-testid={`cluster-stat-${cs.clusterId}`}><CardContent className="p-4 text-center"><div className="h-4 w-full rounded-full mb-3" style={{ backgroundColor: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length], opacity: 0.3 }} /><p className="text-sm text-muted-foreground">Cluster {cs.clusterId}</p><p className="text-3xl font-bold" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>{cs.size}</p><p className="text-xs text-muted-foreground">points</p></CardContent></Card>)}</div></motion.div>
                  <motion.div variants={fadeInUp}><Card data-testid="cluster-scatter-chart"><CardHeader><CardTitle>Visualization</CardTitle><CardDescription>{clusterResult.xFeature} vs {clusterResult.yFeature}</CardDescription></CardHeader><CardContent><ResponsiveContainer width="100%" height={400}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" name={clusterResult.xFeature} type="number" /><YAxis dataKey="y" name={clusterResult.yFeature} type="number" /><ZAxis range={[60, 60]} /><Tooltip /><Legend />{Array.from({ length: clusterResult.k }, (_, i) => <Scatter key={i} name={`Cluster ${i}`} data={clusterResult.points.filter(p => p.cluster === i)} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />)}</ScatterChart></ResponsiveContainer></CardContent></Card></motion.div>
                  <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle>Size Distribution</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={250}><BarChart data={clusterResult.clusterStats.map(cs => ({ name: `Cluster ${cs.clusterId}`, size: cs.size }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="size" radius={[4, 4, 0, 0]}>{clusterResult.clusterStats.map((_, i) => <Cell key={i} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />)}</Bar></BarChart></ResponsiveContainer></CardContent></Card></motion.div>
                  <motion.div variants={fadeInUp}><Card data-testid="cluster-means-table"><CardHeader><CardTitle>Cluster Centers</CardTitle></CardHeader><CardContent><div className="rounded-md border overflow-auto"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Feature</th>{clusterResult.clusterStats.map(cs => <th key={cs.clusterId} className="p-3 text-center font-medium" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>C{cs.clusterId}</th>)}</tr></thead><tbody>{clusterResult.features.map((feat, fi) => <tr key={fi} className="border-b last:border-0"><td className="p-3 font-mono text-xs">{feat}</td>{clusterResult.clusterStats.map(cs => <td key={cs.clusterId} className="p-3 text-center text-xs">{cs.means[fi]?.mean?.toFixed(2)}</td>)}</tr>)}</tbody></table></div></CardContent></Card></motion.div>
                </>)}
              </>)}
            </motion.div>
          )}

          {/* ==================== ANOMALIES ==================== */}
          {activeView === 'anomalies' && (
            <motion.div key="anomalies" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="anomalies-view">
              {!dataProfile ? <DataUploadMini /> : (<>
                <motion.div variants={fadeInUp}><Card data-testid="anomaly-config-card"><CardHeader><CardTitle className="flex items-center gap-2"><ShieldAlert className="h-5 w-5" />Configuration</CardTitle></CardHeader>
                  <CardContent><div className="flex items-center gap-6 flex-wrap">
                    <div className="space-y-2"><label className="text-sm font-medium">Method</label><select value={anomalyMethod} onChange={(e) => setAnomalyMethod(e.target.value)} className="rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="anomaly-method-select"><option value="zscore">Z-Score</option><option value="iqr">IQR</option></select></div>
                    {anomalyMethod === 'zscore' && <div className="space-y-2"><label className="text-sm font-medium">Threshold: <span className="text-primary font-bold">{anomalyThreshold}</span></label><input type="range" min={1.5} max={4} step={0.5} value={anomalyThreshold} onChange={(e) => setAnomalyThreshold(Number(e.target.value))} className="w-40 accent-primary" data-testid="anomaly-threshold-slider" /></div>}
                    <Button onClick={handleAnomalyDetection} size="lg" className="h-12" data-testid="run-anomaly-btn"><ShieldAlert className="h-4 w-4 mr-2" />Detect Anomalies</Button>
                  </div></CardContent></Card></motion.div>
                {anomalyResult && (<>
                  <motion.div variants={fadeInUp}><div className="grid gap-4 md:grid-cols-3"><Card data-testid="anomaly-count-card"><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Anomalies</p><p className="text-4xl font-bold text-destructive">{anomalyResult.totalAnomalies}</p></CardContent></Card><Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Normal</p><p className="text-4xl font-bold text-primary">{anomalyResult.totalRows - anomalyResult.totalAnomalies}</p></CardContent></Card><Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Rate</p><p className="text-4xl font-bold">{(anomalyResult.totalAnomalies / anomalyResult.totalRows * 100).toFixed(1)}%</p></CardContent></Card></div></motion.div>
                  {anomalyResult.xFeature && <motion.div variants={fadeInUp}><Card data-testid="anomaly-scatter-chart"><CardHeader><CardTitle>Normal vs Anomaly</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={400}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" /><YAxis dataKey="y" type="number" /><ZAxis range={[60, 60]} /><Tooltip /><Legend /><Scatter name="Normal" data={anomalyResult.normalPoints} fill="hsl(var(--primary))" /><Scatter name="Anomaly" data={anomalyResult.anomalyPoints} fill="hsl(var(--destructive))" /></ScatterChart></ResponsiveContainer></CardContent></Card></motion.div>}
                  <motion.div variants={fadeInUp}><Card data-testid="anomaly-per-column"><CardHeader><CardTitle>Per Column</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={250}><BarChart data={Object.entries(anomalyResult.anomalies).map(([col, items]) => ({ name: col, count: items.length })).filter(d => d.count > 0)}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-30} textAnchor="end" height={80} /><YAxis /><Tooltip /><Bar dataKey="count" fill="hsl(var(--destructive))" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></CardContent></Card></motion.div>
                  {anomalyResult.anomalyRowIndices.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="anomaly-rows-table"><CardHeader><CardTitle>Anomalous Rows</CardTitle></CardHeader><CardContent><div className="rounded-md border overflow-auto max-h-80"><table className="w-full text-sm"><thead><tr className="border-b bg-destructive/10 sticky top-0"><th className="p-2 text-left font-medium">Row</th>{dataProfile.numericColumns.slice(0, 6).map(col => <th key={col} className="p-2 text-left font-medium">{col}</th>)}</tr></thead><tbody>{anomalyResult.anomalyRowIndices.slice(0, 20).map(ri => <tr key={ri} className="border-b last:border-0 bg-destructive/5"><td className="p-2 font-mono text-xs font-bold">{ri + 1}</td>{dataProfile.numericColumns.slice(0, 6).map(col => { const isA = anomalyResult.anomalies[col]?.some(a => a.index === ri); return <td key={col} className={`p-2 text-xs ${isA ? 'text-destructive font-bold' : ''}`}>{typeof dataProfile.rows[ri]?.[col] === 'number' ? dataProfile.rows[ri][col].toFixed(2) : '-'}</td>; })}</tr>)}</tbody></table></div></CardContent></Card></motion.div>}
                </>)}
              </>)}
            </motion.div>
          )}

          {/* ==================== MODELS ==================== */}
          {activeView === 'models' && (
            <motion.div key="models" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="models-view">
              <Card><CardHeader><div className="flex items-center justify-between"><div><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Model Library</CardTitle></div><div className="flex items-center gap-2">
                <div className="relative"><input type="file" accept=".json" onChange={handleImportModel} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="import-model-input" /><Button variant="outline" size="sm" data-testid="import-model-btn"><Upload className="h-4 w-4 mr-2" />Import Model</Button></div>
                <Badge variant="secondary" className="text-lg px-4 py-2" data-testid="models-count-badge">{models.length} Models</Badge>
              </div></div></CardHeader>
                <CardContent>{models.length === 0 ? <div className="text-center py-12" data-testid="empty-models"><Database className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" /><h3 className="text-lg font-medium mb-2">No Models Yet</h3><Button onClick={() => setActiveView('analysis')} size="lg"><Zap className="h-4 w-4 mr-2" />Train Your First Model</Button></div>
                : <div className="rounded-md border"><table className="w-full" data-testid="models-table"><thead><tr className="border-b bg-muted/50"><th className="p-4 text-left text-sm font-medium">Model ID</th><th className="p-4 text-left text-sm font-medium">Algorithm</th><th className="p-4 text-left text-sm font-medium">Type</th><th className="p-4 text-left text-sm font-medium">Created</th><th className="p-4 text-left text-sm font-medium">Actions</th></tr></thead>
                  <tbody>{models.map((model, idx) => <motion.tr key={model.modelId} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }} className="border-b last:border-0 hover:bg-accent/50 transition-colors" data-testid={`model-row-${idx}`}><td className="p-4"><div className="flex items-center gap-2"><div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center"><Brain className="h-4 w-4 text-primary" /></div><code className="text-xs font-mono">{model.modelId.substring(0, 12)}...</code></div></td><td className="p-4"><Badge variant="outline">{ALGO_NAMES[model.algorithm] || model.algorithm}</Badge></td><td className="p-4 text-sm">{model.problemType}</td><td className="p-4 text-sm text-muted-foreground">{new Date(model.createdAt).toLocaleDateString()}</td><td className="p-4"><div className="flex gap-2"><Button variant="ghost" size="sm" onClick={() => setActiveView('predict')} data-testid={`use-model-${idx}`}><Eye className="h-4 w-4" /></Button><Button variant="ghost" size="sm" onClick={() => handleDownloadModel(model.modelId)} className="text-primary" data-testid={`download-model-${idx}`}><Download className="h-4 w-4" /></Button><Button variant="ghost" size="sm" onClick={() => handleDeleteModel(model.modelId)} data-testid={`delete-model-${idx}`}><Trash2 className="h-4 w-4 text-destructive" /></Button></div></td></motion.tr>)}</tbody></table></div>
                }</CardContent></Card>
            </motion.div>
          )}

          {/* ==================== HISTORY ==================== */}
          {activeView === 'history' && (
            <motion.div key="history" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="history-view">
              <motion.div variants={fadeInUp}><Card><CardHeader>
                <div className="flex items-center justify-between">
                  <div><CardTitle className="flex items-center gap-2"><History className="h-5 w-5" />Saved Analyses</CardTitle>
                    <CardDescription>Your analysis sessions are saved here. Click to restore, share, or delete.</CardDescription></div>
                  <div className="flex items-center gap-2">
                    {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button size="sm" onClick={() => handleSaveAnalysis()} data-testid="save-current-analysis-btn"><Save className="h-4 w-4 mr-2" />Save Current Analysis</Button>}
                    <Button variant="outline" size="sm" onClick={fetchHistory} data-testid="refresh-history-btn"><ArrowUpRight className="h-4 w-4 mr-2" />Refresh</Button>
                  </div>
                </div>
              </CardHeader><CardContent>
                {historyLoading ? (
                  <div className="py-12 text-center"><div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mx-auto mb-3" /><p className="text-sm text-muted-foreground">Loading history...</p></div>
                ) : historyList.length === 0 ? (
                  <div className="py-12 text-center" data-testid="history-empty">
                    <History className="h-14 w-14 text-muted-foreground/30 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No Saved Analyses</h3>
                    <p className="text-muted-foreground text-sm mb-4">Train a model and click "Save" to store your analysis for later.</p>
                  </div>
                ) : (
                  <div className="space-y-3" data-testid="history-list">
                    {historyList.map((snap, idx) => (
                      <div key={snap.snapshot_id || idx} className="group rounded-lg border p-4 hover:bg-accent/30 transition-colors" data-testid={`history-item-${idx}`}>
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <h4 className="font-semibold text-sm truncate">{snap.name}</h4>
                              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${snap.problem_type === 'classification' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' : snap.problem_type === 'regression' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400'}`}>{snap.problem_type || 'unknown'}</span>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-muted-foreground">
                              <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{new Date(snap.created_at).toLocaleString()}</span>
                              <span>{snap.row_count} rows, {snap.col_count} cols</span>
                              {snap.target_column && <span>Target: <strong>{snap.target_column}</strong></span>}
                            </div>
                            {snap.models_summary?.length > 0 && <div className="flex flex-wrap gap-1.5 mt-2">
                              {snap.models_summary.slice(0, 5).map((m, mi) => (
                                <span key={mi} className="text-[10px] px-2 py-0.5 rounded-full bg-muted border">{ALGO_NAMES[m.algorithm] || m.algorithm}: {m.score?.toFixed(3)}</span>
                              ))}
                            </div>}
                          </div>
                          <div className="flex items-center gap-1.5 ml-4 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Button variant="default" size="sm" onClick={() => handleLoadSnapshot(snap.snapshot_id)} data-testid={`load-snapshot-${idx}`}><Play className="h-3.5 w-3.5 mr-1.5" />Restore</Button>
                            <Button variant="outline" size="sm" onClick={async () => { const url = `${window.location.origin}${window.location.pathname}?snapshot=${snap.snapshot_id}`; setShareUrl(url); const ok = await safeCopyToClipboard(url); setShareCopyStatus(ok ? 'copied' : 'manual'); }} data-testid={`share-snapshot-${idx}`}><Share2 className="h-3.5 w-3.5" /></Button>
                            <Button variant="outline" size="sm" className="text-destructive hover:bg-destructive/10" onClick={() => handleDeleteSnapshot(snap.snapshot_id)} data-testid={`delete-snapshot-${idx}`}><Trash2 className="h-3.5 w-3.5" /></Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent></Card></motion.div>
            </motion.div>
          )}

        </AnimatePresence></main>
      </div>
    </div>
  );
}

export default App;
