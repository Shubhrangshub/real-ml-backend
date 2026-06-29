import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AuthPage from './AuthPage';
import { useAuth } from './hooks/useAuth';
import {
  Brain, Sparkles, TrendingUp, Activity, Database, Zap, Upload, Play,
  Eye, Trash2, ChevronRight, ArrowUpRight, FileText, Target, Cpu, BarChart3,
  Download, AlertCircle, Layers, ShieldAlert, Table2, SplitSquareVertical, Info,
  Clock, Trophy, CheckCircle2, XCircle, Shield, Moon, Sun, FileUp, BarChart2,
  BookOpen, Lightbulb, Save, History, Share2, Copy, ExternalLink, Lock, Sheet, LogOut,
  GitBranch, X, Rocket, Sliders, FileDown, Settings2, SlidersHorizontal
} from 'lucide-react';
import { Toaster, toast } from 'sonner';
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
import ExplainabilityView from './components/views/ExplainabilityView';
import CompareModelsView from './components/views/CompareModelsView';
import LeaderboardView from './components/views/LeaderboardView';
import AdminView from './components/views/AdminView';
import DeployView from './components/views/DeployView';
import WhatIfView from './components/views/WhatIfView';
import PublicPredictPage from './components/PublicPredictPage';
import OnboardingGuide from './components/OnboardingGuide';
import DashboardView from './components/views/DashboardView';
import AnalysisView from './components/views/AnalysisView';
import PredictView from './components/views/PredictView';
import DataExplorerView from './components/views/DataExplorerView';
import HistoryView from './components/views/HistoryView';
import { ClustersView, AnomaliesView, ModelsView } from './components/views/SmallViews';
import PreprocessView from './components/views/PreprocessView';
import TuneView from './components/views/TuneView';

// ==================== EXTRACTED MODULES ====================
import {
  CLUSTER_COLORS, fadeInUp, staggerContainer, ALGO_NAMES, ALGO_COLORS,
  ALGO_DESCRIPTIONS, GUIDE_STEPS, METRIC_EXPLANATIONS, UNSUPERVISED_TERMS,
  SAMPLE_DATASETS
} from './constants';
import AppContext from './context/AppContext';
import { getScoreColor, interpretMetric, arrayMin, arrayMax, arrayMinMax, generateId } from './utils/helpers';
import { MetricTip } from './components/SmartTooltip';
import { generateReport } from './utils/reportGenerator';
import {
  handleMissingValues, handleOutliers, scaleFeatures,
  getDefaultPreprocessConfig,
} from './utils/preprocessUtils';
import {
  parseCSV, profileDataset, generateDatasetSummary, suggestTask,
  scanDataset, cleanRemoveDuplicates, cleanFillMissing,
  cleanRemoveOutliers, cleanDropConstants, cleanNormalize,
  runKMeansClustering, detectAnomaliesFunc
} from './utils/datasetUtils';
import {
  detectProblemType, prepareFeatures, trainTestSplit,
  predictOne, predictBatch,
  calcRegressionMetrics, calcClassificationMetrics, extractImportance,
  prepareInputForPrediction, kFoldCrossValidation, buildModelForAlgo
} from './utils/mlEngine';

// ==================== APP COMPONENT ====================

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

function App() {
  const { authUser, setAuthUser, authChecked, handleLogout } = useAuth();

  // Show login page if not authenticated
  if (!authChecked) {
    return <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-zinc-950 dark:to-zinc-900"><div className="h-8 w-8 animate-spin rounded-full border-4 border-violet-500 border-t-transparent" /></div>;
  }
  if (!authUser) {
    // Allow view-only shared snapshots without login
    const params = new URLSearchParams(window.location.search);
    if (!params.get('snapshot')) {
      return <AuthPage onAuth={(user) => { setAuthUser(user); }} />;
    }
  }

  return <AppMain authUser={authUser} onLogout={handleLogout} />;
}

function AppMain({ authUser, onLogout }) {
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
  const isRestoringRef = useRef(false);
  const lastSavedFingerprintRef = useRef(null);
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

  // Helper: compute best model index by score (used for default selection)
  const getBestModelIdx = useCallback((modelList) => {
    if (!modelList || modelList.length === 0) return 0;
    let bi = 0, bs = -Infinity;
    modelList.forEach((m, i) => {
      const s = m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || -Infinity);
      if (s > bs) { bs = s; bi = i; }
    });
    return bi;
  }, []);
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
  const trainingResultRef = useRef(null);
  const unsupervisedResultRef = useRef(null);
  const [showGuide, setShowGuide] = useState(false);
  const [hasSavedOnce, setHasSavedOnce] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [leaderboardEntries, setLeaderboardEntries] = useState([]);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [viewOnlyMode, setViewOnlyMode] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [shareCopyStatus, setShareCopyStatus] = useState(''); // 'copied' | 'manual' | ''
  const [treeModalOpen, setTreeModalOpen] = useState(false);
  const [treeModalAlgo, setTreeModalAlgo] = useState(null);
  const [exportLoading, setExportLoading] = useState('');
  const [preprocessConfig, setPreprocessConfig] = useState(getDefaultPreprocessConfig());
  const [preprocessLog, setPreprocessLog] = useState([]);

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

  // Keep refs in sync for use in callbacks without stale closures
  useEffect(() => { trainingResultRef.current = trainingResult; }, [trainingResult]);
  useEffect(() => { unsupervisedResultRef.current = unsupervisedResult; }, [unsupervisedResult]);

  // ==================== SESSION PERSISTENCE ====================
  const SESSION_KEY = 'automl_session';
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const saveTimerRef = useRef(null);

  // Load full session on mount
  useEffect(() => {
    isRestoringRef.current = true;
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
          if (p && s.datasetFileName) p.fileName = s.datasetFileName;
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
    setTimeout(() => { isRestoringRef.current = false; }, 500);
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
  const datasetFileName = dataProfile?.fileName || null;
  useEffect(() => {
    if (!sessionLoaded) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      try {
        localStorage.setItem(SESSION_KEY, JSON.stringify({
          csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
          datasetFileName,
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
  }, [sessionLoaded, csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan, datasetFileName,
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
    setPreprocessConfig(getDefaultPreprocessConfig()); setPreprocessLog([]);
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
      const res = await fetch(`${API_URL}/api/snapshots`, { headers: { 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` } });
      const data = await res.json();
      setHistoryList(data.snapshots || []);
    } catch { setHistoryList([]); }
    setHistoryLoading(false);
  }, []);

  const fetchLeaderboard = useCallback(async () => {
    setLeaderboardLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/leaderboard`, { headers: { 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` } });
      const data = await res.json();
      setLeaderboardEntries(data.entries || []);
    } catch { setLeaderboardEntries([]); }
    setLeaderboardLoading(false);
  }, []);

  const saveToLeaderboard = useCallback(async (modelEntry) => {
    try {
      await fetch(`${API_URL}/api/leaderboard`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` },
        body: JSON.stringify(modelEntry),
      });
    } catch (e) { console.warn('Leaderboard save failed:', e); }
  }, []);

  const deleteLeaderboardEntry = useCallback(async (modelId) => {
    try {
      await fetch(`${API_URL}/api/leaderboard/${modelId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` },
      });
      setLeaderboardEntries(prev => prev.filter(e => e.model_id !== modelId));
    } catch (e) { console.warn('Leaderboard delete failed:', e); }
  }, []);

  const clearLeaderboard = useCallback(async () => {
    try {
      await fetch(`${API_URL}/api/leaderboard`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` },
      });
      setLeaderboardEntries([]);
    } catch (e) { console.warn('Leaderboard clear failed:', e); }
  }, []);

  const computeFingerprint = useCallback(() => {
    const dsName = dataProfile?.fileName || String(dataProfile?.rowCount || 0) + 'x' + String(dataProfile?.columnCount || 0);
    const target = targetColumn || 'unsupervised';
    const modelKeys = (trainingResult?.leaderboard || []).map(m => m.algorithm).sort().join(',')
      || (unsupervisedResult?.algorithm || 'none');
    return `${dsName}|${target}|${modelKeys}|${evalMode}`;
  }, [dataProfile, targetColumn, trainingResult, unsupervisedResult, evalMode]);

  const saveInProgressRef = useRef(false);
  const handleSaveAnalysisRef = useRef(null);
  const handleSaveAnalysis = useCallback(async (customName, forceNew) => {
    if (!trainingResult && !unsupervisedResult) { setError('Nothing to save — train a model first.'); return; }
    if (saveInProgressRef.current) return; // prevent concurrent saves
    const fingerprint = computeFingerprint();

    // Skip if exact same analysis was already saved (unless forced by Share or manual Save)
    if (!forceNew && !customName && fingerprint === lastSavedFingerprintRef.current) return;

    saveInProgressRef.current = true;
    const name = customName || `${dataProfile?.fileName ? dataProfile.fileName + ' — ' : ''}${targetColumn || 'Unsupervised'} — ${new Date().toLocaleString()}`;
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
        key_metrics: bestModel?.metrics || bestModel?.testMetrics || {},
        fingerprint: forceNew ? null : fingerprint,
        state: {
          csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
          trainingResult, predictionResult, predictionHistory, selectedModelIdx,
          unsupervisedResult, clusterResult, anomalyResult, batchResults,
          shapGlobal, shapBeeswarm, shapLocal, shapDependence,
          limeResult, limeProbs, clusterShap, clusterBeeswarm,
          shapSummary, featureVsPred, clusterComparison, models,
        },
      };
      const res = await fetch(`${API_URL}/api/snapshots`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` }, body: JSON.stringify(body) });
      const data = await res.json();
      if (data.snapshot_id) {
        lastSavedFingerprintRef.current = fingerprint;
        setError(''); fetchHistory();
        if (customName || forceNew) toast.success(data.action === 'updated' ? 'Analysis updated!' : 'Analysis saved!');
        return data.snapshot_id;
      }
      else { toast.error('Save failed: ' + (data.detail || 'Unknown error')); }
    } catch (e) { toast.error('Save failed: ' + e.message); }
    finally { saveInProgressRef.current = false; }
    return null;
  }, [csvText, targetColumn, algorithm, evalMode, cleaningLog, precleanScan,
    trainingResult, predictionResult, predictionHistory, selectedModelIdx,
    unsupervisedResult, clusterResult, anomalyResult, batchResults,
    shapGlobal, shapBeeswarm, shapLocal, shapDependence,
    limeResult, limeProbs, clusterShap, clusterBeeswarm,
    shapSummary, featureVsPred, clusterComparison, models, dataProfile, fetchHistory, computeFingerprint]);
  handleSaveAnalysisRef.current = handleSaveAnalysis;
  // Track save milestone for onboarding
  useEffect(() => { if (historyList?.length > 0) setHasSavedOnce(true); }, [historyList]);

  const handleLoadSnapshot = useCallback(async (snapshotId) => {
    try {
      isRestoringRef.current = true;
      const res = await fetch(`${API_URL}/api/snapshots/${snapshotId}`);
      const data = await res.json();
      if (!data.snapshot?.state) { setError('Snapshot not found or corrupted.'); return; }
      const s = data.snapshot.state;
      if (s.csvText) { setCsvText(s.csvText); const p = profileDataset(s.csvText); if (p && data.snapshot.dataset_name) p.fileName = data.snapshot.dataset_name; setDataProfile(p); setColumns(p?.headers || []); }
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
      // Reset restoring flag after state settles
      setTimeout(() => { isRestoringRef.current = false; }, 500);
    } catch (e) { setError('Failed to load snapshot: ' + e.message); isRestoringRef.current = false; }
  }, []);

  const handleDeleteSnapshot = useCallback(async (snapshotId) => {
    try {
      await fetch(`${API_URL}/api/snapshots/${snapshotId}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${localStorage.getItem('automl_token') || ''}` } });
      fetchHistory();
    } catch (e) { setError('Delete failed: ' + e.message); }
  }, [fetchHistory]);

  const handleShareAnalysis = useCallback(async () => {
    setExportLoading('share');
    try {
      let sid = await handleSaveAnalysis(null, true);
      if (sid) {
        const url = `${window.location.origin}${window.location.pathname}?snapshot=${sid}`;
        setShareUrl(url);
        const ok = await safeCopyToClipboard(url);
        setShareCopyStatus(ok ? 'copied' : 'manual');
        if (ok) toast.success('Share link copied to clipboard!');
        else toast.info('Share link generated — please copy it manually.');
      } else { toast.error('Failed to generate share link.'); }
    } catch { toast.error('Share failed. Please try again.'); setShareCopyStatus('manual'); }
    finally { setExportLoading(''); }
  }, [handleSaveAnalysis, safeCopyToClipboard]);

  // ==================== AUTO-SAVE AFTER TRAINING ====================
  const autoSaveTimerRef = useRef(null);
  useEffect(() => {
    if (isRestoringRef.current || !sessionLoaded) return;
    if (trainingResult && trainingResult.status === 'success') {
      // Debounce auto-save to prevent rapid re-runs creating duplicates
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = setTimeout(() => handleSaveAnalysis(), 800);
    }
    return () => { if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current); };
  }, [trainingResult]);

  useEffect(() => {
    if (isRestoringRef.current || !sessionLoaded) return;
    if (unsupervisedResult) {
      if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = setTimeout(() => handleSaveAnalysis(), 800);
    }
    return () => { if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current); };
  }, [unsupervisedResult]);

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
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      if (data.token) {
        const downloadUrl = `${API_URL}/api/export/download/${data.token}`;
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        document.body.appendChild(iframe);
        iframe.src = downloadUrl;
        setTimeout(() => { try { document.body.removeChild(iframe); } catch {} }, 30000);
        toast.success(`Download started: ${filename}`);
        return true;
      } else { toast.error('Export failed — could not prepare download.'); return false; }
    } catch (e) { toast.error('Export failed: ' + e.message); return false; }
  }, []);

  const handleExportCSV = useCallback(async () => {
    if (!trainingResult && !shapGlobal && !limeResult && !predictionHistory?.length) { toast.error('Nothing to export — train a model first.'); return; }
    setExportLoading('csv');
    try {
      const csv = buildFullCSV();
      await triggerBackendDownload(csv, `analysis_data_${Date.now()}.csv`);
    } catch (e) { toast.error('CSV export failed: ' + e.message); }
    finally { setExportLoading(''); }
  }, [trainingResult, shapGlobal, limeResult, predictionHistory, buildFullCSV, triggerBackendDownload]);

  const handleExportSheets = useCallback(async () => {
    if (!trainingResult && !shapGlobal && !limeResult && !predictionHistory?.length) { toast.error('Nothing to export — train a model first.'); return; }
    setExportLoading('sheets');
    try {
      const csv = buildFullCSV();
      await triggerBackendDownload(csv, `analysis_for_google_sheets_${Date.now()}.csv`);
    } catch (e) { toast.error('Google Sheets export failed: ' + e.message); }
    finally { setExportLoading(''); }
  }, [trainingResult, shapGlobal, limeResult, predictionHistory, buildFullCSV, triggerBackendDownload]);

  const handleExportPDF = useCallback(async () => {
    if (!trainingResult && !unsupervisedResult && !dataProfile) { toast.error('Nothing to report — load data and run analysis first.'); return; }
    setExportLoading('pdf');
    try {
      // Fetch deployments for the report
      let deployments = [];
      try {
        const token = localStorage.getItem('automl_token') || '';
        const res = await fetch(`${API_URL}/api/deploy`, { headers: { 'Authorization': `Bearer ${token}` } });
        if (res.ok) { const d = await res.json(); deployments = d.deployments || []; }
      } catch {} // silent — deployments are optional

      const filename = await generateReport({
        dataProfile, trainingResult, models, targetColumn, evalMode,
        shapGlobal, limeResult, predictionHistory, unsupervisedResult,
        clusterResult, anomalyResult, leaderboardEntries, deployments,
        authUser, preprocessConfig,
      });
      toast.success(`Report downloaded: ${filename}`);
    } catch (e) { toast.error('PDF generation failed: ' + e.message); console.error(e); }
    finally { setExportLoading(''); }
  }, [dataProfile, trainingResult, models, targetColumn, evalMode, shapGlobal, limeResult,
    predictionHistory, unsupervisedResult, clusterResult, anomalyResult, leaderboardEntries, authUser, preprocessConfig]);

  const handleDownloadSnapshotPDF = useCallback(async (snapshotId) => {
    try {
      toast.info('Generating PDF from saved analysis...');
      const res = await fetch(`${API_URL}/api/snapshots/${snapshotId}`);
      const data = await res.json();
      if (!data.snapshot?.state) { toast.error('Snapshot not found or corrupted.'); return; }
      const s = data.snapshot.state;
      const dp = s.csvText ? profileDataset(s.csvText) : null;
      if (dp && data.snapshot.dataset_name) dp.fileName = data.snapshot.dataset_name;
      const filename = await generateReport({
        dataProfile: dp,
        trainingResult: s.trainingResult || null,
        models: s.models || [],
        targetColumn: s.targetColumn || null,
        evalMode: s.evalMode || 'split',
        shapGlobal: s.shapGlobal || null,
        limeResult: s.limeResult || null,
        predictionHistory: s.predictionHistory || [],
        unsupervisedResult: s.unsupervisedResult || null,
        clusterResult: s.clusterResult || null,
        anomalyResult: s.anomalyResult || null,
        leaderboardEntries: [],
        deployments: [],
        authUser,
        preprocessConfig: null,
      });
      toast.success(`Report downloaded: ${filename}`);
    } catch (e) { toast.error('PDF generation failed: ' + e.message); console.error(e); }
  }, [authUser]);

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
  useEffect(() => { if (activeView === 'leaderboard' || activeView === 'dashboard') fetchLeaderboard(); }, [activeView, fetchLeaderboard]);

  // ==================== SAMPLE DATASETS ====================
  const sampleDatasets = [
    { name: 'Loan Approval', description: 'Classification — 1,200 rows, 12 features', file: '/api/sample_data/loan_approval.csv' },
    { name: 'House Prices', description: 'Regression — 1,000 rows, 12 features', file: '/api/sample_data/house_prices.csv' },
    { name: 'Insurance Costs', description: 'Regression — 1,100 rows, 9 features', file: '/api/sample_data/insurance.csv' },
    { name: 'Customer Churn', description: 'Classification — 1,500 rows, 11 features', file: '/api/sample_data/customer_churn.csv' },
    { name: 'Customer Segmentation', description: 'Unsupervised — 1,000 rows, 10 features', file: '/api/sample_data/customer_segmentation.csv' },
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
    const financialKeywords = ['charges', 'charge', 'cost', 'price', 'amount', 'salary', 'income', 'revenue', 'profit', 'loss', 'fee', 'premium', 'insurance', 'payment', 'spend', 'expense', 'total', 'sales'];
    const targetKeywords = ['target', 'label', 'class', 'output', 'result', 'outcome', 'status', 'approved', 'churned', 'survived', 'default'];
    let bestCol = null, bestScore = -1;
    for (const c of cols) {
      const nameLower = c.name.toLowerCase();
      let score = 0;
      const matchesKeyword = targetKeywords.some(k => nameLower.includes(k));
      const matchesFinancial = financialKeywords.some(k => nameLower.includes(k));
      // Strong signal: column name matches common target/financial keywords
      if (matchesKeyword) score += 20;
      if (matchesFinancial && c.type === 'numeric') score += 25;
      // Classification targets: categorical with 2-10 classes
      if (c.type === 'categorical' && c.uniqueCount >= 2 && c.uniqueCount <= 10) {
        score += 10 + (10 - c.uniqueCount);
      }
      // Regression targets: numeric with high variability and many unique values
      else if (c.type === 'numeric' && c.uniqueCount > 10) {
        score += 5 + Math.min(5, c.std / (Math.abs(c.mean) || 1));
        // Bonus: continuous numeric with many unique values is a strong regression target signal
        if (c.uniqueCount > dataProfile.rowCount * 0.5) score += 8;
      }
      // Penalize ID-like, name-like, or index columns
      if (['id', 'index', 'name', 'date', 'timestamp', 'key'].some(k => nameLower.includes(k))) score -= 30;
      // Penalize columns with too many unique values ONLY if non-numeric or no keyword match
      // Numeric columns matching financial/target keywords are likely valid targets (e.g., charges, price)
      if (c.uniqueCount > dataProfile.rowCount * 0.8 && !(c.type === 'numeric' && (matchesKeyword || matchesFinancial))) {
        score -= 20;
      }
      if (score > bestScore) { bestScore = score; bestCol = c; }
    }
    if (!bestCol || bestScore <= 0) return null;
    const task = bestCol.type === 'categorical' ? 'classification' : 'regression';
    return { name: bestCol.name, task, reason: bestCol.type === 'categorical' ? `Categorical with ${bestCol.uniqueCount} classes` : `Numeric with high variability (std: ${bestCol.std?.toFixed(2)})` };
  }, [dataProfile, targetColumn]);

  // Smart XAI row suggestions: suggest interesting rows for explanation
  const smartRowSuggestions = useMemo(() => {
    if (!models.length || !dataProfile) return [];
    const model = models[selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : getBestModelIdx(models)];
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
  }, [models, selectedModelIdx, dataProfile, getBestModelIdx]);

  // ==================== XAI MODEL HELPER (moved up to avoid hoisting issues) ====================
  const bestXaiModelIdx = useMemo(() => {
    if (models.length === 0) return -1;
    let bestIdx = 0;
    let bestScore = -Infinity;
    models.forEach((m, i) => {
      const score = m.problemType === 'classification'
        ? (m.metrics?.accuracy ?? m.metrics?.f1 ?? 0)
        : (m.metrics?.r2 ?? 0);
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    });
    return bestIdx;
  }, [models]);

  const getXaiModel = useCallback(() => {
    if (models.length === 0) return null;
    if (selectedModelIdx >= 0 && selectedModelIdx < models.length) return models[selectedModelIdx];
    return bestXaiModelIdx >= 0 ? models[bestXaiModelIdx] : models[models.length - 1];
  }, [models, selectedModelIdx, bestXaiModelIdx]);

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
      const o1 = optimalRanges[top2[0].feature] || {};
      actionItems.push(`Focus on "${top2[0].feature}" — it has the strongest influence on ${target}. ${o1.recommendation === 'higher' ? 'Increasing' : 'Decreasing'} this feature tends to improve outcomes.`);
    }
    if (top2[1]) {
      const o = optimalRanges[top2[1].feature] || {};
      actionItems.push(`Optimize "${top2[1].feature}" as the second most impactful factor. ${o.recommendation === 'higher' ? 'Higher' : 'Lower'} values of this feature are associated with better ${target} outcomes.`);
    }
    if (correlation && Math.abs(correlation.r) > 0.3) {
      actionItems.push(`"${correlation.feature1}" and "${correlation.feature2}" have a ${correlation.strength} ${correlation.dir} relationship (r=${correlation.r.toFixed(2)}). Changes in one may affect the other.`);
    }

    return { topN, top2, top5, directions, optimalRanges, correlation, pairScatter, actionItems, target };
  }, [shapGlobal, shapSummary, shapBeeswarm, dataProfile, targetColumn, getXaiModel]);

  // ==================== BUSINESS INTERPRETATION ====================
  const businessInterpretation = useMemo(() => {
    if (!trainingResult || trainingResult.status !== 'success') return null;
    const lb = trainingResult.leaderboard || [];
    if (!lb.length) return null;
    const best = lb[0];
    const worst = lb.length > 1 ? lb[lb.length - 1] : null;
    const pt = trainingResult.problemType;
    const isReg = pt === 'regression';
    const bestScore = isReg ? (best?.testMetrics?.r2 || 0) : (best?.testMetrics?.accuracy || 0);
    const worstScore = worst ? (isReg ? (worst?.testMetrics?.r2 || 0) : (worst?.testMetrics?.accuracy || 0)) : null;
    const target = targetColumn || 'target';
    const domain = datasetSummary?.domain || 'your business';
    const domainLower = domain.toLowerCase();
    const rowCount = dataProfile?.rowCount || 0;

    // Key features from SHAP if available
    const topFeats = featureInsights?.top2?.map(f => f.feature) || [];
    const topDir = featureInsights?.optimalRanges || {};

    // Build interpretation lines
    const lines = [];

    // 1) Key insight
    if (isReg) {
      const pct = (bestScore * 100).toFixed(0);
      lines.push(bestScore > 0.7
        ? `Your data shows clear, predictable patterns in ${target} — the model can explain ${pct}% of the variation, meaning most of what drives ${target} is captured in your data.`
        : `The ${target} values show some unpredictable behavior — only ${pct}% of the variation is explained, suggesting external factors or noise play a role.`);
    } else {
      const pct = (bestScore * 100).toFixed(0);
      lines.push(bestScore > 0.85
        ? `The analysis reveals strong, distinguishable patterns in your ${domainLower} data — the model correctly identifies ${target} outcomes ${pct}% of the time.`
        : bestScore > 0.65
        ? `There are moderate patterns in your ${domainLower} data, with ${pct}% of ${target} outcomes predicted correctly. Some cases remain ambiguous.`
        : `Predicting ${target} is challenging with the current data — only ${pct}% accuracy, indicating the factors captured don't fully explain the outcome.`);
    }

    // 2) What's performing well
    const bestName = best?.algorithm?.replace(/_/g, ' ') || 'the top model';
    lines.push(`${bestName.charAt(0).toUpperCase() + bestName.slice(1)} performs best for this task, making it the most reliable approach for predicting ${target} in ${domainLower}.`);

    // 3) What's not performing well
    if (worst && worstScore !== null && worstScore < bestScore - 0.05) {
      const worstName = worst?.algorithm?.replace(/_/g, ' ') || 'the weakest model';
      lines.push(`${worstName.charAt(0).toUpperCase() + worstName.slice(1)} struggles with this data — likely because it can't capture the non-linear relationships between your variables.`);
    } else if (lb.length === 1) {
      lines.push(`Only one approach was tested. Training additional models could reveal whether a different method better captures the data patterns.`);
    } else {
      lines.push(`All tested approaches perform similarly, suggesting the data patterns are straightforward and well-captured regardless of method.`);
    }

    // 4) Why + 5) Actions
    if (topFeats.length >= 2) {
      const f1 = topFeats[0], f2 = topFeats[1];
      const d1 = topDir[f1]?.recommendation;
      const d2 = topDir[f2]?.recommendation;
      lines.push(`The biggest drivers are "${f1}" and "${f2}" — together they account for most of the variation in ${target}.`);
      const action1 = d1 === 'higher' ? `invest in increasing ${f1}` : `work on reducing ${f1}`;
      const action2 = d2 === 'higher' ? `prioritize growing ${f2}` : `focus on lowering ${f2}`;
      lines.push(`To improve outcomes: ${action1}, and ${action2}. These two levers will have the most direct impact on your ${domainLower} results.`);
    } else {
      lines.push(`With ${rowCount} records analyzed, the patterns are ${bestScore > 0.7 ? 'reliable' : 'emerging but not definitive'} — ${bestScore > 0.7 ? 'you can confidently act on these findings' : 'consider collecting more data or additional variables to strengthen insights'}.`);
      lines.push(`Run SHAP analysis in the Explainability tab to identify exactly which factors drive ${target} and get specific, actionable recommendations.`);
    }

    return lines;
  }, [trainingResult, targetColumn, datasetSummary, dataProfile, featureInsights]);

  // ==================== DATA HANDLERS ====================
  const handleCsvTextChange = useCallback((text, isCleanAction, fileName) => {
    // When loading a NEW dataset (not a cleaning action), auto-save current analysis and reset all state
    if (!isCleanAction) {
      // Auto-save current analysis to History before switching
      if (trainingResultRef.current || unsupervisedResultRef.current) {
        try { handleSaveAnalysisRef.current?.(); } catch {}
        toast.info('Previous analysis saved to History');
      }
      // Reset all analysis-derived state for the new dataset
      setTargetColumn(''); setAlgorithm('auto');
      setModels([]); setPredictionFormData({}); setPredictionResult(null);
      setPredictionHistory([]); setSelectedModelIdx(-1);
      setUnsupervisedResult(null); setClusterPredFormData({}); setClusterPredResult(null);
      setBatchCsvText(''); setBatchResults(null);
      setShapGlobal(null); setShapBeeswarm(null); setShapLocal(null); setShapDependence(null);
      setLimeResult(null); setLimeProbs(null);
      setClusterShap(null); setClusterBeeswarm(null); setShapSummary(null);
      setFeatureVsPred(null); setClusterComparison(null);
      setCorrVarX(''); setCorrVarY(''); setHistogramCol('');
      setXaiTab('shap'); setXaiRow(0); setXaiDepFeature(0);
      xaiCacheRef.current = {};
      lastSavedFingerprintRef.current = null;
      setCleaningLog([]); setPrecleanScan(null);
    }
    setCsvText(text); setTrainingResult(null); setClusterResult(null); setAnomalyResult(null);
    if (text.trim()) {
      const p = profileDataset(text);
      if (p && fileName) p.fileName = fileName;
      setDataProfile(p); setColumns(p?.headers || []);
      if (!isCleanAction && p) toast.success(`Dataset uploaded successfully — ${p.rowCount} rows, ${p.headers.length} columns`);
    }
    else { setDataProfile(null); setColumns([]); }
  }, []);
  const loadSampleDataset = useCallback(async (sample) => {
    if (sample.data) { handleCsvTextChange(sample.data, false, sample.name); }
    else if (sample.file) {
      try { const res = await fetch(`${API_URL}${sample.file}`); if (!res.ok) throw new Error('Failed to load'); handleCsvTextChange(await res.text(), false, sample.name); }
      catch (e) { setError('Failed to load sample dataset: ' + e.message); }
    }
  }, [handleCsvTextChange]);
  const handleFileUpload = (event) => { const file = event.target.files[0]; if (file) { const reader = new FileReader(); reader.onload = (e) => handleCsvTextChange(e.target.result, false, file.name); reader.readAsText(file); } };
  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover'); };
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); const f = e.dataTransfer.files?.[0]; if (f) { const reader = new FileReader(); reader.onload = (ev) => handleCsvTextChange(ev.target.result, false, f.name); reader.readAsText(f); } };

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
        let rows = dataProfile?.rows || parseCSV(csvText).rows;
        if (rows.length < 4) throw new Error('Need at least 4 rows for train-test splitting');

        // Apply preprocessing pipeline
        const ppLog = [];
        const ppHeaders = dataProfile?.headers || Object.keys(rows[0]);
        const ppNumCols = dataProfile?.numericColumns || [];

        // 1. Exclude selected features
        if (preprocessConfig.excludeFeatures?.length > 0) {
          rows = rows.map(r => {
            const nr = { ...r };
            preprocessConfig.excludeFeatures.forEach(col => { delete nr[col]; });
            return nr;
          });
          ppLog.push({ step: 'Feature Selection', message: `Excluded ${preprocessConfig.excludeFeatures.length} feature(s): ${preprocessConfig.excludeFeatures.join(', ')}` });
        }

        // 2. Handle missing values
        if (preprocessConfig.missingValues !== 'none') {
          const mv = handleMissingValues(rows, ppHeaders, ppNumCols, preprocessConfig.missingValues);
          rows = mv.rows;
          mv.log.forEach(l => ppLog.push({ step: 'Missing Values', message: `${l.col}: filled ${l.count} values using ${l.strategy} (${l.fillValue})` }));
        }

        // 3. Handle outliers
        if (preprocessConfig.outlierMethod !== 'none') {
          const ol = handleOutliers(rows, ppNumCols, preprocessConfig.outlierMethod, preprocessConfig.outlierThreshold);
          rows = ol.rows;
          ol.log.forEach(l => ppLog.push({ step: 'Outliers', message: `${l.col}: ${l.affected} values ${l.method === 'clip' ? 'clipped' : 'removed'} ${l.range}` }));
        }

        if (rows.length < 4) throw new Error('Too few rows remaining after preprocessing — try less aggressive settings');

        const prepared = prepareFeatures(rows, targetColumn);
        let { X, y, featureNames, encodingMap, numericCols, categoricalCols, textCols, targetEncoding, leakageCols, targetMin, targetMax } = prepared;
        if (featureNames.length === 0) throw new Error('No usable features after preprocessing');
        const problemType = detectProblemType(y);

        // 4. Feature scaling
        let scaleParams = null;
        if (preprocessConfig.scaling !== 'none') {
          const scaled = scaleFeatures(X, preprocessConfig.scaling);
          X = scaled.X;
          scaleParams = scaled.scaleParams;
          ppLog.push({ step: 'Scaling', message: `Applied ${preprocessConfig.scaling === 'standard' ? 'standardization (z-score)' : 'min-max normalization'} to ${featureNames.length} features` });
        }

        setPreprocessLog(ppLog);

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
        const modelData = { featureNames, numericCols, categoricalCols, encodingMap, targetEncoding, scaleParams, targetMin, targetMax, ...bestModelObj };
        // Limit KNN storage for localStorage
        if (modelData.type === 'knn' && modelData.X_train && modelData.X_train.length > 500) {
          const si = Array.from({ length: modelData.X_train.length }, (_, i) => i);
          for (let i = si.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [si[i], si[j]] = [si[j], si[i]]; }
          const kept = si.slice(0, 500);
          modelData.X_train = kept.map(i => modelData.X_train[i]);
          modelData.y_train = kept.map(i => modelData.y_train[i]);
        }

        setModels(prev => {
          const newModels = [...prev, {
            modelId: best.modelId, algorithm: best.algorithm, problemType,
            metrics: best.testMetrics, trainMetrics: best.trainMetrics,
            featureImportance: best.featureImportance,
            createdAt: new Date().toISOString(), durationSec: best.durationSec,
            evalMode, targetColumn, modelData
          }];
          // Also store tree-based models separately for visualization (if not already best)
          for (const tm of trainModels) {
            if ((tm.algo === 'decision_tree' || tm.algo === 'random_forest') && tm.algo !== best.algorithm) {
              const lbEntry = leaderboard.find(l => l.algorithm === tm.algo);
              if (lbEntry) {
                newModels.push({
                  modelId: lbEntry.modelId, algorithm: tm.algo, problemType,
                  metrics: lbEntry.testMetrics, trainMetrics: lbEntry.trainMetrics,
                  featureImportance: lbEntry.featureImportance,
                  createdAt: new Date().toISOString(), durationSec: lbEntry.durationSec,
                  evalMode, targetColumn,
                  modelData: { featureNames, numericCols, categoricalCols, encodingMap, targetEncoding, scaleParams, targetMin, targetMax, ...tm.modelObj }
                });
              }
            }
          }
          return newModels;
        });

        setTrainingResult({
          status: 'success', problemType, bestModel: best, leaderboard, evalMode,
          totalTime: (performance.now() - startTime) / 1000,
          splitInfo: { trainSize, testSize, totalSize: rows.length },
          dataInfo: { numSamples: rows.length, numFeatures: featureNames.length, targetColumn, columns: featureNames, removedLeakageColumns: leakageCols, textColumns: textCols },
          predictionsVsActual, residualStats,
          preprocessConfig: { ...preprocessConfig }, preprocessLog: ppLog,
        });

        // Auto-save all trained models to leaderboard
        const dsName = dataProfile?.fileName || csvText?.substring(0, 30) || 'Unknown Dataset';
        for (const entry of leaderboard) {
          saveToLeaderboard({
            model_id: entry.modelId,
            algorithm: entry.algorithm,
            problem_type: problemType,
            dataset_name: dsName,
            target_column: targetColumn,
            metrics: entry.testMetrics || {},
            feature_importance: entry.featureImportance || [],
            duration_sec: entry.durationSec || 0,
            eval_mode: evalMode,
            num_features: featureNames.length,
            num_samples: rows.length,
          });
        }
      } catch (err) { setError(err.message || 'Training failed'); }
      finally { setIsTraining(false); }
    }, 50);
  };

  // ==================== PREDICTION ====================
  const handlePredict = () => {
    setError(''); setPredictionResult(null);
    const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : getBestModelIdx(models);
    const am = models[idx];
    if (!am) { setError('No trained model available.'); return; }
    try {
      // Check if ALL fields are blank/empty — return 0 for regression, default class for classification
      const allNumericEmpty = am.modelData.numericCols.every(col => predictionFormData[col] === undefined || predictionFormData[col] === '' || predictionFormData[col] === null);
      const allCatEmpty = am.modelData.categoricalCols.every(col => !predictionFormData[col] || predictionFormData[col] === '');
      if (allNumericEmpty && allCatEmpty) {
        const zeroPred = am.problemType === 'regression' ? 0 : 0;
        setPredictionResult({ status: 'success', modelId: am.modelId, algorithm: am.algorithm, predictions: [zeroPred], probabilities: null, problemType: am.problemType, inputData: {} });
        setPredictionHistory(prev => [...prev, { id: Date.now(), type: 'supervised', model: ALGO_NAMES[am.algorithm] || am.algorithm, target: am.targetColumn, prediction: zeroPred, input: {}, timestamp: Date.now() }]);
        toast.warning('All input fields are empty — prediction defaults to 0.');
        return;
      }
      const row = {};
      am.modelData.numericCols.forEach(col => { row[col] = Number(predictionFormData[col]) || 0; });
      am.modelData.categoricalCols.forEach(col => { row[col] = predictionFormData[col] || ''; });
      const fvs = prepareInputForPrediction([row], am.modelData);
      const predictions = fvs.map(x => {
        let raw = predictOne(am.modelData, x);
        raw = am.modelData.targetEncoding ? am.modelData.targetEncoding[raw] : raw;
        // Clamp regression predictions: prevent nonsensical negatives when target is naturally non-negative
        if (am.problemType === 'regression' && typeof raw === 'number' && am.modelData.targetMin !== undefined && am.modelData.targetMin >= 0) {
          raw = Math.max(0, raw);
        }
        return raw;
      });
      const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
      const probabilities = am.modelData.type === 'logistic_regression' ? fvs.map(x => { const z = am.modelData.coefficients[0] + x.reduce((s, v, i) => s + v * am.modelData.coefficients[i + 1], 0); const p = sigmoid(z); return [1 - p, p]; }) : null;
      setPredictionResult({ status: 'success', modelId: am.modelId, algorithm: am.algorithm, predictions, probabilities, problemType: am.problemType, inputData: row });
      setPredictionHistory(prev => [...prev, { id: Date.now(), type: 'supervised', model: ALGO_NAMES[am.algorithm] || am.algorithm, target: am.targetColumn, prediction: predictions[0], input: { ...row }, timestamp: Date.now() }]);
    } catch (err) { setError('Prediction failed: ' + err.message); }
  };

  // ==================== CLUSTERING & ANOMALY ====================
  const handleClustering = () => { setError(''); setClusterResult(null); if (!dataProfile) { setError('Please upload data first'); return; } if (dataProfile.numericColumns.length < 1) { setError('Need numeric columns'); return; } try { setClusterResult(runKMeansClustering(dataProfile.rows, dataProfile.numericColumns, numClusters, kmeans)); } catch (err) { setError('Clustering failed: ' + err.message); } };
  const handleAnomalyDetection = () => { setError(''); setAnomalyResult(null); if (!dataProfile) { setError('Please upload data first'); return; } if (dataProfile.numericColumns.length < 1) { setError('Need numeric columns'); return; } try { setAnomalyResult(detectAnomaliesFunc(dataProfile.rows, dataProfile.numericColumns, anomalyMethod, anomalyThreshold)); } catch (err) { setError('Anomaly detection failed: ' + err.message); } };

  // ==================== BATCH PREDICTIONS ====================
  const handleBatchPredict = () => {
    setError(''); setBatchResults(null);
    const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : getBestModelIdx(models);
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
      <div className="flex gap-2 flex-wrap justify-center">{sampleDatasets.slice(0, 3).map((ds, i) => <Button key={i} variant="outline" size="sm" onClick={() => loadSampleDataset(ds)} data-testid={`mini-sample-${i}`}>{ds.name}</Button>)}</div>
    </div></CardContent></Card>
  );

  // Metric display helper with color coding and tooltips
  const MetricCard = ({ label, value, score, metricKey }) => {
    const explanation = METRIC_EXPLANATIONS[metricKey];
    const color = score !== undefined ? getScoreColor(score, explanation?.higherBetter !== false) : null;
    const interp = metricKey ? interpretMetric(metricKey, score !== undefined ? score : parseFloat(String(value))) : null;
    const tipRef = React.useRef(null);
    const trigRef = React.useRef(null);
    React.useEffect(() => {
      const el = tipRef.current; const tr = trigRef.current;
      if (!el || !tr) return;
      const handleEnter = () => { const rect = tr.getBoundingClientRect(); const vw = window.innerWidth;
        el.style.left = ''; el.style.right = '';
        if (rect.right + 280 > vw) { el.style.right = '0'; el.style.left = 'auto'; }
        else { el.style.right = '0'; }
      };
      tr.addEventListener('mouseenter', handleEnter);
      return () => tr.removeEventListener('mouseenter', handleEnter);
    }, []);
    return (
      <div className={`rounded-xl p-4 border-2 transition-all ${color ? `${color.bg} ${color.border}` : 'bg-muted/50 border-transparent'}`} data-testid={`metric-${metricKey || label}`}>
        <div className="flex items-center justify-between mb-1">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">{label}</p>
          {explanation && (
            <div className="group relative" ref={trigRef}>
              <Info className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help hover:text-foreground transition-colors" />
              <div ref={tipRef} className="invisible group-hover:visible opacity-0 group-hover:opacity-100 absolute z-[9999] bottom-full right-0 mb-2 w-72 p-3 rounded-lg bg-popover border shadow-lg text-xs text-popover-foreground transition-all duration-200 pointer-events-none">
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

  // ==================== DECISION TREE VIEWER ====================
  const TreeNode = ({ node, featureNames, depth = 0, maxDepth = 5 }) => {
    if (!node) return null;
    const isLeaf = node.leaf;
    if (depth > maxDepth) return <div className="text-xs text-muted-foreground italic px-2 py-1">...</div>;
    const featName = !isLeaf && featureNames?.[node.feature] ? featureNames[node.feature] : `Feature ${node.feature}`;
    return (
      <div className="flex flex-col items-center" data-testid={`tree-node-depth-${depth}`}>
        <div className={`rounded-lg border-2 px-3 py-2 text-xs text-center shadow-sm transition-all hover:shadow-md ${isLeaf ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-700' : 'bg-blue-50 dark:bg-blue-950/30 border-blue-300 dark:border-blue-700'}`} style={{ minWidth: 100 }}>
          {isLeaf ? (
            <><span className="block font-bold text-emerald-700 dark:text-emerald-400">Result</span><span className="font-mono font-bold">{typeof node.value === 'number' ? node.value.toFixed(2) : node.value}</span><span className="block text-muted-foreground">{node.n} samples</span></>
          ) : (
            <><span className="block font-semibold text-blue-700 dark:text-blue-400 truncate max-w-[120px]" title={featName}>{featName}</span><span className="font-mono">{'\u2264'} {node.threshold?.toFixed(2)}</span><span className="block text-muted-foreground">{node.n} samples</span></>
          )}
        </div>
        {!isLeaf && (
          <div className="flex items-start gap-4 mt-2 pt-2 relative">
            <div className="absolute top-0 left-1/2 w-px h-2 bg-border" />
            <div className="absolute top-2 left-[25%] right-[25%] h-px bg-border" />
            <div className="flex flex-col items-center relative">
              <div className="w-px h-2 bg-border" />
              <span className="text-[10px] text-green-600 dark:text-green-400 font-semibold mb-1">Yes</span>
              <TreeNode node={node.left} featureNames={featureNames} depth={depth + 1} maxDepth={maxDepth} />
            </div>
            <div className="flex flex-col items-center relative">
              <div className="w-px h-2 bg-border" />
              <span className="text-[10px] text-red-500 dark:text-red-400 font-semibold mb-1">No</span>
              <TreeNode node={node.right} featureNames={featureNames} depth={depth + 1} maxDepth={maxDepth} />
            </div>
          </div>
        )}
      </div>
    );
  };

  const getDecisionTreeModel = useCallback(() => {
    if (!treeModalAlgo) return null;
    // Find model with matching algorithm that has a tree structure
    const m = models.find(m => m.algorithm === treeModalAlgo && m.modelData?.tree);
    if (m) return m;
    // For random forest, show first individual tree
    const rf = models.find(m => m.algorithm === treeModalAlgo && m.modelData?.trees?.length > 0);
    if (rf) return { ...rf, modelData: { ...rf.modelData, tree: rf.modelData.trees[0] } };
    return null;
  }, [treeModalAlgo, models]);

  // ==================== RENDER ====================
  const ctx = {
    // State
    activeView, setActiveView, error, setError, dragActive, csvText, setCsvText, columns,
    dataProfile, targetColumn, setTargetColumn, algorithm, setAlgorithm, evalMode, setEvalMode,
    cleaningLog, precleanScan, isTraining, trainingResult, setTrainingResult, models, setModels,
    predictionFormData, setPredictionFormData, predictionResult, setPredictionResult,
    numClusters, setNumClusters, clusterResult, anomalyMethod, setAnomalyMethod,
    anomalyThreshold, setAnomalyThreshold, anomalyResult, unsupervisedResult,
    isRunningUnsupervised, clusterPredFormData, setClusterPredFormData, clusterPredResult,
    predictTab, setPredictTab, predictionHistory, selectedModelIdx, setSelectedModelIdx,
    setUnsupervisedResult,
    corrVarX, setCorrVarX, corrVarY, setCorrVarY, darkMode, setDarkMode,
    batchCsvText, setBatchCsvText, batchResults, batchProcessing,
    histogramCol, setHistogramCol, xaiTab, setXaiTab, xaiLoading, xaiRow, setXaiRow,
    xaiDepFeature, setXaiDepFeature, shapGlobal, shapBeeswarm, shapLocal, shapDependence,
    limeResult, limeProbs, clusterShap, clusterBeeswarm, shapSummary, featureVsPred,
    clusterComparison, showGuide, setShowGuide, historyList, historyLoading, viewOnlyMode,
    leaderboardEntries, leaderboardLoading, fetchLeaderboard,
    deleteLeaderboardEntry, clearLeaderboard,
    setViewOnlyMode, shareUrl, setShareUrl, shareCopyStatus, setShareCopyStatus,
    treeModalOpen, setTreeModalOpen, treeModalAlgo, setTreeModalAlgo, exportLoading,
    preprocessConfig, setPreprocessConfig, preprocessLog,
    // Computed
    stats, topModels, taskSuggestion, datasetSummary, suggestedTarget, smartRowSuggestions,
    bestXaiModelIdx, datasetScan, corrMatrix, histogramData, businessInterpretation,
    featureInsights, dataPreview,
    // Handlers
    handleCsvTextChange, loadSampleDataset, handleFileUpload, handleDrag, handleDrop,
    handleClean, handleTrain, handlePredict, handleClustering, handleAnomalyDetection,
    handleBatchPredict, handleBatchCsvUpload, downloadBatchCsv, handleDeleteModel,
    handleDownloadModel, handleImportModel, handleRunSHAP, handleRunLIME,
    handleExplainPrediction, handleClusterExplain, handleRunUnsupervised, handleClusterPredict,
    fetchHistory, handleSaveAnalysis, handleLoadSnapshot, handleDeleteSnapshot,
    handleShareAnalysis, handleExportCSV, handleExportSheets, handleClearSession,
    handleDownloadSnapshotPDF, safeCopyToClipboard, getXaiModel, getDecisionTreeModel,
    // Sub-components
    StatCard, MetricCard, DataUploadMini, TreeNode,
    // Auth
    authUser, onLogout,
    // Sample data
    sampleDatasets,
  };

  return (
    <AppContext.Provider value={ctx}>
    <div className="min-h-screen bg-background" data-testid="app-root">
      <Toaster position="top-right" richColors closeButton />
      <OnboardingGuide
        setActiveView={setActiveView}
        csvText={csvText}
        trainingResult={trainingResult}
        unsupervisedResult={unsupervisedResult}
        predictionHistory={predictionHistory}
        shapGlobal={shapGlobal}
        limeResult={limeResult}
        hasSavedOnce={hasSavedOnce}
      />
      <motion.aside initial={{ x: -300 }} animate={{ x: 0 }} className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-sidebar" data-testid="app-sidebar">
        <div className="flex h-full flex-col gap-2">
          <div className="flex h-16 items-center border-b border-sidebar-border px-6"><div className="flex items-center gap-2"><div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground"><Brain className="h-6 w-6" /></div><div><h1 className="text-lg font-bold text-sidebar-foreground">AutoML</h1><p className="text-xs text-sidebar-foreground/60">Universal Dashboard</p></div></div></div>
          <nav className="flex-1 space-y-1 px-3 py-4" data-testid="sidebar-nav">
            {[{ id: 'dashboard', label: 'Dashboard', icon: Activity }, { id: 'analysis', label: 'Analysis', icon: Zap }, { id: 'preprocess', label: 'Preprocess', icon: Settings2 }, { id: 'predict', label: 'Predictions', icon: Sparkles }, { id: 'explainability', label: 'Explainability', icon: Eye }, { id: 'compare', label: 'Compare', icon: GitBranch }, { id: 'tune', label: 'Tune', icon: SlidersHorizontal }, { id: 'whatif', label: 'What-If', icon: Sliders }, { id: 'leaderboard', label: 'Leaderboard', icon: Trophy }, { id: 'explore', label: 'Data Explorer', icon: BarChart2 }, ...(targetColumn && targetColumn !== '__none__' ? [{ id: 'anomalies', label: 'Anomalies', icon: ShieldAlert }] : []), { id: 'models', label: 'Model Library', icon: Database }, { id: 'deploy', label: 'Deploy', icon: Rocket }, { id: 'history', label: 'History', icon: History }, ...(authUser?.is_admin ? [{ id: 'admin', label: 'Admin', icon: Shield }] : [])].map((item) => (
              <Button key={item.id} variant={activeView === item.id ? 'secondary' : 'ghost'} className="w-full justify-start gap-3" onClick={() => setActiveView(item.id)} data-testid={`nav-${item.id}`}><item.icon className="h-4 w-4" />{item.label}</Button>
            ))}
          </nav>
          <div className="border-t border-sidebar-border p-4"><Card className="bg-sidebar-accent"><CardContent className="p-4"><p className="text-xs font-medium text-sidebar-foreground">Client-Side ML</p><p className="text-xs text-sidebar-foreground/70">All analysis runs in your browser</p></CardContent></Card></div>
        </div>
      </motion.aside>

      <div className="pl-64">
        <motion.header initial={{ y: -100 }} animate={{ y: 0 }} className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="flex items-center justify-between px-6 py-2.5 gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="min-w-0">
                <div className="flex items-center gap-2.5">
                  <h2 className="text-lg font-bold tracking-tight whitespace-nowrap" data-testid="page-title">
                    {activeView === 'dashboard' && 'Dashboard'}{activeView === 'analysis' && 'Universal Analysis'}{activeView === 'preprocess' && 'Data Preprocessing'}{activeView === 'predict' && 'Predictions & Analysis'}{activeView === 'anomalies' && 'Anomaly Detection'}{activeView === 'models' && 'Model Library'}{activeView === 'explore' && 'Data Explorer'}{activeView === 'explainability' && 'Model Explainability'}{activeView === 'compare' && 'Compare Models'}{activeView === 'tune' && 'Hyperparameter Tuning'}{activeView === 'leaderboard' && 'Model Leaderboard'}{activeView === 'history' && 'Analysis History'}{activeView === 'admin' && 'Admin Dashboard'}{activeView === 'deploy' && 'Model Deployment'}{activeView === 'whatif' && 'What-If Analyzer'}
                  </h2>
                  {dataProfile?.fileName && <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-gradient-to-r from-violet-500/15 to-fuchsia-500/15 text-violet-700 dark:text-violet-300 text-xs font-semibold border border-violet-200 dark:border-violet-800" data-testid="current-dataset-badge"><Database className="h-3 w-3" />{dataProfile.fileName}</span>}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <Button variant="ghost" size="sm" onClick={() => { try { localStorage.removeItem('automl_tour_seen'); localStorage.removeItem('automl_milestones_dismissed'); } catch {} window.location.reload(); }} data-testid="guide-toggle-btn" className="h-8 px-2.5 text-xs"><BookOpen className="h-3.5 w-3.5 mr-1.5" />Guide</Button>
              {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button variant="ghost" size="sm" onClick={() => handleSaveAnalysis()} data-testid="save-analysis-btn" className="h-8 px-2.5 text-xs"><Save className="h-3.5 w-3.5 mr-1.5" />Save</Button>}
              {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button variant="ghost" size="sm" onClick={handleShareAnalysis} disabled={exportLoading === 'share'} data-testid="share-analysis-btn" className="h-8 px-2.5 text-xs">{exportLoading === 'share' ? <><span className="h-3.5 w-3.5 mr-1.5 animate-spin rounded-full border-2 border-current border-t-transparent" />...</> : <><Share2 className="h-3.5 w-3.5 mr-1.5" />Share</>}</Button>}
              {(trainingResult || shapGlobal || limeResult || predictionHistory?.length > 0) && !viewOnlyMode && <>
                <Button variant="ghost" size="sm" onClick={handleExportPDF} disabled={!!exportLoading} data-testid="export-pdf-btn" title="Download PDF Report" className="h-8 px-2.5 text-xs font-semibold text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-950/30">{exportLoading === 'pdf' ? <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" /> : <><FileDown className="h-3.5 w-3.5 mr-1.5" />PDF Report</>}</Button>
                <Button variant="ghost" size="sm" onClick={handleExportSheets} disabled={!!exportLoading} data-testid="export-sheets-btn" title="Export to Google Sheets" className="h-8 px-2.5 text-xs">{exportLoading === 'sheets' ? <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" /> : <><Sheet className="h-3.5 w-3.5 mr-1.5" />Sheets</>}</Button>
                <Button variant="ghost" size="sm" onClick={handleExportCSV} disabled={!!exportLoading} data-testid="export-csv-btn" className="h-8 px-2.5 text-xs">{exportLoading === 'csv' ? <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" /> : <><Download className="h-3.5 w-3.5 mr-1.5" />CSV</>}</Button>
              </>}
              {csvText && !viewOnlyMode && <Button variant="ghost" size="sm" onClick={handleClearSession} data-testid="clear-session-btn" className="h-8 px-2.5 text-xs text-destructive hover:text-destructive hover:bg-destructive/10"><Trash2 className="h-3.5 w-3.5 mr-1.5" />Clear</Button>}
              <div className="w-px h-6 bg-border mx-1" />
              <Button variant="ghost" size="icon" onClick={() => setDarkMode(prev => !prev)} data-testid="dark-mode-toggle" className="h-8 w-8"><span className="sr-only">Toggle theme</span>{darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
              {authUser && <div className="flex items-center gap-2 ml-1 pl-2 border-l border-border">
                {authUser.picture ? <img src={authUser.picture} alt="" className="h-8 w-8 rounded-full ring-2 ring-violet-200 dark:ring-violet-800" referrerPolicy="no-referrer" /> : <div className="h-8 w-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white text-sm font-bold ring-2 ring-violet-200 dark:ring-violet-800">{(authUser.name || authUser.email)?.[0]?.toUpperCase()}</div>}
                <span className="text-sm font-medium hidden lg:inline max-w-[140px] truncate" data-testid="user-name">{authUser.name || authUser.email}</span>
                <Button variant="ghost" size="icon" onClick={onLogout} title="Sign out" data-testid="logout-btn" className="h-8 w-8 text-muted-foreground hover:text-destructive"><LogOut className="h-3.5 w-3.5" /></Button>
              </div>}
            </div>
          </div>
        </motion.header>

        <AnimatePresence>{error && <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="mx-8 mt-4" data-testid="error-banner"><Card className="border-destructive bg-destructive/10"><CardContent className="p-4"><p className="text-sm text-destructive font-medium flex items-center gap-2"><AlertCircle className="h-4 w-4" /> {error}</p></CardContent></Card></motion.div>}</AnimatePresence>

        {/* ==================== DECISION TREE MODAL ==================== */}
        <AnimatePresence>{treeModalOpen && (() => {
          const treeModel = getDecisionTreeModel();
          const treeData = treeModel?.modelData?.tree;
          const featureNames = treeModel?.modelData?.featureNames || [];
          return (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4" onClick={() => setTreeModalOpen(false)} data-testid="tree-modal-overlay">
              <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }} className="bg-background border-2 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()} data-testid="tree-modal">
                <div className="flex items-center justify-between px-6 py-4 border-b">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-green-500/10 flex items-center justify-center"><GitBranch className="h-5 w-5 text-green-600" /></div>
                    <div>
                      <h3 className="font-bold text-lg">Decision Tree Visualization</h3>
                      <p className="text-sm text-muted-foreground">{treeModel ? `${ALGO_NAMES[treeModel.algorithm] || treeModel.algorithm} — ${treeModel.targetColumn || 'Model'}` : 'No tree available'}</p>
                    </div>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => setTreeModalOpen(false)} data-testid="tree-modal-close"><X className="h-5 w-5" /></Button>
                </div>
                <div className="flex-1 overflow-auto p-6">
                  {treeData ? (
                    <div className="min-w-fit">
                      <div className="mb-4 p-3 rounded-lg bg-muted/50 text-xs text-muted-foreground">
                        <p className="font-medium text-foreground mb-1">How to read this tree:</p>
                        <p>Each <span className="text-blue-600 dark:text-blue-400 font-semibold">blue box</span> is a decision point — the model checks if a feature value is below or equal to the threshold. <span className="text-green-600 dark:text-green-400 font-semibold">Yes</span> goes left, <span className="text-red-500 dark:text-red-400 font-semibold">No</span> goes right. <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Green boxes</span> are final predictions.</p>
                      </div>
                      {treeData.leaf && !treeData.left && !treeData.right && (
                        <div className="mb-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800 text-xs">
                          <p className="font-medium text-amber-700 dark:text-amber-400 mb-1">Simple tree detected</p>
                          <p className="text-amber-600 dark:text-amber-500">This model resolved to a single prediction — likely because one class dominates the data, or the tree was pruned aggressively. This can happen with imbalanced datasets. Try increasing tree depth or balancing the dataset for a more complex tree.</p>
                        </div>
                      )}
                      <div className="flex justify-center">
                        <TreeNode node={treeData} featureNames={featureNames} depth={0} maxDepth={4} />
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <GitBranch className="h-12 w-12 mx-auto mb-4 opacity-30" />
                      <p className="font-medium">No decision tree data available</p>
                      <p className="text-sm mt-1">Train a Decision Tree model first to see its visualization.</p>
                    </div>
                  )}
                </div>
              </motion.div>
            </motion.div>
          );
        })()}</AnimatePresence>

        {/* ==================== GUIDE PANEL ==================== */}
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
          {activeView === 'dashboard' && <DashboardView />}

          {/* ==================== ANALYSIS ==================== */}
          {activeView === 'analysis' && <AnalysisView />}

          {/* ==================== PREPROCESS ==================== */}
          {activeView === 'preprocess' && <PreprocessView />}

          {/* ==================== PREDICTIONS ==================== */}
          {activeView === 'predict' && <PredictView />}

          {/* ==================== EXPLAINABILITY ==================== */}
          {activeView === 'explainability' && <ExplainabilityView />}

          {/* ==================== COMPARE MODELS ==================== */}
          {activeView === 'compare' && <CompareModelsView />}

          {/* ==================== LEADERBOARD ==================== */}
          {activeView === 'leaderboard' && <LeaderboardView />}

          {/* ==================== DATA EXPLORER ==================== */}
          {activeView === 'explore' && <DataExplorerView />}

          {/* ==================== CLUSTERS ==================== */}
          {activeView === 'clusters' && <ClustersView />}

          {/* ==================== ANOMALIES ==================== */}
          {activeView === 'anomalies' && <AnomaliesView />}

          {/* ==================== MODELS ==================== */}
          {activeView === 'models' && <ModelsView />}

          {/* ==================== HISTORY ==================== */}
          {activeView === 'history' && <HistoryView />}

          {/* ==================== ADMIN ==================== */}
          {activeView === 'admin' && authUser?.is_admin && <AdminView />}

          {/* ==================== DEPLOY ==================== */}
          {activeView === 'deploy' && <DeployView />}

          {/* ==================== WHAT-IF ==================== */}
          {activeView === 'whatif' && <WhatIfView />}

          {/* ==================== TUNE ==================== */}
          {activeView === 'tune' && <TuneView />}

        </AnimatePresence></main>
      </div>
    </div>
    </AppContext.Provider>
  );
}

export default function AppWrapper() {
  const path = window.location.pathname;
  const match = path.match(/^\/predict\/([a-zA-Z0-9-]+)/);
  if (match) {
    return <PublicPredictPage deployId={match[1]} onBack={() => { window.location.href = '/'; }} />;
  }
  return <App />;
}
