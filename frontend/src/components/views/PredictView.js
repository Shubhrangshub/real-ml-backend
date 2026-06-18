import React from 'react';
import { motion } from 'framer-motion';
import {
  Sparkles, FileUp, Eye, BarChart3, TrendingUp, Target, Activity, Download,
  AlertCircle, Database, Zap, Brain, Layers, Upload, CheckCircle2, XCircle,
  ShieldAlert, Cpu, FileText, Play, Clock, Trophy, Info, Lightbulb, SplitSquareVertical
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell, ReferenceLine
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, ALGO_COLORS, CLUSTER_COLORS, UNSUPERVISED_TERMS } from '../../constants';
import { getScoreColor, arrayMin, arrayMax } from '../../utils/helpers';
import { MetricTip, HelpTip } from '../SmartTooltip';
import { useApp } from '../../context/AppContext';

export default function PredictView() {
  const {
    models, dataProfile, predictionFormData, setPredictionFormData, predictionResult,
    setPredictionResult, handlePredict, predictTab, setPredictTab, predictionHistory,
    selectedModelIdx, setSelectedModelIdx, batchCsvText, setBatchCsvText, batchResults,
    setBatchResults, batchProcessing, handleBatchPredict, handleBatchCsvUpload, downloadBatchCsv,
    setActiveView, corrVarX, setCorrVarX, corrVarY, setCorrVarY, corrMatrix,
    unsupervisedResult, clusterPredFormData, setClusterPredFormData, clusterPredResult,
    handleClusterPredict, MetricCard, trainingResult, dataPreview,
    setTrainingResult, setUnsupervisedResult
  } = useApp();

  return (
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
          <CardContent><select value={selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : (() => { let bestIdx = 0, bestScore = -Infinity; models.forEach((m, i) => { const s = m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || -Infinity); if (s > bestScore) { bestScore = s; bestIdx = i; } }); return bestIdx; })()} onChange={e => { setSelectedModelIdx(Number(e.target.value)); setPredictionResult(null); setPredictionFormData({}); }} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="model-selector">
            {(() => { const seen = new Set(); return models.map((m, i) => { const key = `${m.algorithm}-${m.targetColumn}`; if (seen.has(key)) return null; seen.add(key); return <option key={`${m.modelId}-${i}`} value={i}>{ALGO_NAMES[m.algorithm] || m.algorithm} — {m.problemType === 'classification' ? `${((m.metrics?.accuracy || 0) * 100).toFixed(1)}% acc` : `R² ${(m.metrics?.r2 || 0).toFixed(3)}`} — Target: {m.targetColumn}</option>; }).filter(Boolean); })()}
          </select></CardContent></Card>}

        {/* Supervised Prediction Form */}
        {models.length > 0 && (() => { const bestDefault = (() => { let bi = 0, bs = -Infinity; models.forEach((m, i) => { const s = m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || -Infinity); if (s > bs) { bs = s; bi = i; } }); return bi; })(); const idx = selectedModelIdx >= 0 && selectedModelIdx < models.length ? selectedModelIdx : bestDefault; const am = models[idx]; return (
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
            <MetricCard label="R\u00B2" value={`${((trainingResult.bestModel?.testMetrics?.r2 || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.r2} metricKey="r2" />
            <MetricCard label="MAE" value={(trainingResult.bestModel?.testMetrics?.mae || 0).toFixed(2)} metricKey="mae" />
          </>) : (<>
            <MetricCard label="Accuracy" value={`${((trainingResult.bestModel?.testMetrics?.accuracy || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.accuracy} metricKey="accuracy" />
            <MetricCard label="F1" value={`${((trainingResult.bestModel?.testMetrics?.f1 || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.f1} metricKey="f1" />
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
            <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map(m => ({ name: ALGO_NAMES[m.algorithm] || m.algorithm, score: trainingResult.problemType === 'regression' ? +((m.testMetrics?.r2 || 0) * 100).toFixed(2) : +((m.testMetrics?.accuracy || 0) * 100).toFixed(2), fill: ALGO_COLORS[m.algorithm] || '#6b7280' }))}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" tick={{fontSize: 11}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="8 4" /><Bar dataKey="score" radius={[6, 6, 0, 0]}>{trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
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
  );
}
