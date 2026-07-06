import React from 'react';
import { motion } from 'framer-motion';
import {
  Upload, Play, FileText, Target, ChevronRight, AlertCircle, Zap, Activity,
  ShieldAlert, CheckCircle2, XCircle, Eye, Trash2, Brain, Trophy, Download,
  Sparkles, BarChart3, Table2, Layers, Database, GitBranch, Shield, X, Info,
  Lightbulb, SplitSquareVertical, TrendingUp, Settings2, ArrowRight
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, ReferenceLine, PieChart, Pie, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, ALGO_COLORS, ALGO_DESCRIPTIONS } from '../../constants';
import { getScoreColor, arrayMin, arrayMax } from '../../utils/helpers';
import { MetricTip, HelpTip } from '../SmartTooltip';
import { useApp } from '../../context/AppContext';

export default function AnalysisView() {
  const {
    csvText, setCsvText, dataProfile, targetColumn, setTargetColumn, algorithm, setAlgorithm,
    evalMode, setEvalMode, cleaningLog, precleanScan, isTraining, trainingResult,
    models, dragActive, handleCsvTextChange, handleFileUpload, handleDrag, handleDrop,
    handleClean, handleTrain, handleRunUnsupervised, isRunningUnsupervised, unsupervisedResult,
    setActiveView, setPredictTab, datasetSummary, taskSuggestion, suggestedTarget,
    sampleDatasets, loadSampleDataset, datasetScan, handleDeleteModel, handleDownloadModel,
    setTreeModalAlgo, setTreeModalOpen, MetricCard, TreeNode, columns, error,
    dataPreview, setTrainingResult, setUnsupervisedResult,
    preprocessConfig, preprocessLog
  } = useApp();

  return (
    <>
    <motion.div key="analysis" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="analysis-view">
    <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" />Sample Data</CardTitle></CardHeader>
      <CardContent><div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">{sampleDatasets.map((sample, idx) => {
        const colors = ['from-blue-500/10 to-cyan-500/10 border-blue-200 dark:border-blue-800 hover:border-blue-400', 'from-emerald-500/10 to-teal-500/10 border-emerald-200 dark:border-emerald-800 hover:border-emerald-400', 'from-amber-500/10 to-orange-500/10 border-amber-200 dark:border-amber-800 hover:border-amber-400', 'from-violet-500/10 to-purple-500/10 border-violet-200 dark:border-violet-800 hover:border-violet-400', 'from-rose-500/10 to-pink-500/10 border-rose-200 dark:border-rose-800 hover:border-rose-400'];
        const iconColors = ['text-blue-600 dark:text-blue-400', 'text-emerald-600 dark:text-emerald-400', 'text-amber-600 dark:text-amber-400', 'text-violet-600 dark:text-violet-400', 'text-rose-600 dark:text-rose-400'];
        const icons = [<Target className="h-5 w-5" />, <TrendingUp className="h-5 w-5" />, <Activity className="h-5 w-5" />, <Brain className="h-5 w-5" />, <Layers className="h-5 w-5" />];
        return <Card key={sample.name || idx} className={`cursor-pointer hover:shadow-lg transition-all duration-200 border bg-gradient-to-br ${colors[idx % 5]}`} onClick={() => loadSampleDataset(sample)} data-testid={`sample-dataset-${idx}`}><CardContent className="p-4"><div className="flex items-center justify-between gap-3"><div className="flex items-center gap-3"><div className={`p-2 rounded-lg bg-white/60 dark:bg-white/10 ${iconColors[idx % 5]}`}>{icons[idx]}</div><div><p className="font-semibold text-sm">{sample.name}</p><p className="text-xs text-muted-foreground mt-0.5">{sample.description}</p></div></div><ChevronRight className={`h-4 w-4 shrink-0 ${iconColors[idx % 5]}`} /></div></CardContent></Card>;
      })}</div></CardContent></Card></motion.div>

    <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle className="flex items-center gap-2"><Upload className="h-5 w-5" />Upload Data</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop} className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'} hover:border-primary hover:bg-accent/50`} data-testid="csv-dropzone"><input type="file" accept=".csv" onChange={handleFileUpload} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="csv-file-input" /><Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" /><p className="text-lg font-medium mb-2">Drop your CSV file here</p><p className="text-sm text-muted-foreground mb-3">or click to browse</p><p className="text-xs text-muted-foreground/60">Upload a clean dataset in CSV format. Ensure it has meaningful column headers and at least 10 rows for reliable results.</p></div>
        <Separator className="my-6" />
        <div><label className="text-sm font-medium mb-2 block">Or paste CSV data:</label><textarea value={csvText} onChange={(e) => handleCsvTextChange(e.target.value)} placeholder="Paste CSV data..." rows={6} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2" data-testid="csv-text-input" /></div>
      </CardContent></Card></motion.div>

    {dataProfile && <motion.div variants={fadeInUp}><Card className="border-2 border-blue-500/30" data-testid="data-profile-card"><CardHeader><CardTitle className="flex items-center gap-2"><Table2 className="h-5 w-5" />Dataset Profile</CardTitle><CardDescription>{dataProfile.rowCount} rows x {dataProfile.columnCount} columns</CardDescription></CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3"><div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Numeric</p><p className="text-2xl font-bold text-blue-600">{dataProfile.numericColumns.length}</p></div><div className="bg-emerald-50 dark:bg-emerald-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Categorical</p><p className="text-2xl font-bold text-emerald-600">{dataProfile.categoricalColumns.length}</p></div><div className="bg-violet-50 dark:bg-violet-950/30 rounded-lg p-3 text-center"><p className="text-xs text-muted-foreground">Rows</p><p className="text-2xl font-bold text-violet-600">{dataProfile.rowCount}</p></div></div>
        <div className="rounded-md border overflow-auto max-h-64"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Column</th><th className="p-2 text-left font-medium">Type</th><th className="p-2 text-left font-medium">Unique</th><th className="p-2 text-left font-medium">Range / Values</th></tr></thead><tbody>{dataProfile.columns.map((col, idx) => <tr key={idx} className="border-b last:border-0"><td className="p-2 font-mono text-xs">{col.name}</td><td className="p-2"><Badge variant={col.type === 'numeric' ? 'default' : 'secondary'} className="text-xs">{col.type}</Badge></td><td className="p-2 text-xs">{col.uniqueCount}</td><td className="p-2 text-xs text-muted-foreground">{col.type === 'numeric' ? `${col.min?.toFixed(1)} — ${col.max?.toFixed(1)} (mean: ${col.mean?.toFixed(1)})` : `${col.sampleValues.join(', ')}${col.uniqueCount > col.sampleValues.length ? ` (+${col.uniqueCount - col.sampleValues.length} more)` : ''}`}</td></tr>)}</tbody></table></div>
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

        {/* Preprocessing Nudge */}
        {(() => {
          const ppActive = [
            preprocessConfig.missingValues !== 'none' && 'Missing Values',
            preprocessConfig.scaling !== 'none' && 'Feature Scaling',
            preprocessConfig.outlierMethod !== 'none' && 'Outlier Treatment',
            (preprocessConfig.excludeFeatures?.length || 0) > 0 && 'Feature Selection',
          ].filter(Boolean);
          const hasIssues = datasetScan && (datasetScan.totalMissing > 0 || datasetScan.totalOutliers > 0 || datasetScan.scaleIssue);
          return (
            <div className={`mt-4 p-3 rounded-lg border flex items-center gap-3 ${ppActive.length > 0 ? 'border-emerald-300 bg-emerald-50/50 dark:bg-emerald-950/20 dark:border-emerald-800' : hasIssues ? 'border-amber-300 bg-amber-50/50 dark:bg-amber-950/20 dark:border-amber-800' : 'border-blue-200 bg-blue-50/50 dark:bg-blue-950/20 dark:border-blue-800'}`} data-testid="preprocessing-nudge">
              <Settings2 className={`h-4 w-4 shrink-0 ${ppActive.length > 0 ? 'text-emerald-600' : hasIssues ? 'text-amber-600' : 'text-blue-500'}`} />
              <div className="flex-1 min-w-0">
                {ppActive.length > 0
                  ? <p className="text-sm text-emerald-700 dark:text-emerald-400"><span className="font-medium">Preprocessing active:</span> {ppActive.join(' → ')} will be applied during training.</p>
                  : hasIssues
                    ? <p className="text-sm text-amber-700 dark:text-amber-400"><span className="font-medium">Data issues detected.</span> Configure preprocessing to improve model accuracy.</p>
                    : <p className="text-sm text-blue-700 dark:text-blue-400">No preprocessing configured — training will use raw data.</p>
                }
              </div>
              <Button variant="outline" size="sm" className="shrink-0 text-xs h-7 gap-1" onClick={() => setActiveView('preprocess')} data-testid="go-to-preprocess-btn">
                <Settings2 className="h-3 w-3" />{ppActive.length > 0 ? 'Edit' : 'Configure'}
              </Button>
            </div>
          );
        })()}

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
      <CardDescription className="text-sm mt-2">Your <strong>{trainingResult.problemType}</strong> model was trained on {trainingResult.dataInfo?.numSamples} samples in {trainingResult.totalTime?.toFixed(2)}s. The best algorithm is <strong>{ALGO_NAMES[trainingResult.bestModel?.algorithm] || trainingResult.bestModel?.algorithm}</strong> with {trainingResult.problemType === 'regression' ? `an R\u00B2 of ${((trainingResult.bestModel?.testMetrics?.r2 || 0) * 100).toFixed(1)}%` : `${((trainingResult.bestModel?.testMetrics?.accuracy || 0) * 100).toFixed(1)}% accuracy`}.</CardDescription></CardHeader>
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

        {/* Preprocessing Applied */}
        {preprocessLog && preprocessLog.length > 0 && (
          <div className="p-4 rounded-lg bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-950/20 dark:to-amber-950/20 border border-orange-200 dark:border-orange-800" data-testid="preprocessing-applied-card">
            <div className="flex items-center gap-2 mb-2">
              <Settings2 className="h-4 w-4 text-orange-600" />
              <p className="text-sm font-semibold text-orange-800 dark:text-orange-200">Preprocessing Applied ({preprocessLog.length} step{preprocessLog.length > 1 ? 's' : ''})</p>
            </div>
            <div className="space-y-1">
              {preprocessLog.map((entry, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <CheckCircle2 className="h-3 w-3 text-emerald-500 shrink-0" />
                  <Badge variant="outline" className="text-[9px] shrink-0 border-orange-300 text-orange-700 dark:text-orange-400">{entry.step}</Badge>
                  <span className="text-muted-foreground">{entry.message}</span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-orange-600/70 dark:text-orange-400/60 mt-2 flex items-center gap-1">
              <ArrowRight className="h-3 w-3" />
              To compare: retrain without preprocessing to see the difference in metrics.
            </p>
          </div>
        )}

        {trainingResult.dataInfo?.removedLeakageColumns?.length > 0 && <Card className="border-2 border-orange-500 bg-orange-50 dark:bg-orange-950" data-testid="leakage-warning"><CardContent className="p-4"><div className="flex items-start gap-3"><AlertCircle className="h-6 w-6 text-orange-600 mt-0.5 shrink-0" /><div><p className="font-semibold text-orange-900 dark:text-orange-100">Data Leakage Prevention</p><div className="flex flex-wrap gap-2 mt-2">{trainingResult.dataInfo.removedLeakageColumns.map((col, idx) => <Badge key={idx} variant="outline" className="bg-orange-100 dark:bg-orange-900">{col}</Badge>)}</div></div></div></CardContent></Card>}

        {/* Test Metrics */}
        <Card><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />How Well Did My Model Perform?</CardTitle><CardDescription>These metrics show how accurately {ALGO_NAMES[trainingResult.bestModel?.algorithm]} predicts on {trainingResult.splitInfo?.testSize} unseen test samples. Green = great, amber = okay, red = needs improvement.</CardDescription></CardHeader>
          <CardContent>
            {trainingResult.problemType === 'regression' ? (
              <div className="grid gap-3 md:grid-cols-3" data-testid="test-metrics-grid">
                <MetricCard label="R\u00B2 Score" value={`${((trainingResult.bestModel?.testMetrics?.r2 || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.r2} metricKey="r2" />
                <MetricCard label="MAE" value={(trainingResult.bestModel?.testMetrics?.mae || 0).toFixed(2)} metricKey="mae" />
                <MetricCard label="RMSE" value={(trainingResult.bestModel?.testMetrics?.rmse || 0).toFixed(2)} metricKey="rmse" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-3 md:grid-cols-4" data-testid="test-metrics-grid">
                  <MetricCard label="Accuracy" value={`${((trainingResult.bestModel?.testMetrics?.accuracy || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.accuracy} metricKey="accuracy" />
                  <MetricCard label="Precision" value={`${((trainingResult.bestModel?.testMetrics?.precision || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.precision} metricKey="precision" />
                  <MetricCard label="Recall" value={`${((trainingResult.bestModel?.testMetrics?.recall || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.recall} metricKey="recall" />
                  <MetricCard label="F1 Score" value={`${((trainingResult.bestModel?.testMetrics?.f1 || 0) * 100).toFixed(2)}%`} score={trainingResult.bestModel?.testMetrics?.f1} metricKey="f1" />
                </div>

                {/* Confusion Matrix Heatmap */}
                {trainingResult.bestModel?.testMetrics?.confusionMatrix && (
                  <div data-testid="confusion-matrix">
                    <p className="text-sm font-medium mb-2 flex items-center gap-2"><Target className="h-4 w-4" />Confusion Matrix</p>
                    <p className="text-xs text-muted-foreground mb-4">Green cells on the diagonal = correct predictions. Red cells = misclassifications. A perfect model would only have green diagonal cells.</p>
                    <div className="overflow-auto"><table className="text-sm border-collapse">
                      <thead><tr><td className="p-2"></td><td className="p-2 text-xs text-center text-muted-foreground font-medium" colSpan={trainingResult.bestModel?.testMetrics?.confusionMatrix.classes.length}>Predicted</td></tr>
                      <tr><td className="p-2 text-xs text-muted-foreground font-medium">Actual</td>{trainingResult.bestModel?.testMetrics?.confusionMatrix.classes.map((cls, i) => <td key={i} className="p-2 text-center font-mono text-xs font-medium min-w-[60px]">{trainingResult.bestModel?.testMetrics?.confusionMatrix.classes.length <= 2 && trainingResult.problemType === 'classification' && models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[cls] : cls}</td>)}</tr></thead>
                      <tbody>{trainingResult.bestModel?.testMetrics?.confusionMatrix.classes.map((cls, i) => {
                        const maxVal = Math.max(...trainingResult.bestModel?.testMetrics?.confusionMatrix.matrix.flat());
                        return (
                        <tr key={i}><td className="p-2 font-mono text-xs font-medium">{trainingResult.bestModel?.testMetrics?.confusionMatrix.classes.length <= 2 && trainingResult.problemType === 'classification' && models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[cls] : cls}</td>
                        {trainingResult.bestModel?.testMetrics?.confusionMatrix.matrix[i].map((val, j) => {
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
                {trainingResult.bestModel?.testMetrics?.perClassMetrics && (
                  <div data-testid="per-class-metrics">
                    <p className="text-sm font-medium mb-3">Per-Class Metrics</p>
                    <div className="rounded-md border"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-2 text-left font-medium">Class</th><th className="p-2 text-right font-medium"><MetricTip metricKey="precision">Precision</MetricTip></th><th className="p-2 text-right font-medium"><MetricTip metricKey="recall">Recall</MetricTip></th><th className="p-2 text-right font-medium"><MetricTip metricKey="f1">F1 Score</MetricTip></th></tr></thead>
                    <tbody>{trainingResult.bestModel?.testMetrics?.perClassMetrics.map((pc, i) => <tr key={i} className="border-b last:border-0"><td className="p-2 font-mono text-xs">{models[models.length - 1]?.modelData?.targetEncoding ? models[models.length - 1].modelData.targetEncoding[pc.class] : pc.class}</td><td className="p-2 text-right">{(pc.precision * 100).toFixed(1)}%</td><td className="p-2 text-right">{(pc.recall * 100).toFixed(1)}%</td><td className="p-2 text-right font-medium">{(pc.f1 * 100).toFixed(1)}%</td></tr>)}</tbody></table></div>
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
          <ResponsiveContainer width="100%" height={350}><BarChart data={trainingResult.bestModel.featureImportance} margin={{ bottom: 20 }}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="feature" angle={-35} textAnchor="end" height={100} tick={{fontSize: 11}} interval={0} /><YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} /><Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} /><Bar dataKey="importance" radius={[6, 6, 0, 0]}>{trainingResult.bestModel.featureImportance.map((_, i) => <Cell key={i} fill={`hsl(${210 + i * 15}, 70%, ${45 + i * 3}%)`} />)}</Bar></BarChart></ResponsiveContainer></CardContent></Card>}

        {/* Decision Tree Visualization (inline chart) */}
        {(() => {
          const dtModel = models.find(m => m.algorithm === 'decision_tree' && m.modelData?.tree);
          const rfModel = !dtModel ? models.find(m => m.algorithm === 'random_forest' && m.modelData?.trees?.length > 0) : null;
          const treeSource = dtModel || rfModel;
          const treeRoot = dtModel?.modelData?.tree || rfModel?.modelData?.trees?.[0];
          const tNames = treeSource?.modelData?.featureNames || [];
          if (!treeRoot || treeRoot.leaf) return null;

          // Layout the tree: compute positions for each node
          const nodePositions = [];
          const NODE_W = 140, NODE_H = 54, H_GAP = 16, V_GAP = 60;
          const MAX_DEPTH = 4;
          let nextX = 0;

          function layoutTree(node, depth) {
            if (!node || depth > MAX_DEPTH) return null;
            if (node.leaf) {
              const x = nextX; nextX += NODE_W + H_GAP;
              const pos = { x, y: depth * (NODE_H + V_GAP), w: NODE_W, h: NODE_H, node, depth };
              nodePositions.push(pos);
              return pos;
            }
            const leftPos = layoutTree(node.left, depth + 1);
            const rightPos = layoutTree(node.right, depth + 1);
            const x = leftPos && rightPos ? (leftPos.x + rightPos.x) / 2
                     : leftPos ? leftPos.x : rightPos ? rightPos.x : nextX;
            if (!leftPos && !rightPos) nextX += NODE_W + H_GAP;
            const pos = { x, y: depth * (NODE_H + V_GAP), w: NODE_W, h: NODE_H, node, depth, leftChild: leftPos, rightChild: rightPos };
            nodePositions.push(pos);
            return pos;
          }

          const rootPos = layoutTree(treeRoot, 0);
          const svgW = Math.max(nextX + 20, 400);
          const maxY = nodePositions.reduce((m, p) => Math.max(m, p.y), 0);
          const svgH = maxY + NODE_H + 40;

          function renderSVGNode(pos) {
            if (!pos) return null;
            const { x, y, w, h, node: nd, leftChild, rightChild } = pos;
            const isLeaf = nd.leaf;
            const fName = !isLeaf && tNames[nd.feature] ? tNames[nd.feature] : (!isLeaf ? `F${nd.feature}` : '');
            const truncName = fName.length > 14 ? fName.slice(0, 13) + '...' : fName;
            const cx = x + w / 2, cy = y + h / 2;

            return (
              <g key={`${x}-${y}-${nd.n}`}>
                {/* Connecting lines to children */}
                {leftChild && <><line x1={cx} y1={y + h} x2={leftChild.x + leftChild.w / 2} y2={leftChild.y} stroke="var(--border)" strokeWidth="2" /><text x={(cx + leftChild.x + leftChild.w / 2) / 2 - 12} y={(y + h + leftChild.y) / 2} fontSize="9" fill="#22c55e" fontWeight="bold">Yes</text></>}
                {rightChild && <><line x1={cx} y1={y + h} x2={rightChild.x + rightChild.w / 2} y2={rightChild.y} stroke="var(--border)" strokeWidth="2" /><text x={(cx + rightChild.x + rightChild.w / 2) / 2 + 4} y={(y + h + rightChild.y) / 2} fontSize="9" fill="#ef4444" fontWeight="bold">No</text></>}
                {/* Node box */}
                <rect x={x} y={y} width={w} height={h} rx={8} ry={8}
                  fill={isLeaf ? 'var(--color-emerald-50, #ecfdf5)' : 'var(--color-blue-50, #eff6ff)'}
                  stroke={isLeaf ? '#6ee7b7' : '#93c5fd'} strokeWidth="2" />
                {isLeaf ? (<>
                  <text x={cx} y={y + 18} textAnchor="middle" fontSize="10" fontWeight="bold" fill="#059669">Predict</text>
                  <text x={cx} y={y + 33} textAnchor="middle" fontSize="12" fontWeight="bold" fill="#047857" fontFamily="monospace">{typeof nd.value === 'number' ? nd.value.toFixed(2) : nd.value}</text>
                  <text x={cx} y={y + 47} textAnchor="middle" fontSize="9" fill="#6b7280">{nd.n} samples</text>
                </>) : (<>
                  <text x={cx} y={y + 16} textAnchor="middle" fontSize="10" fontWeight="600" fill="#2563eb"><title>{fName}</title>{truncName}</text>
                  <text x={cx} y={y + 32} textAnchor="middle" fontSize="11" fontWeight="bold" fill="#1e40af" fontFamily="monospace">{'\u2264'} {nd.threshold?.toFixed(2)}</text>
                  <text x={cx} y={y + 47} textAnchor="middle" fontSize="9" fill="#6b7280">{nd.n} samples</text>
                </>)}
                {/* Render children recursively */}
                {renderSVGNode(leftChild)}
                {renderSVGNode(rightChild)}
              </g>
            );
          }

          return (
            <Card data-testid="decision-tree-chart">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2"><GitBranch className="h-4 w-4 text-green-600" />Decision Tree Structure</CardTitle>
                <CardDescription>Visual flowchart showing how the {treeSource?.algorithm === 'random_forest' ? 'first tree in the Random Forest' : 'Decision Tree'} makes decisions. Blue nodes are split points, green nodes are predictions.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-auto rounded-lg border bg-muted/20 p-4" style={{ maxHeight: 500 }}>
                  <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} className="mx-auto" style={{ minWidth: svgW }}>
                    {renderSVGNode(rootPos)}
                  </svg>
                </div>
              </CardContent>
            </Card>
          );
        })()}

        {/* Model Comparison Chart */}
        {trainingResult.leaderboard?.length > 1 && <Card data-testid="model-comparison-chart"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><BarChart3 className="h-4 w-4" />Model Comparison</CardTitle><CardDescription>Side-by-side comparison of all algorithms. The best model is automatically selected for predictions.</CardDescription></CardHeader><CardContent>
          <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map(m => ({
            name: ALGO_NAMES[m.algorithm] || m.algorithm,
            score: trainingResult.problemType === 'regression' ? +((m.testMetrics?.r2 || 0) * 100).toFixed(2) : +((m.testMetrics?.accuracy || 0) * 100).toFixed(2),
            fill: ALGO_COLORS[m.algorithm] || '#6b7280'
          }))}><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="name" tick={{fontSize: 11}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="8 4" label={{ value: '50% baseline', position: 'right', fontSize: 10, fill: '#94a3b8' }} /><Bar dataKey="score" radius={[6, 6, 0, 0]}>{trainingResult.leaderboard.filter(m => m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
        </CardContent></Card>}

        {/* Cross-Validation Performance Chart */}
        {trainingResult.evalMode === 'cv' && trainingResult.leaderboard?.some(m => m.cvScore !== null) && (
          <Card data-testid="cv-performance-chart"><CardHeader><CardTitle className="text-lg flex items-center gap-2"><Activity className="h-4 w-4" />Cross-Validation Performance</CardTitle><CardDescription>Average performance across 5 data folds. Higher CV scores indicate more reliable, consistent models.</CardDescription></CardHeader><CardContent>
            <ResponsiveContainer width="100%" height={300}><BarChart data={trainingResult.leaderboard.filter(m => m.cvScore !== null && m.algorithm !== 'baseline').map(m => ({
              name: ALGO_NAMES[m.algorithm] || m.algorithm,
              cvScore: +(m.cvScore * 100).toFixed(2),
              testScore: +(trainingResult.problemType === 'regression' ? ((m.testMetrics?.r2 || 0) * 100) : ((m.testMetrics?.accuracy || 0) * 100)).toFixed(2),
              fill: ALGO_COLORS[m.algorithm] || '#6b7280'
            }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={(v) => `${v}%`} /><Legend /><Bar dataKey="cvScore" name="CV Score" radius={[4, 4, 0, 0]}>{trainingResult.leaderboard.filter(m => m.cvScore !== null && m.algorithm !== 'baseline').map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar><Bar dataKey="testScore" name="Test Score" radius={[4, 4, 0, 0]} fill="#94a3b8" opacity={0.5} /></BarChart></ResponsiveContainer>
          </CardContent></Card>
        )}

        {/* Algorithm Leaderboard */}
        <Card><CardHeader><CardTitle className="text-lg">Algorithm Leaderboard</CardTitle><CardDescription>All algorithms ranked by {trainingResult.evalMode === 'cv' ? 'cross-validation score' : 'test performance'} — the best model is automatically selected</CardDescription></CardHeader>
          <CardContent><div className="space-y-2" data-testid="leaderboard">{trainingResult.leaderboard?.map((model, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 rounded-lg border" data-testid={`leaderboard-entry-${idx}`} style={idx === 0 ? { borderColor: ALGO_COLORS[model.algorithm], borderWidth: 2 } : {}}>
              <div className="flex items-center gap-3"><Badge variant={idx === 0 ? 'default' : 'secondary'} style={idx === 0 ? { backgroundColor: ALGO_COLORS[model.algorithm] } : {}}>{idx + 1}</Badge><div><p className="font-medium">{ALGO_NAMES[model.algorithm] || model.algorithm}</p><p className="text-xs text-muted-foreground">{model.durationSec ? `${model.durationSec.toFixed(3)}s` : '-'}{idx === 0 && ' — Best Model'}</p></div></div>
              <div className="flex items-center gap-3">
                {(model.algorithm === 'decision_tree' || model.algorithm === 'random_forest') && <Button variant="outline" size="sm" className="text-green-600 border-green-300 hover:bg-green-50 dark:border-green-700 dark:hover:bg-green-950/30" onClick={() => { setTreeModalAlgo(model.algorithm); setTreeModalOpen(true); }} data-testid={`view-tree-btn-${idx}`}><GitBranch className="h-3.5 w-3.5 mr-1.5" />View Tree</Button>}
                <div className="text-right font-mono text-sm" data-testid={`leaderboard-score-${idx}`}>
                {(() => { const m = model.testMetrics; const s = m.accuracy !== undefined ? m.accuracy : (m.r2 !== undefined ? m.r2 : 0); const c = getScoreColor(s); return <div className={`font-semibold ${c.text}`}>{m.accuracy !== undefined ? <MetricTip metricKey="accuracy" value={m.accuracy}>{`${(m.accuracy * 100).toFixed(2)}% acc`}</MetricTip> : m.r2 !== undefined ? <MetricTip metricKey="r2" value={m.r2}>{`${(m.r2 * 100).toFixed(2)}% R\u00B2`}</MetricTip> : '-'}</div>; })()}
                {model.cvScore !== null && model.cvScore !== undefined && <div className="text-xs text-emerald-600 font-semibold" data-testid={`cv-score-${idx}`}><MetricTip metricKey="cvScore" value={model.cvScore}>CV: {(model.cvScore * 100).toFixed(2)}%</MetricTip></div>}
                </div>
              </div>
            </div>
          ))}</div></CardContent></Card>
      </CardContent></Card></motion.div>}
  </motion.div>

          {unsupervisedResult && (
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
    </>
  );
}
