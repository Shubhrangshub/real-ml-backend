import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  GitBranch, Trophy, Zap, Brain, Target, TrendingUp, BarChart3,
  CheckCircle2, AlertCircle, Shield, AlertTriangle,
  Activity, Clock, Layers
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES } from '../../constants';
import { useApp } from '../../context/AppContext';

const MODEL_COLORS = [
  '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6',
  '#ef4444', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
];

export default function CompareModelsView() {
  const { models, setActiveView } = useApp();
  const [selectedIds, setSelectedIds] = useState(() => models.map(m => m.modelId));

  const selectedModels = useMemo(
    () => models.filter(m => selectedIds.includes(m.modelId)),
    [models, selectedIds]
  );

  const toggleModel = (id) => {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const toggleAll = () => {
    setSelectedIds(prev =>
      prev.length === models.length ? [] : models.map(m => m.modelId)
    );
  };

  // Determine problem type from selected models
  const problemType = selectedModels[0]?.problemType || 'classification';
  const isClassification = problemType === 'classification';

  // ==================== RADAR DATA ====================
  const radarData = useMemo(() => {
    if (selectedModels.length === 0) return [];
    const metrics = isClassification
      ? ['accuracy', 'precision', 'recall', 'f1']
      : ['r2', 'mae', 'rmse'];

    const metricLabels = isClassification
      ? { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1 Score' }
      : { r2: 'R²', mae: '1-MAE(n)', rmse: '1-RMSE(n)' };

    // For regression, normalize MAE and RMSE (lower is better, so invert)
    let maxMae = 0, maxRmse = 0;
    if (!isClassification) {
      selectedModels.forEach(m => {
        maxMae = Math.max(maxMae, m.metrics?.mae || 0);
        maxRmse = Math.max(maxRmse, m.metrics?.rmse || 0);
      });
    }

    return metrics.map(metric => {
      const point = { metric: metricLabels[metric] || metric };
      selectedModels.forEach((m, i) => {
        const name = ALGO_NAMES[m.algorithm] || m.algorithm;
        let val = m.metrics?.[metric] || 0;
        // Normalize regression error metrics to 0-1 (inverted: higher = better)
        if (metric === 'mae' && maxMae > 0) val = 1 - val / maxMae;
        if (metric === 'rmse' && maxRmse > 0) val = 1 - val / maxRmse;
        point[name] = Math.max(0, Math.min(1, val));
      });
      return point;
    });
  }, [selectedModels, isClassification]);

  // ==================== FEATURE IMPORTANCE COMPARISON ====================
  const importanceData = useMemo(() => {
    if (selectedModels.length === 0) return [];
    // Collect all features across models
    const featureSet = new Set();
    selectedModels.forEach(m => {
      (m.featureImportance || []).slice(0, 8).forEach(f => featureSet.add(f.feature));
    });
    const features = [...featureSet].slice(0, 10);
    return features.map(feat => {
      const point = { feature: feat.length > 15 ? feat.slice(0, 12) + '...' : feat };
      selectedModels.forEach(m => {
        const name = ALGO_NAMES[m.algorithm] || m.algorithm;
        const fi = (m.featureImportance || []).find(f => f.feature === feat);
        point[name] = fi ? +(fi.importance * 100).toFixed(1) : 0;
      });
      return point;
    }).sort((a, b) => {
      const sumA = Object.values(a).filter(v => typeof v === 'number').reduce((s, v) => s + v, 0);
      const sumB = Object.values(b).filter(v => typeof v === 'number').reduce((s, v) => s + v, 0);
      return sumB - sumA;
    });
  }, [selectedModels]);

  // ==================== WINNER ANALYSIS ====================
  const winner = useMemo(() => {
    if (selectedModels.length < 2) return null;
    let bestIdx = 0;
    let bestScore = -Infinity;
    selectedModels.forEach((m, i) => {
      const score = isClassification
        ? (m.metrics?.f1 || 0) * 0.4 + (m.metrics?.accuracy || 0) * 0.3 + (m.metrics?.precision || 0) * 0.15 + (m.metrics?.recall || 0) * 0.15
        : (m.metrics?.r2 || 0) * 0.5 + (1 - Math.min(1, (m.metrics?.mae || 0) / 100)) * 0.25 + (1 - Math.min(1, (m.metrics?.rmse || 0) / 100)) * 0.25;
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    });
    const w = selectedModels[bestIdx];
    const wName = ALGO_NAMES[w.algorithm] || w.algorithm;

    // Build reasoning
    const reasons = [];
    if (isClassification) {
      const acc = w.metrics?.accuracy || 0;
      const f1 = w.metrics?.f1 || 0;
      if (acc >= 0.95) reasons.push('exceptional accuracy');
      else if (acc >= 0.85) reasons.push('strong accuracy');
      else reasons.push('reasonable accuracy');
      if (f1 >= 0.9) reasons.push('excellent F1 balance');
      else if (f1 >= 0.7) reasons.push('good precision-recall trade-off');
    } else {
      const r2 = w.metrics?.r2 || 0;
      if (r2 >= 0.9) reasons.push('explains >90% of variance');
      else if (r2 >= 0.7) reasons.push('strong explanatory power');
      else reasons.push('moderate fit');
      if (w.metrics?.mae) reasons.push(`MAE of ${w.metrics.mae.toFixed(3)}`);
    }
    if (w.durationSec && w.durationSec < 1) reasons.push('fast training time');

    // Runner-up
    let runnerUp = null;
    if (selectedModels.length > 2) {
      let secondBest = -Infinity, secondIdx = -1;
      selectedModels.forEach((m, i) => {
        if (i === bestIdx) return;
        const score = isClassification
          ? (m.metrics?.f1 || 0) * 0.4 + (m.metrics?.accuracy || 0) * 0.3 + (m.metrics?.precision || 0) * 0.15 + (m.metrics?.recall || 0) * 0.15
          : (m.metrics?.r2 || 0);
        if (score > secondBest) { secondBest = score; secondIdx = i; }
      });
      if (secondIdx >= 0) runnerUp = selectedModels[secondIdx];
    }

    return { model: w, name: wName, score: bestScore, reasons, runnerUp, index: bestIdx };
  }, [selectedModels, isClassification]);

  // ==================== METRIC COMPARISON TABLE ====================
  const metricKeys = isClassification
    ? ['accuracy', 'precision', 'recall', 'f1']
    : ['r2', 'mae', 'rmse'];
  const metricLabels = { accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1 Score', r2: 'R²', mae: 'MAE', rmse: 'RMSE' };
  const higherIsBetter = { accuracy: true, precision: true, recall: true, f1: true, r2: true, mae: false, rmse: false };

  // Empty state
  if (models.length === 0) {
    return (
      <motion.div key="compare" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="compare-view">
        <Card className="border-2 border-dashed">
          <CardContent className="py-20 text-center">
            <GitBranch className="h-16 w-16 text-muted-foreground/40 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Models to Compare</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              Train at least 2 models to unlock side-by-side comparison with radar charts, feature importance analysis, and winner recommendations.
            </p>
            <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="go-train-btn">
              <Zap className="h-4 w-4 mr-2" />Train Models
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  if (models.length === 1) {
    return (
      <motion.div key="compare" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="compare-view">
        <Card className="border-2 border-dashed">
          <CardContent className="py-20 text-center">
            <GitBranch className="h-16 w-16 text-muted-foreground/40 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Need More Models</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              You have 1 model. Train at least one more to compare them side by side.
            </p>
            <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="go-train-btn">
              <Zap className="h-4 w-4 mr-2" />Train Another Model
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div key="compare" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="compare-view">

      {/* ==================== MODEL SELECTOR ==================== */}
      <motion.div variants={fadeInUp}>
        <Card data-testid="model-selector-card">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <GitBranch className="h-5 w-5" />Select Models to Compare
                </CardTitle>
                <CardDescription>{selectedModels.length} of {models.length} models selected</CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={toggleAll} data-testid="toggle-all-btn">
                {selectedIds.length === models.length ? 'Deselect All' : 'Select All'}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {models.map((m, i) => {
                const isSelected = selectedIds.includes(m.modelId);
                const name = ALGO_NAMES[m.algorithm] || m.algorithm;
                const color = MODEL_COLORS[i % MODEL_COLORS.length];
                const score = isClassification ? m.metrics?.accuracy : m.metrics?.r2;
                return (
                  <button
                    key={m.modelId}
                    onClick={() => toggleModel(m.modelId)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border-2 transition-all text-sm font-medium ${
                      isSelected
                        ? 'border-current bg-accent shadow-sm'
                        : 'border-transparent bg-muted/50 opacity-50 hover:opacity-75'
                    }`}
                    style={{ borderColor: isSelected ? color : undefined, color: isSelected ? color : undefined }}
                    data-testid={`model-toggle-${i}`}
                  >
                    <div className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                    <span>{name}</span>
                    {score !== undefined && (
                      <Badge variant={!isClassification && score < 0 ? 'destructive' : 'secondary'} className="text-xs px-1.5 py-0">
                        {!isClassification && score < 0 && <AlertTriangle className="h-3 w-3 mr-0.5 inline" />}
                        {(score * 100).toFixed(1)}%
                      </Badge>
                    )}
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {selectedModels.length < 2 && (
        <motion.div variants={fadeInUp}>
          <Card className="border-dashed">
            <CardContent className="py-10 text-center">
              <AlertCircle className="h-10 w-10 text-muted-foreground/40 mx-auto mb-3" />
              <p className="text-muted-foreground">Select at least 2 models to see the comparison</p>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {selectedModels.length >= 2 && (
        <>
          {/* ==================== NEGATIVE R² WARNING ==================== */}
          {!isClassification && selectedModels.some(m => (m.metrics?.r2 || 0) < 0) && (
            <motion.div variants={fadeInUp}>
              <Card className="border-amber-300 dark:border-amber-700 bg-amber-50/60 dark:bg-amber-950/20" data-testid="negative-r2-warning">
                <CardContent className="p-4 flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-amber-800 dark:text-amber-300">Models Performing Worse Than Baseline</p>
                    <p className="text-xs text-amber-700/80 dark:text-amber-400/70 mt-1">
                      {selectedModels.filter(m => (m.metrics?.r2 || 0) < 0).map(m => ALGO_NAMES[m.algorithm] || m.algorithm).join(', ')} {selectedModels.filter(m => (m.metrics?.r2 || 0) < 0).length === 1 ? 'has' : 'have'} a negative R² score, meaning {selectedModels.filter(m => (m.metrics?.r2 || 0) < 0).length === 1 ? 'it performs' : 'they perform'} worse than simply predicting the mean value. Consider excluding {selectedModels.filter(m => (m.metrics?.r2 || 0) < 0).length === 1 ? 'this model' : 'these models'} from production use.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ==================== WINNER RECOMMENDATION ==================== */}
          {winner && (
            <motion.div variants={fadeInUp}>
              <Card className="border-2 overflow-hidden" style={{ borderColor: MODEL_COLORS[winner.index % MODEL_COLORS.length] + '60' }} data-testid="winner-card">
                <div className="absolute top-0 left-0 right-0 h-1" style={{ backgroundColor: MODEL_COLORS[winner.index % MODEL_COLORS.length] }} />
                <CardContent className="p-6">
                  <div className="flex items-start gap-4 flex-wrap">
                    <div className="h-14 w-14 rounded-2xl flex items-center justify-center shrink-0" style={{ backgroundColor: MODEL_COLORS[winner.index % MODEL_COLORS.length] + '18' }}>
                      <Trophy className="h-7 w-7" style={{ color: MODEL_COLORS[winner.index % MODEL_COLORS.length] }} />
                    </div>
                    <div className="flex-1 min-w-[200px]">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-lg font-bold">Recommended: {winner.name}</h3>
                        <Badge className="text-xs" style={{ backgroundColor: MODEL_COLORS[winner.index % MODEL_COLORS.length], color: 'white' }}>
                          Best Overall
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        {winner.name} outperforms the other {selectedModels.length - 1} model{selectedModels.length > 2 ? 's' : ''} with {winner.reasons.join(', ')}.
                      </p>
                      <div className="flex flex-wrap gap-3">
                        {metricKeys.map(key => {
                          const val = winner.model.metrics?.[key];
                          if (val === undefined) return null;
                          return (
                            <div key={key} className="text-center px-3 py-1.5 rounded-lg bg-muted/50">
                              <p className="text-xs text-muted-foreground">{metricLabels[key]}</p>
                              <p className="text-sm font-bold" style={{ color: MODEL_COLORS[winner.index % MODEL_COLORS.length] }}>
                                {higherIsBetter[key] ? (val * 100).toFixed(1) + '%' : val.toFixed(4)}
                              </p>
                            </div>
                          );
                        })}
                        {winner.model.durationSec && (
                          <div className="text-center px-3 py-1.5 rounded-lg bg-muted/50">
                            <p className="text-xs text-muted-foreground">Time</p>
                            <p className="text-sm font-bold">{winner.model.durationSec.toFixed(2)}s</p>
                          </div>
                        )}
                      </div>
                      {winner.runnerUp && (
                        <p className="text-xs text-muted-foreground mt-2">
                          Runner-up: <span className="font-medium">{ALGO_NAMES[winner.runnerUp.algorithm] || winner.runnerUp.algorithm}</span>
                          {' '}({isClassification ? `${((winner.runnerUp.metrics?.accuracy || 0) * 100).toFixed(1)}% accuracy` : `R² ${(winner.runnerUp.metrics?.r2 || 0).toFixed(3)}`})
                        </p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ==================== RADAR CHART ==================== */}
          <motion.div variants={fadeInUp}>
            <Card data-testid="radar-chart-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" />Performance Radar</CardTitle>
                <CardDescription>
                  {isClassification ? 'Classification metrics (0-1 scale, higher is better)' : 'Regression metrics (normalized, higher is better)'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                    <PolarGrid stroke="hsl(var(--border))" />
                    <PolarAngleAxis dataKey="metric" className="text-xs" tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }} />
                    <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                    {selectedModels.map((m, i) => {
                      const name = ALGO_NAMES[m.algorithm] || m.algorithm;
                      return (
                        <Radar
                          key={m.modelId}
                          name={name}
                          dataKey={name}
                          stroke={MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length]}
                          fill={MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length]}
                          fillOpacity={0.1}
                          strokeWidth={2}
                          dot={{ r: 4, fill: MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length] }}
                        />
                      );
                    })}
                    <Legend />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                      formatter={(value) => [(value * 100).toFixed(1) + '%']}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>

          {/* ==================== METRIC COMPARISON TABLE ==================== */}
          <motion.div variants={fadeInUp}>
            <Card data-testid="metrics-table-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Metric Breakdown</CardTitle>
                <CardDescription>Side-by-side comparison of all metrics. Green = best in class.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg border overflow-auto">
                  <table className="w-full text-sm" data-testid="metrics-comparison-table">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="p-3 text-left font-medium text-muted-foreground">Metric</th>
                        {selectedModels.map((m, i) => (
                          <th key={m.modelId} className="p-3 text-center font-medium" style={{ color: MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length] }}>
                            <div className="flex items-center justify-center gap-1.5">
                              <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length] }} />
                              {ALGO_NAMES[m.algorithm] || m.algorithm}
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metricKeys.map(key => {
                        const values = selectedModels.map(m => m.metrics?.[key] ?? null);
                        const validVals = values.filter(v => v !== null);
                        const bestVal = higherIsBetter[key]
                          ? Math.max(...validVals)
                          : Math.min(...validVals);
                        return (
                          <tr key={key} className="border-b last:border-0">
                            <td className="p-3 font-medium">{metricLabels[key]}</td>
                            {values.map((val, i) => {
                              const isBest = val !== null && val === bestVal && validVals.length > 1;
                              return (
                                <td key={i} className={`p-3 text-center font-mono text-sm ${isBest ? 'text-emerald-600 dark:text-emerald-400 font-bold' : ''}`}>
                                  {val !== null ? (
                                    <span className="flex items-center justify-center gap-1">
                                      {higherIsBetter[key] ? (val * 100).toFixed(2) + '%' : val.toFixed(4)}
                                      {isBest && <CheckCircle2 className="h-3.5 w-3.5" />}
                                    </span>
                                  ) : '—'}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                      {/* Training time row */}
                      <tr className="border-b last:border-0 bg-muted/20">
                        <td className="p-3 font-medium flex items-center gap-1.5"><Clock className="h-3.5 w-3.5" />Training Time</td>
                        {selectedModels.map((m, i) => {
                          const dur = m.durationSec;
                          const fastest = Math.min(...selectedModels.map(x => x.durationSec || Infinity));
                          const isFastest = dur && dur === fastest && selectedModels.length > 1;
                          return (
                            <td key={i} className={`p-3 text-center font-mono text-sm ${isFastest ? 'text-emerald-600 dark:text-emerald-400 font-bold' : ''}`}>
                              {dur ? (
                                <span className="flex items-center justify-center gap-1">
                                  {dur.toFixed(3)}s
                                  {isFastest && <CheckCircle2 className="h-3.5 w-3.5" />}
                                </span>
                              ) : '—'}
                            </td>
                          );
                        })}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* ==================== FEATURE IMPORTANCE COMPARISON ==================== */}
          {importanceData.length > 0 && (
            <motion.div variants={fadeInUp}>
              <Card data-testid="feature-importance-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Feature Importance Comparison</CardTitle>
                  <CardDescription>How each model weighs the top features differently</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={Math.max(300, importanceData.length * 45)}>
                    <BarChart data={importanceData} layout="vertical" margin={{ left: 20, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" domain={[0, 'auto']} tickFormatter={v => v + '%'} />
                      <YAxis dataKey="feature" type="category" width={100} tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                        formatter={(value) => [value.toFixed(1) + '%', '']}
                      />
                      <Legend />
                      {selectedModels.map((m, i) => {
                        const name = ALGO_NAMES[m.algorithm] || m.algorithm;
                        return (
                          <Bar
                            key={m.modelId}
                            dataKey={name}
                            fill={MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length]}
                            radius={[0, 3, 3, 0]}
                            barSize={12}
                          />
                        );
                      })}
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ==================== CONFUSION MATRIX COMPARISON ==================== */}
          {isClassification && (
            <motion.div variants={fadeInUp}>
              <Card data-testid="confusion-matrix-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Target className="h-5 w-5" />Confusion Matrix Comparison</CardTitle>
                  <CardDescription>Classification accuracy per class for each model</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-6" style={{ gridTemplateColumns: `repeat(${Math.min(selectedModels.length, 3)}, 1fr)` }}>
                    {selectedModels.map((m, mi) => {
                      const cm = m.metrics?.confusionMatrix;
                      if (!cm) return null;
                      const color = MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length];
                      const name = ALGO_NAMES[m.algorithm] || m.algorithm;
                      const maxVal = Math.max(...cm.matrix.flat());
                      return (
                        <div key={m.modelId} className="space-y-2" data-testid={`cm-${mi}`}>
                          <div className="flex items-center gap-2 mb-2">
                            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                            <span className="text-sm font-semibold">{name}</span>
                          </div>
                          <div className="rounded-lg border overflow-hidden">
                            <table className="w-full text-xs">
                              <thead>
                                <tr className="bg-muted/50">
                                  <th className="p-2 text-left font-medium">Pred / Act</th>
                                  {cm.classes.map(c => <th key={c} className="p-2 text-center font-medium">C{c}</th>)}
                                </tr>
                              </thead>
                              <tbody>
                                {cm.matrix.map((row, ri) => (
                                  <tr key={ri} className="border-t">
                                    <td className="p-2 font-medium bg-muted/30">C{cm.classes[ri]}</td>
                                    {row.map((val, ci) => {
                                      const isDiag = ri === ci;
                                      const intensity = maxVal > 0 ? val / maxVal : 0;
                                      return (
                                        <td
                                          key={ci}
                                          className={`p-2 text-center font-mono ${isDiag ? 'font-bold' : ''}`}
                                          style={{
                                            backgroundColor: isDiag
                                              ? `${color}${Math.round(intensity * 40 + 10).toString(16).padStart(2, '0')}`
                                              : val > 0 ? 'hsl(var(--destructive) / 0.1)' : 'transparent'
                                          }}
                                        >
                                          {val}
                                        </td>
                                      );
                                    })}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          {m.metrics?.perClassMetrics && (
                            <div className="text-xs text-muted-foreground space-y-0.5 mt-1">
                              {m.metrics.perClassMetrics.map(pc => (
                                <div key={pc.class} className="flex justify-between">
                                  <span>Class {pc.class}:</span>
                                  <span>P={(pc.precision * 100).toFixed(0)}% R={(pc.recall * 100).toFixed(0)}% F1={(pc.f1 * 100).toFixed(0)}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ==================== QUICK STATS GRID ==================== */}
          <motion.div variants={fadeInUp}>
            <div className="grid gap-4 md:grid-cols-3">
              <Card data-testid="models-compared-stat">
                <CardContent className="p-5 text-center">
                  <Brain className="h-8 w-8 text-primary mx-auto mb-2" />
                  <p className="text-3xl font-bold">{selectedModels.length}</p>
                  <p className="text-sm text-muted-foreground">Models Compared</p>
                </CardContent>
              </Card>
              <Card data-testid="best-metric-stat">
                <CardContent className="p-5 text-center">
                  <TrendingUp className="h-8 w-8 text-emerald-500 mx-auto mb-2" />
                  <p className="text-3xl font-bold">
                    {isClassification
                      ? ((Math.max(...selectedModels.map(m => m.metrics?.accuracy || 0))) * 100).toFixed(1) + '%'
                      : (Math.max(...selectedModels.map(m => m.metrics?.r2 || 0))).toFixed(3)
                    }
                  </p>
                  <p className="text-sm text-muted-foreground">Best {isClassification ? 'Accuracy' : 'R²'}</p>
                </CardContent>
              </Card>
              <Card data-testid="spread-stat">
                <CardContent className="p-5 text-center">
                  <Shield className="h-8 w-8 text-amber-500 mx-auto mb-2" />
                  <p className="text-3xl font-bold">
                    {(() => {
                      const scores = selectedModels.map(m => isClassification ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || 0));
                      return ((Math.max(...scores) - Math.min(...scores)) * 100).toFixed(1) + '%';
                    })()}
                  </p>
                  <p className="text-sm text-muted-foreground">Performance Spread</p>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        </>
      )}
    </motion.div>
  );
}
