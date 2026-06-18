import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SlidersHorizontal, Play, Trophy, ArrowRight, Loader2, BarChart3, TrendingUp, Zap, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { fadeInUp, ALGO_NAMES } from '../../constants';
import { useApp } from '../../context/AppContext';
import {
  buildModelWithParams, predictBatch, calcRegressionMetrics,
  calcClassificationMetrics, extractImportance, trainTestSplit,
  HYPERPARAMETER_DEFS, prepareFeatures,
} from '../../utils/mlEngine';

const STRATEGIES = [
  { value: 'random', label: 'Random Search', desc: 'Sample random combinations — fast and effective', trials: 20 },
  { value: 'grid', label: 'Grid Search', desc: 'Exhaustive search — slower but thorough', trials: null },
];

function generateGridCombinations(paramDefs) {
  const keys = paramDefs.map(p => p.key);
  const grids = paramDefs.map(p => {
    const vals = [];
    for (let v = p.min; v <= p.max; v += p.step) vals.push(Math.round(v * 1000) / 1000);
    // Limit to ~8 values per param to keep grid size manageable
    if (vals.length > 8) {
      const step = Math.floor(vals.length / 7);
      const sampled = [vals[0]];
      for (let i = step; i < vals.length - 1; i += step) sampled.push(vals[i]);
      sampled.push(vals[vals.length - 1]);
      return sampled;
    }
    return vals;
  });
  // Cartesian product
  const combos = [];
  const recurse = (idx, current) => {
    if (idx === keys.length) { combos.push({ ...current }); return; }
    for (const val of grids[idx]) { current[keys[idx]] = val; recurse(idx + 1, current); }
  };
  recurse(0, {});
  return combos;
}

function generateRandomCombinations(paramDefs, count = 20) {
  const combos = [];
  for (let i = 0; i < count; i++) {
    const params = {};
    paramDefs.forEach(p => {
      const steps = Math.round((p.max - p.min) / p.step);
      const randomStep = Math.floor(Math.random() * (steps + 1));
      params[p.key] = Math.round((p.min + randomStep * p.step) * 1000) / 1000;
    });
    combos.push(params);
  }
  return combos;
}

export default function TuneView() {
  const { models, trainingResult, dataProfile, targetColumn, setModels } = useApp();
  const [selectedAlgo, setSelectedAlgo] = useState('');
  const [strategy, setStrategy] = useState('random');
  const [isTuning, setIsTuning] = useState(false);
  const [tuningResult, setTuningResult] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });

  // Algorithms that have tunable hyperparameters
  const tunableAlgos = useMemo(() => {
    if (!trainingResult?.leaderboard) return [];
    return trainingResult.leaderboard.filter(m => HYPERPARAMETER_DEFS[m.algorithm]).map(m => m.algorithm);
  }, [trainingResult]);

  const paramDefs = useMemo(() => selectedAlgo ? (HYPERPARAMETER_DEFS[selectedAlgo] || []) : [], [selectedAlgo]);

  // The original model's score for comparison
  const originalEntry = trainingResult?.leaderboard?.find(m => m.algorithm === selectedAlgo);
  const problemType = trainingResult?.problemType || 'classification';
  const primaryKey = problemType === 'classification' ? 'accuracy' : 'r2';

  const runTuning = useCallback(async () => {
    if (!selectedAlgo || !dataProfile || !targetColumn) return;
    setIsTuning(true);
    setTuningResult(null);

    // Use a timeout to let React render the loading state
    await new Promise(r => setTimeout(r, 50));

    try {
      // Re-prepare data from dataProfile - rows are already objects with header keys
      const rows = dataProfile.rows.map(r => {
        const obj = {};
        dataProfile.headers.forEach((h) => {
          const v = r[h];  // r is an object, access by header key
          obj[h] = dataProfile.numericColumns?.includes(h) ? (v === '' ? NaN : Number(v)) : v;
        });
        return obj;
      });

      const { X, y } = prepareFeatures(rows, targetColumn);
      const { X_train, X_test, y_train, y_test } = trainTestSplit(X, y, 0.2);

      // Generate parameter combinations
      const combos = strategy === 'grid' ? generateGridCombinations(paramDefs) : generateRandomCombinations(paramDefs, 20);
      setProgress({ current: 0, total: combos.length });

      const trials = [];
      for (let i = 0; i < combos.length; i++) {
        const params = combos[i];
        try {
          const modelObj = buildModelWithParams(selectedAlgo, X_train, y_train, problemType, params);
          const testPreds = predictBatch(modelObj, X_test);
          const metrics = problemType === 'regression'
            ? calcRegressionMetrics(y_test, testPreds)
            : calcClassificationMetrics(y_test, testPreds);
          trials.push({ params, metrics, score: metrics[primaryKey] || 0 });
        } catch {
          trials.push({ params, metrics: {}, score: -Infinity, error: true });
        }
        setProgress({ current: i + 1, total: combos.length });
        // Yield to UI every 5 trials
        if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
      }

      // Sort by score
      trials.sort((a, b) => b.score - a.score);
      const best = trials[0];
      const originalScore = originalEntry?.testMetrics?.[primaryKey] || 0;
      const improvement = best.score - originalScore;

      setTuningResult({
        algorithm: selectedAlgo,
        strategy,
        totalTrials: trials.length,
        bestParams: best.params,
        bestMetrics: best.metrics,
        bestScore: best.score,
        originalScore,
        improvement,
        topTrials: trials.slice(0, 10),
      });
    } catch (err) {
      console.error('Tuning error:', err);
    }
    setIsTuning(false);
  }, [selectedAlgo, strategy, paramDefs, dataProfile, targetColumn, problemType, primaryKey, originalEntry]);

  const applyBestParams = useCallback(() => {
    if (!tuningResult || !dataProfile || !targetColumn) return;
    // Retrain with best params and add to models - rows are already objects with header keys
    const rows = dataProfile.rows.map(r => {
      const obj = {};
      dataProfile.headers.forEach((h) => {
        const v = r[h];  // r is an object, access by header key
        obj[h] = dataProfile.numericColumns?.includes(h) ? (v === '' ? NaN : Number(v)) : v;
      });
      return obj;
    });
    const { X, y, featureNames, numericCols, categoricalCols, encodingMap, targetEncoding } = prepareFeatures(rows, targetColumn);
    const { X_train, y_train } = trainTestSplit(X, y, 0.2);

    const modelObj = buildModelWithParams(tuningResult.algorithm, X_train, y_train, problemType, tuningResult.bestParams);
    const modelData = { featureNames, numericCols, categoricalCols, encodingMap, targetEncoding, ...modelObj };

    setModels(prev => [...prev, {
      modelId: `tuned-${Date.now()}`, algorithm: tuningResult.algorithm, problemType,
      metrics: tuningResult.bestMetrics, trainMetrics: tuningResult.bestMetrics,
      featureImportance: extractImportance(modelObj, featureNames),
      createdAt: new Date().toISOString(), durationSec: 0,
      evalMode: 'split', targetColumn, modelData,
      tuned: true, tuningParams: tuningResult.bestParams,
    }]);
  }, [tuningResult, dataProfile, targetColumn, problemType, setModels]);

  if (!trainingResult) {
    return (
      <motion.div {...fadeInUp} className="p-8 text-center" data-testid="tune-empty">
        <SlidersHorizontal className="h-16 w-16 mx-auto mb-4 text-muted-foreground/30" />
        <h3 className="text-lg font-bold mb-2">Train a Model First</h3>
        <p className="text-sm text-muted-foreground">Train models on a dataset, then come here to optimize their hyperparameters.</p>
      </motion.div>
    );
  }

  return (
    <motion.div {...fadeInUp} className="p-8 space-y-6" data-testid="tune-view">
      {/* Header */}
      <Card>
        <CardContent className="p-5">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
              <SlidersHorizontal className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="font-bold" data-testid="tune-title">Hyperparameter Tuning</h3>
              <p className="text-xs text-muted-foreground">Optimize model parameters to improve performance. Select an algorithm and a search strategy below.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Algorithm Selection */}
        <Card data-testid="tune-algo-select">
          <CardHeader className="pb-3"><CardTitle className="text-sm flex items-center gap-2"><Zap className="h-4 w-4 text-amber-500" />Select Algorithm</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {tunableAlgos.length === 0 && <p className="text-xs text-muted-foreground">No tunable algorithms found. Train models first — most algorithms have tunable parameters.</p>}
            {tunableAlgos.map(algo => {
              const entry = trainingResult.leaderboard.find(m => m.algorithm === algo);
              const score = entry?.testMetrics?.[primaryKey];
              return (
                <label key={algo} className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer border transition-all ${selectedAlgo === algo ? 'border-cyan-300 bg-cyan-50/50 dark:border-cyan-700 dark:bg-cyan-950/20' : 'border-transparent hover:bg-muted/30'}`} data-testid={`tune-algo-${algo}`}>
                  <input type="radio" name="tuneAlgo" checked={selectedAlgo === algo} onChange={() => { setSelectedAlgo(algo); setTuningResult(null); }} className="accent-cyan-500" />
                  <div className="flex-1 min-w-0">
                    <span className="text-sm font-medium">{ALGO_NAMES[algo] || algo}</span>
                    <span className="block text-[11px] text-muted-foreground">{HYPERPARAMETER_DEFS[algo]?.length || 0} tunable parameters</span>
                  </div>
                  {score != null && <Badge variant="outline" className="text-[10px] shrink-0">{primaryKey}: {(score * (problemType === 'classification' ? 100 : 1)).toFixed(2)}{problemType === 'classification' ? '%' : ''}</Badge>}
                </label>
              );
            })}
          </CardContent>
        </Card>

        {/* Search Strategy */}
        <Card data-testid="tune-strategy">
          <CardHeader className="pb-3"><CardTitle className="text-sm flex items-center gap-2"><BarChart3 className="h-4 w-4 text-violet-500" />Search Strategy</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {STRATEGIES.map(s => (
              <label key={s.value} className={`flex items-start gap-3 p-3 rounded-lg cursor-pointer border transition-all ${strategy === s.value ? 'border-violet-300 bg-violet-50/50 dark:border-violet-700 dark:bg-violet-950/20' : 'border-transparent hover:bg-muted/30'}`} data-testid={`strategy-${s.value}`}>
                <input type="radio" name="strategy" checked={strategy === s.value} onChange={() => setStrategy(s.value)} className="mt-0.5 accent-violet-500" />
                <div>
                  <span className="text-sm font-medium">{s.label}</span>
                  <span className="block text-[11px] text-muted-foreground">{s.desc}</span>
                </div>
              </label>
            ))}

            {/* Parameter Preview */}
            {selectedAlgo && paramDefs.length > 0 && (
              <div className="mt-4 pt-3 border-t space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Parameters to tune:</p>
                {paramDefs.map(p => (
                  <div key={p.key} className="flex items-center justify-between text-xs">
                    <span>{p.label}</span>
                    <span className="text-muted-foreground font-mono">{p.min} → {p.max} (step: {p.step})</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Run Button */}
      <Button className="w-full h-12 text-base gap-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-600 hover:to-blue-600" disabled={!selectedAlgo || isTuning} onClick={runTuning} data-testid="run-tuning-btn">
        {isTuning ? (
          <><Loader2 className="h-5 w-5 animate-spin" />Tuning... ({progress.current}/{progress.total})</>
        ) : (
          <><Play className="h-5 w-5" />Run Hyperparameter Tuning</>
        )}
      </Button>

      {/* Progress Bar */}
      {isTuning && (
        <div className="h-2 rounded-full bg-muted overflow-hidden">
          <motion.div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500" initial={{ width: 0 }} animate={{ width: `${progress.total ? (progress.current / progress.total * 100) : 0}%` }} transition={{ duration: 0.3 }} />
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {tuningResult && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4" data-testid="tuning-results">
            {/* Summary Card */}
            <Card className={tuningResult.improvement > 0 ? 'border-emerald-300 dark:border-emerald-700' : 'border-amber-300 dark:border-amber-700'}>
              <CardContent className="p-5">
                <div className="flex items-center gap-3 mb-4">
                  <Trophy className={`h-8 w-8 ${tuningResult.improvement > 0 ? 'text-emerald-500' : 'text-amber-500'}`} />
                  <div>
                    <h4 className="font-bold">{tuningResult.improvement > 0 ? 'Improvement Found!' : 'No Significant Improvement'}</h4>
                    <p className="text-xs text-muted-foreground">{tuningResult.totalTrials} trials evaluated using {tuningResult.strategy === 'grid' ? 'Grid Search' : 'Random Search'}</p>
                  </div>
                </div>

                {/* Before vs After */}
                <div className="grid grid-cols-3 gap-4 items-center" data-testid="tuning-comparison">
                  <div className="text-center p-3 rounded-lg bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-1">Original</p>
                    <p className="text-2xl font-bold" data-testid="original-score">{(tuningResult.originalScore * (problemType === 'classification' ? 100 : 1)).toFixed(2)}{problemType === 'classification' ? '%' : ''}</p>
                    <p className="text-[10px] text-muted-foreground">{primaryKey}</p>
                  </div>
                  <div className="text-center">
                    <ArrowRight className="h-6 w-6 mx-auto text-muted-foreground" />
                    <p className={`text-sm font-bold mt-1 ${tuningResult.improvement > 0 ? 'text-emerald-600' : 'text-amber-600'}`}>
                      {tuningResult.improvement > 0 ? '+' : ''}{(tuningResult.improvement * (problemType === 'classification' ? 100 : 1)).toFixed(2)}{problemType === 'classification' ? '%' : ''}
                    </p>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800">
                    <p className="text-xs text-muted-foreground mb-1">Tuned Best</p>
                    <p className="text-2xl font-bold text-emerald-600" data-testid="tuned-score">{(tuningResult.bestScore * (problemType === 'classification' ? 100 : 1)).toFixed(2)}{problemType === 'classification' ? '%' : ''}</p>
                    <p className="text-[10px] text-muted-foreground">{primaryKey}</p>
                  </div>
                </div>

                {/* Best Parameters */}
                <div className="mt-4 pt-3 border-t">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Best Parameters:</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(tuningResult.bestParams).map(([k, v]) => (
                      <Badge key={k} variant="outline" className="font-mono text-[10px]">{k}: {typeof v === 'number' ? v.toFixed(v % 1 === 0 ? 0 : 3) : v}</Badge>
                    ))}
                  </div>
                </div>

                {/* Apply Button or Keep Original Recommendation */}
                {tuningResult.improvement > 0 ? (
                  <Button className="w-full mt-4 gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600" onClick={applyBestParams} data-testid="apply-tuned-btn">
                    <TrendingUp className="h-4 w-4" />Apply Tuned Model
                  </Button>
                ) : (
                  <div className="mt-4 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800" data-testid="keep-original-rec">
                    <p className="text-sm font-semibold text-amber-700 dark:text-amber-400 flex items-center gap-1.5">
                      <AlertTriangle className="h-4 w-4" />Recommendation: Keep original model
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Tuning did not improve the score. The original model is already performing at or near its best for this parameter range. Try a different algorithm or expand the dataset.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Top Trials Table */}
            <Card data-testid="tuning-trials">
              <CardHeader className="pb-3"><CardTitle className="text-sm">Top 10 Trials</CardTitle></CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2 font-medium text-muted-foreground">#</th>
                        {paramDefs.map(p => <th key={p.key} className="text-left p-2 font-medium text-muted-foreground">{p.label}</th>)}
                        <th className="text-left p-2 font-medium text-muted-foreground">{primaryKey}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tuningResult.topTrials.map((t, i) => (
                        <tr key={i} className={`border-b last:border-0 ${i === 0 ? 'bg-emerald-50/30 dark:bg-emerald-950/10 font-semibold' : ''}`}>
                          <td className="p-2">{i === 0 ? <Trophy className="h-3.5 w-3.5 text-amber-500 inline" /> : i + 1}</td>
                          {paramDefs.map(p => <td key={p.key} className="p-2 font-mono">{typeof t.params[p.key] === 'number' ? t.params[p.key].toFixed(t.params[p.key] % 1 === 0 ? 0 : 3) : t.params[p.key]}</td>)}
                          <td className="p-2 font-mono">{(t.score * (problemType === 'classification' ? 100 : 1)).toFixed(2)}{problemType === 'classification' ? '%' : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
