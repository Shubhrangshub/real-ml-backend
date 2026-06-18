import React, { useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Sliders, Play, RotateCcw, TrendingUp, TrendingDown, Minus, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { fadeInUp, ALGO_NAMES } from '../../constants';
import { useApp } from '../../context/AppContext';
import { predictOne, prepareInputForPrediction } from '../../utils/mlEngine';

export default function WhatIfView() {
  const { models, columns, targetColumn, dataProfile } = useApp();
  const [selectedModelIdx, setSelectedModelIdx] = useState(0);
  const [baseline, setBaseline] = useState({});
  const [modified, setModified] = useState({});
  const [baselineResult, setBaselineResult] = useState(null);
  const [modifiedResult, setModifiedResult] = useState(null);
  const [initialized, setInitialized] = useState(false);

  const modelObj = models[selectedModelIdx];
  const md = modelObj?.modelData;

  // Use the model's own numericCols/categoricalCols for features
  const features = useMemo(() => {
    if (!md) return columns.filter(c => c !== targetColumn);
    return [...(md.numericCols || []), ...(md.categoricalCols || [])];
  }, [md, columns, targetColumn]);

  // Compute feature ranges/categories from the data, with frequency info for rare-category flagging
  const featureStats = useMemo(() => {
    if (!dataProfile) return {};
    const stats = {};
    const rows = dataProfile.rows || [];
    const totalRows = rows.length;
    features.forEach(feat => {
      const vals = rows.map(r => r[feat]).filter(v => v !== '' && v != null && v !== undefined);
      const numeric = vals.map(Number).filter(v => !isNaN(v));
      if (numeric.length > vals.length * 0.5 && numeric.length > 0) {
        const sorted = [...numeric].sort((a, b) => a - b);
        stats[feat] = {
          type: 'numeric', min: Math.min(...numeric), max: Math.max(...numeric),
          mean: numeric.reduce((a, b) => a + b, 0) / numeric.length,
          median: sorted[Math.floor(sorted.length / 2)],
        };
      } else {
        // Use encodingMap categories if available
        const mapCats = md?.encodingMap?.[feat];
        const unique = mapCats || [...new Set(vals.map(String))].sort();
        // Compute frequency for each category to flag rare ones
        const freq = {};
        vals.forEach(v => { const s = String(v); freq[s] = (freq[s] || 0) + 1; });
        const categories = unique.slice(0, 30);
        const categoryFreq = {};
        categories.forEach(c => { categoryFreq[c] = freq[c] || 0; });
        const rareThreshold = Math.max(2, totalRows * 0.02); // < 2% occurrence
        stats[feat] = { type: 'categorical', categories, categoryFreq, rareThreshold, totalRows };
      }
    });
    return stats;
  }, [dataProfile, features, md]);

  // Initialize with median/mode values
  const initialize = useCallback(() => {
    const init = {};
    features.forEach(feat => {
      const s = featureStats[feat];
      if (!s) { init[feat] = ''; return; }
      if (s.type === 'numeric') init[feat] = s.median?.toFixed(2) || '0';
      else init[feat] = s.categories?.[0] || '';
    });
    setBaseline(init);
    setModified({ ...init });
    setBaselineResult(null);
    setModifiedResult(null);
    setInitialized(true);
  }, [features, featureStats]);

  const runPrediction = useCallback((inputData) => {
    if (!md) return null;
    try {
      // Build a row object matching the model's expected column format
      const row = {};
      (md.numericCols || []).forEach(col => { row[col] = Number(inputData[col]) || 0; });
      (md.categoricalCols || []).forEach(col => { row[col] = inputData[col] || ''; });
      const fvs = prepareInputForPrediction([row], md);
      if (!fvs || !fvs.length) return null;
      return predictOne(md, fvs[0]);
    } catch {
      return null;
    }
  }, [md]);

  const handleRunBoth = useCallback(() => {
    setBaselineResult(runPrediction(baseline));
    setModifiedResult(runPrediction(modified));
  }, [baseline, modified, runPrediction]);

  const handleReset = useCallback(() => {
    setModified({ ...baseline });
    setModifiedResult(null);
  }, [baseline]);

  if (!models.length || !targetColumn || !md) {
    return (
      <motion.div variants={fadeInUp} initial="initial" animate="animate" data-testid="whatif-view">
        <Card>
          <CardContent className="p-12 text-center">
            <AlertCircle className="h-12 w-12 mx-auto mb-3 text-muted-foreground/40" />
            <h3 className="font-bold text-lg mb-2">Train a Model First</h3>
            <p className="text-sm text-muted-foreground">The What-If Analyzer needs a trained model. Go to Analysis, select a target, and train.</p>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div variants={fadeInUp} initial="initial" animate="animate" className="space-y-6" data-testid="whatif-view">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base"><Sliders className="h-5 w-5 text-fuchsia-500" />What-If Analyzer</CardTitle>
            <div className="flex items-center gap-2">
              <select value={selectedModelIdx} onChange={e => setSelectedModelIdx(Number(e.target.value))}
                className="text-xs px-2 py-1.5 rounded-lg border bg-background" data-testid="whatif-model-select">
                {models.map((m, i) => <option key={m.modelId || i} value={i}>{ALGO_NAMES[m.algorithm] || m.algorithm}</option>)}
              </select>
              {!initialized && <Button size="sm" className="gap-1.5 bg-gradient-to-r from-fuchsia-500 to-violet-500 text-white" onClick={initialize} data-testid="whatif-init-btn"><Play className="h-3.5 w-3.5" />Start Analyzing</Button>}
            </div>
          </div>
        </CardHeader>
        {!initialized && (
          <CardContent>
            <p className="text-sm text-muted-foreground">Explore how changing individual feature values affects your model&apos;s predictions. Click &ldquo;Start Analyzing&rdquo; to begin.</p>
          </CardContent>
        )}
      </Card>

      {initialized && (
        <>
          {/* Side-by-side feature inputs */}
          <div className="grid md:grid-cols-2 gap-4">
            {/* Baseline */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-blue-500" />Baseline Scenario
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 max-h-[400px] overflow-y-auto">
                {features.map(feat => {
                  const s = featureStats[feat];
                  return (
                    <div key={feat} className="flex items-center gap-2">
                      <label className="text-xs font-medium w-36 truncate shrink-0" title={feat}>{feat}</label>
                      {s?.type === 'categorical' ? (
                        <select value={baseline[feat] || ''} onChange={e => setBaseline(prev => ({ ...prev, [feat]: e.target.value }))}
                          className="flex-1 text-xs px-2 py-1.5 rounded border bg-background" data-testid={`whatif-baseline-${feat}`}>
                          {s.categories.map(c => {
                            const isRare = s.categoryFreq && s.rareThreshold && (s.categoryFreq[c] || 0) < s.rareThreshold;
                            return <option key={c} value={c}>{c}{isRare ? ' (rare)' : ''}</option>;
                          })}
                        </select>
                      ) : (
                        <div className="flex-1 flex items-center gap-2">
                          <input type="range" min={s?.min || 0} max={s?.max || 100} step={(s?.max - s?.min) / 100 || 1}
                            value={Number(baseline[feat]) || 0}
                            onChange={e => setBaseline(prev => ({ ...prev, [feat]: e.target.value }))}
                            className="flex-1 h-1.5 accent-blue-500" />
                          <input type="number" value={baseline[feat] || ''} onChange={e => setBaseline(prev => ({ ...prev, [feat]: e.target.value }))}
                            className="w-20 text-xs px-2 py-1 rounded border bg-background text-right" />
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Modified */}
            <Card className="border-fuchsia-200 dark:border-fuchsia-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-fuchsia-500" />Modified Scenario
                  <Button variant="ghost" size="sm" className="ml-auto h-6 text-[10px]" onClick={handleReset}><RotateCcw className="h-3 w-3 mr-1" />Reset</Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 max-h-[400px] overflow-y-auto">
                {features.map(feat => {
                  const s = featureStats[feat];
                  const changed = baseline[feat] !== modified[feat];
                  return (
                    <div key={feat} className={`flex items-center gap-2 ${changed ? 'bg-fuchsia-50 dark:bg-fuchsia-950/20 rounded-lg px-1 -mx-1' : ''}`}>
                      <label className={`text-xs font-medium w-36 truncate shrink-0 ${changed ? 'text-fuchsia-700 dark:text-fuchsia-300' : ''}`} title={feat}>{feat}</label>
                      {s?.type === 'categorical' ? (
                        <select value={modified[feat] || ''} onChange={e => setModified(prev => ({ ...prev, [feat]: e.target.value }))}
                          className={`flex-1 text-xs px-2 py-1.5 rounded border bg-background ${changed ? 'border-fuchsia-300 dark:border-fuchsia-700' : ''}`} data-testid={`whatif-modified-${feat}`}>
                          {s.categories.map(c => {
                            const isRare = s.categoryFreq && s.rareThreshold && (s.categoryFreq[c] || 0) < s.rareThreshold;
                            return <option key={c} value={c}>{c}{isRare ? ' (rare)' : ''}</option>;
                          })}
                        </select>
                      ) : (
                        <div className="flex-1 flex items-center gap-2">
                          <input type="range" min={s?.min || 0} max={s?.max || 100} step={(s?.max - s?.min) / 100 || 1}
                            value={Number(modified[feat]) || 0}
                            onChange={e => setModified(prev => ({ ...prev, [feat]: e.target.value }))}
                            className="flex-1 h-1.5 accent-fuchsia-500" />
                          <input type="number" value={modified[feat] || ''} onChange={e => setModified(prev => ({ ...prev, [feat]: e.target.value }))}
                            className={`w-20 text-xs px-2 py-1 rounded border bg-background text-right ${changed ? 'border-fuchsia-300 dark:border-fuchsia-700' : ''}`} />
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>

          {/* Run button */}
          <Button className="w-full gap-2 bg-gradient-to-r from-fuchsia-500 to-violet-500 text-white hover:from-fuchsia-600 hover:to-violet-600 h-11"
            onClick={handleRunBoth} data-testid="whatif-run-btn">
            <Play className="h-4 w-4" />Compare Predictions
          </Button>

          {/* Results comparison */}
          {(baselineResult !== null || modifiedResult !== null) && (
            <Card data-testid="whatif-results">
              <CardContent className="p-6">
                <div className="grid md:grid-cols-3 gap-6 items-center">
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground mb-1">Baseline Prediction</p>
                    <p className="text-3xl font-bold text-blue-600">
                      {baselineResult !== null ? (typeof baselineResult === 'number' ? baselineResult.toFixed(4) : String(baselineResult)) : '—'}
                    </p>
                  </div>
                  <div className="text-center">
                    {baselineResult !== null && modifiedResult !== null && typeof baselineResult === 'number' && typeof modifiedResult === 'number' ? (() => {
                      const diff = modifiedResult - baselineResult;
                      const pct = baselineResult !== 0 ? (diff / Math.abs(baselineResult)) * 100 : 0;
                      return (
                        <div>
                          <div className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-lg font-bold ${diff > 0 ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : diff < 0 ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 'bg-gray-100 text-gray-700'}`}>
                            {diff > 0 ? <TrendingUp className="h-5 w-5" /> : diff < 0 ? <TrendingDown className="h-5 w-5" /> : <Minus className="h-5 w-5" />}
                            {diff > 0 ? '+' : ''}{diff.toFixed(4)}
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{pct > 0 ? '+' : ''}{pct.toFixed(1)}% change</p>
                        </div>
                      );
                    })() : (
                      <Badge variant="outline" className="text-xs">
                        {baselineResult !== null && modifiedResult !== null && baselineResult !== modifiedResult ? 'Changed' : baselineResult === modifiedResult ? 'No change' : '—'}
                      </Badge>
                    )}
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-muted-foreground mb-1">Modified Prediction</p>
                    <p className="text-3xl font-bold text-fuchsia-600">
                      {modifiedResult !== null ? (typeof modifiedResult === 'number' ? modifiedResult.toFixed(4) : String(modifiedResult)) : '—'}
                    </p>
                  </div>
                </div>

                {/* Changed features summary */}
                <div className="mt-4 pt-4 border-t">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Changes Made:</p>
                  <div className="flex flex-wrap gap-2">
                    {features.filter(f => baseline[f] !== modified[f]).map(f => (
                      <Badge key={f} variant="outline" className="text-[10px] gap-1">
                        {f}: {baseline[f]} → <span className="text-fuchsia-600 font-semibold">{modified[f]}</span>
                      </Badge>
                    ))}
                    {features.filter(f => baseline[f] !== modified[f]).length === 0 && (
                      <span className="text-xs text-muted-foreground">No changes — modify values in the right panel</span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </motion.div>
  );
}
