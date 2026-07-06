import React, { useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Settings2, Droplets, BarChart3, Scissors, Filter, CheckCircle2, Info, Lightbulb, Sparkles, Check, ArrowRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { fadeInUp } from '../../constants';
import { useApp } from '../../context/AppContext';

const MISSING_OPTIONS = [
  { value: 'auto', label: 'Auto (median for numeric, mode for categorical)', desc: 'Smart default' },
  { value: 'mean', label: 'Mean', desc: 'Average value for numeric columns' },
  { value: 'median', label: 'Median', desc: 'Middle value — robust to outliers' },
  { value: 'mode', label: 'Mode', desc: 'Most frequent value' },
  { value: 'zero', label: 'Zero / Empty', desc: 'Fill with 0 or blank' },
  { value: 'drop', label: 'Drop Rows', desc: 'Remove rows with any missing value' },
  { value: 'none', label: 'None', desc: 'Skip missing value handling' },
];

const SCALING_OPTIONS = [
  { value: 'none', label: 'None', desc: 'Use raw feature values' },
  { value: 'standard', label: 'Standardize (Z-Score)', desc: 'Mean=0, Std=1 — good for SVM, KNN' },
  { value: 'minmax', label: 'Min-Max Normalize', desc: 'Scale to [0, 1] range' },
];

const OUTLIER_OPTIONS = [
  { value: 'none', label: 'None', desc: 'Keep all data points' },
  { value: 'clip', label: 'Clip (Winsorize)', desc: 'Cap extreme values at IQR bounds' },
  { value: 'remove', label: 'Remove', desc: 'Drop rows with outlier values' },
];

export default function PreprocessView() {
  const { dataProfile, preprocessConfig, setPreprocessConfig, columns, targetColumn, preprocessLog, setActiveView } = useApp();

  const totalMissing = Object.values(dataProfile?.missingCounts || {}).reduce((a, b) => a + b, 0);
  const numericCols = dataProfile?.numericColumns || [];
  const features = columns.filter(c => c !== targetColumn);

  const update = useCallback((key, value) => {
    setPreprocessConfig(prev => ({ ...prev, [key]: value }));
    const labels = {
      missingValues: 'Missing Values', scaling: 'Feature Scaling',
      outlierMethod: 'Outlier Treatment', outlierThreshold: null,
    };
    if (labels[key] !== undefined && labels[key] !== null) {
      toast.success(`${labels[key]} updated`, { duration: 1500 });
    }
  }, [setPreprocessConfig]);
  const toggleExclude = (col) => {
    setPreprocessConfig(prev => {
      const ex = prev.excludeFeatures || [];
      const excluding = !ex.includes(col);
      return { ...prev, excludeFeatures: excluding ? [...ex, col] : ex.filter(c => c !== col) };
    });
  };

  const activeSteps = [
    preprocessConfig.missingValues !== 'none' && 'Missing Values',
    preprocessConfig.scaling !== 'none' && 'Feature Scaling',
    preprocessConfig.outlierMethod !== 'none' && 'Outlier Treatment',
    (preprocessConfig.excludeFeatures?.length || 0) > 0 && 'Feature Selection',
  ].filter(Boolean);

  // ---- Smart Recommendations Engine ----
  const recommendations = useMemo(() => {
    if (!dataProfile) return [];
    const recs = [];
    const rows = dataProfile.rows || [];
    const headers = dataProfile.headers || [];
    const rowCount = dataProfile.rowCount || rows.length;
    const nc = dataProfile.numericColumns || [];

    // 1. Missing values
    const missingCols = Object.entries(dataProfile.missingCounts || {}).filter(([, v]) => v > 0);
    const totalMissingPct = rowCount > 0 ? missingCols.reduce((a, [, v]) => a + v, 0) / (rowCount * headers.length) * 100 : 0;
    if (missingCols.length > 0) {
      const highMissing = missingCols.filter(([, v]) => v / rowCount > 0.3);
      if (highMissing.length > 0) {
        recs.push({ key: 'missing', severity: 'high',
          title: `${missingCols.length} column(s) have missing values (${totalMissingPct.toFixed(1)}% overall)`,
          detail: `${highMissing.map(([c, v]) => `"${c}" (${(v/rowCount*100).toFixed(0)}%)`).join(', ')} have >30% missing. Recommend median fill.`,
          action: { key: 'missingValues', value: 'median' },
        });
      } else {
        recs.push({ key: 'missing', severity: 'medium',
          title: `${missingCols.length} column(s) have missing values`,
          detail: `Affected: ${missingCols.slice(0, 4).map(([c, v]) => `"${c}" (${v})`).join(', ')}${missingCols.length > 4 ? '...' : ''}. Auto-fill recommended.`,
          action: { key: 'missingValues', value: 'auto' },
        });
      }
    }

    // 2. Outliers
    const outlierCols = [];
    nc.forEach(col => {
      const vals = rows.map(r => Number(r[col])).filter(v => !isNaN(v)).sort((a, b) => a - b);
      if (vals.length < 10) return;
      const q1 = vals[Math.floor(vals.length * 0.25)];
      const q3 = vals[Math.floor(vals.length * 0.75)];
      const iqr = q3 - q1;
      if (iqr === 0) return;
      const oc = vals.filter(v => v < q1 - 1.5 * iqr || v > q3 + 1.5 * iqr).length;
      if (oc / vals.length > 0.03) outlierCols.push({ col, pct: (oc / vals.length * 100).toFixed(1) });
    });
    if (outlierCols.length > 0) {
      recs.push({ key: 'outliers', severity: outlierCols.some(c => parseFloat(c.pct) > 10) ? 'high' : 'medium',
        title: `Outliers detected in ${outlierCols.length} column(s)`,
        detail: `${outlierCols.slice(0, 3).map(c => `"${c.col}" (${c.pct}%)`).join(', ')}${outlierCols.length > 3 ? '...' : ''}. Clipping preserves data while reducing extremes.`,
        action: { key: 'outlierMethod', value: 'clip' },
      });
    }

    // 3. Scaling
    if (nc.length >= 2) {
      const ranges = nc.slice(0, 20).map(col => {
        const vals = rows.map(r => Number(r[col])).filter(v => !isNaN(v));
        return vals.length ? Math.max(...vals) - Math.min(...vals) : 0;
      }).filter(r => r > 0);
      if (ranges.length >= 2) {
        const ratio = Math.max(...ranges) / (Math.min(...ranges) || 1);
        if (ratio > 100) {
          recs.push({ key: 'scaling', severity: 'medium',
            title: 'Feature scales vary widely',
            detail: `Range ratio is ${ratio.toFixed(0)}x. Standardization recommended for KNN, SVM, Ridge.`,
            action: { key: 'scaling', value: 'standard' },
          });
        }
      }
    }

    // 4. High-cardinality categorical
    const catCols = headers.filter(h => !nc.includes(h) && h !== targetColumn);
    const highCardCols = catCols.filter(col => new Set(rows.map(r => r[col] ?? '')).size > 20);
    if (highCardCols.length > 0) {
      recs.push({ key: 'highcard', severity: 'low',
        title: `${highCardCols.length} high-cardinality categorical column(s)`,
        detail: `${highCardCols.slice(0, 3).map(c => `"${c}"`).join(', ')} have many unique values. May cause overfitting.`,
        suggestExclude: highCardCols,
      });
    }

    // 5. Low-variance
    const lowVarCols = nc.filter(col => {
      const vals = rows.map(r => Number(r[col])).filter(v => !isNaN(v));
      if (vals.length < 5) return false;
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      return vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length < 0.001;
    });
    if (lowVarCols.length > 0) {
      recs.push({ key: 'lowvar', severity: 'low',
        title: `${lowVarCols.length} near-constant feature(s)`,
        detail: `${lowVarCols.slice(0, 3).map(c => `"${c}"`).join(', ')} add noise without predictive value.`,
        suggestExclude: lowVarCols,
      });
    }

    return recs;
  }, [dataProfile, targetColumn]);

  const isRecApplied = useCallback((rec) => {
    if (rec.action) {
      if (preprocessConfig[rec.action.key] !== rec.action.value) return false;
    }
    if (rec.suggestExclude) {
      const excluded = preprocessConfig.excludeFeatures || [];
      if (!rec.suggestExclude.every(c => excluded.includes(c))) return false;
    }
    return true;
  }, [preprocessConfig]);

  const applyRecommendation = useCallback((rec) => {
    if (isRecApplied(rec)) return;
    if (rec.action) update(rec.action.key, rec.action.value);
    if (rec.suggestExclude) {
      setPreprocessConfig(prev => ({
        ...prev, excludeFeatures: [...new Set([...(prev.excludeFeatures || []), ...rec.suggestExclude])],
      }));
    }
    toast.success(`Applied: ${rec.title}`);
  }, [setPreprocessConfig, update, isRecApplied]);

  const applyAllRecommendations = useCallback(() => {
    const pending = recommendations.filter(rec => !isRecApplied(rec));
    if (pending.length === 0) {
      toast.info('All recommendations already applied');
      return;
    }
    pending.forEach(rec => {
      if (rec.action) update(rec.action.key, rec.action.value);
      if (rec.suggestExclude) {
        setPreprocessConfig(prev => ({
          ...prev, excludeFeatures: [...new Set([...(prev.excludeFeatures || []), ...rec.suggestExclude])],
        }));
      }
    });
    toast.success(`Applied ${pending.length} recommendation${pending.length > 1 ? 's' : ''}`);
  }, [recommendations, isRecApplied, update, setPreprocessConfig]);

  if (!dataProfile) {
    return (
      <motion.div {...fadeInUp} className="p-8 text-center" data-testid="preprocess-empty">
        <Settings2 className="h-16 w-16 mx-auto mb-4 text-muted-foreground/30" />
        <h3 className="text-lg font-bold mb-2">Load a Dataset First</h3>
        <p className="text-sm text-muted-foreground">Upload a CSV or pick a sample dataset to configure preprocessing.</p>
      </motion.div>
    );
  }

  return (
    <motion.div {...fadeInUp} className="p-8 space-y-6" data-testid="preprocess-view">
      {/* Pipeline Summary */}
      <Card>
        <CardContent className="p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
                <Settings2 className="h-5 w-5 text-white" />
              </div>
              <div>
                <h3 className="font-bold" data-testid="preprocess-title">Preprocessing Pipeline</h3>
                <p className="text-xs text-muted-foreground">
                  {activeSteps.length > 0 ? `${activeSteps.length} active step(s): ${activeSteps.join(' \u2192 ')}` : 'No preprocessing steps configured \u2014 using raw data'}
                </p>
              </div>
            </div>
            <Badge variant="outline" className="text-xs">{dataProfile.rowCount?.toLocaleString()} rows &middot; {features.length} features</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Smart Recommendations */}
      {recommendations.length > 0 && (() => {
        const allApplied = recommendations.every(rec => isRecApplied(rec));
        const pendingCount = recommendations.filter(rec => !isRecApplied(rec)).length;
        return (
        <Card className={`border-amber-200 dark:border-amber-800/50 ${allApplied ? 'bg-gradient-to-r from-emerald-50/50 to-green-50/30 dark:from-emerald-950/20 dark:to-green-950/10 border-emerald-200 dark:border-emerald-800/50' : 'bg-gradient-to-r from-amber-50/50 to-orange-50/30 dark:from-amber-950/20 dark:to-orange-950/10'}`} data-testid="smart-recommendations">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                {allApplied ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <Lightbulb className="h-4 w-4 text-amber-500" />}
                Smart Recommendations
                {allApplied
                  ? <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-0 text-[10px]">All applied</Badge>
                  : <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 border-0 text-[10px]">{pendingCount} pending</Badge>
                }
              </CardTitle>
              {!allApplied && (
                <Button size="sm" className="h-7 text-xs gap-1.5 bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-600 hover:to-orange-600" onClick={applyAllRecommendations} data-testid="apply-all-recommendations">
                  <Sparkles className="h-3 w-3" />Apply All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-2.5 pt-0">
            {recommendations.map(rec => {
              const applied = isRecApplied(rec);
              return (
              <div key={rec.key} className={`flex items-start gap-3 p-3 rounded-lg border transition-all ${applied ? 'bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800/50' : 'bg-white/60 dark:bg-zinc-900/40 border-white/80 dark:border-zinc-800'}`}>
                <div className="mt-0.5">
                  {applied ? <Badge className="text-[9px] px-1.5 py-0 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-0 gap-0.5"><Check className="h-2.5 w-2.5" />DONE</Badge> : rec.severity === 'high' ? <Badge variant="destructive" className="text-[9px] px-1.5 py-0">HIGH</Badge> : rec.severity === 'medium' ? <Badge className="text-[9px] px-1.5 py-0 bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 border-0">MED</Badge> : <Badge variant="outline" className="text-[9px] px-1.5 py-0">LOW</Badge>}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${applied ? 'text-emerald-700 dark:text-emerald-400' : ''}`}>{rec.title}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5">{rec.detail}</p>
                </div>
                {applied
                  ? <span className="shrink-0 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1" data-testid={`applied-rec-${rec.key}`}><CheckCircle2 className="h-3.5 w-3.5" />Applied</span>
                  : <Button variant="outline" size="sm" className="shrink-0 h-7 text-[11px]" onClick={() => applyRecommendation(rec)} data-testid={`apply-rec-${rec.key}`}>Apply</Button>
                }
              </div>
              );
            })}
          </CardContent>
        </Card>
        );
      })()}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Step 1: Missing Values */}
        <Card data-testid="preprocess-missing">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Droplets className="h-4 w-4 text-blue-500" />Missing Values
              {totalMissing > 0 ? <Badge variant="destructive" className="ml-auto text-[10px]">{totalMissing} missing</Badge> : <Badge className="ml-auto text-[10px] bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-0">Complete</Badge>}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {MISSING_OPTIONS.map(opt => (
              <label key={opt.value} className={`flex items-start gap-3 p-2.5 rounded-lg cursor-pointer border transition-all ${preprocessConfig.missingValues === opt.value ? 'border-blue-300 bg-blue-50/50 dark:border-blue-700 dark:bg-blue-950/20' : 'border-transparent hover:bg-muted/30'}`} data-testid={`missing-${opt.value}`}>
                <input type="radio" name="missing" value={opt.value} checked={preprocessConfig.missingValues === opt.value} onChange={() => update('missingValues', opt.value)} className="mt-0.5 accent-blue-500" />
                <div>
                  <span className="text-sm font-medium">{opt.label}</span>
                  <span className="block text-[11px] text-muted-foreground">{opt.desc}</span>
                </div>
              </label>
            ))}
          </CardContent>
        </Card>

        {/* Step 2: Feature Scaling */}
        <Card data-testid="preprocess-scaling">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-violet-500" />Feature Scaling
              <Badge variant="outline" className="ml-auto text-[10px]">{numericCols.length} numeric</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {SCALING_OPTIONS.map(opt => (
              <label key={opt.value} className={`flex items-start gap-3 p-2.5 rounded-lg cursor-pointer border transition-all ${preprocessConfig.scaling === opt.value ? 'border-violet-300 bg-violet-50/50 dark:border-violet-700 dark:bg-violet-950/20' : 'border-transparent hover:bg-muted/30'}`} data-testid={`scaling-${opt.value}`}>
                <input type="radio" name="scaling" value={opt.value} checked={preprocessConfig.scaling === opt.value} onChange={() => update('scaling', opt.value)} className="mt-0.5 accent-violet-500" />
                <div>
                  <span className="text-sm font-medium">{opt.label}</span>
                  <span className="block text-[11px] text-muted-foreground">{opt.desc}</span>
                </div>
              </label>
            ))}
            <div className="flex items-start gap-2 mt-2 p-2 rounded-lg bg-muted/30 text-[11px] text-muted-foreground">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>Scaling is recommended for KNN, SVM, and Ridge. Tree-based models are scale-invariant.</span>
            </div>
          </CardContent>
        </Card>

        {/* Step 3: Outlier Treatment */}
        <Card data-testid="preprocess-outliers">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Scissors className="h-4 w-4 text-red-500" />Outlier Treatment
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {OUTLIER_OPTIONS.map(opt => (
              <label key={opt.value} className={`flex items-start gap-3 p-2.5 rounded-lg cursor-pointer border transition-all ${preprocessConfig.outlierMethod === opt.value ? 'border-red-300 bg-red-50/50 dark:border-red-700 dark:bg-red-950/20' : 'border-transparent hover:bg-muted/30'}`} data-testid={`outlier-${opt.value}`}>
                <input type="radio" name="outlier" value={opt.value} checked={preprocessConfig.outlierMethod === opt.value} onChange={() => update('outlierMethod', opt.value)} className="mt-0.5 accent-red-500" />
                <div>
                  <span className="text-sm font-medium">{opt.label}</span>
                  <span className="block text-[11px] text-muted-foreground">{opt.desc}</span>
                </div>
              </label>
            ))}
            {preprocessConfig.outlierMethod !== 'none' && (
              <div className="pt-2">
                <label className="text-xs font-medium text-muted-foreground">IQR Multiplier: {preprocessConfig.outlierThreshold}</label>
                <input type="range" min="1" max="3" step="0.1" value={preprocessConfig.outlierThreshold} onChange={e => update('outlierThreshold', parseFloat(e.target.value))} className="w-full accent-red-500 mt-1" data-testid="outlier-threshold" />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  <span>1.0 (aggressive)</span><span>3.0 (conservative)</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Step 4: Feature Selection */}
        <Card data-testid="preprocess-features">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Filter className="h-4 w-4 text-emerald-500" />Feature Selection
              {(preprocessConfig.excludeFeatures?.length || 0) > 0 && <Badge variant="outline" className="ml-auto text-[10px]">{preprocessConfig.excludeFeatures.length} excluded</Badge>}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-[11px] text-muted-foreground mb-3">Uncheck features to exclude them from training. Target column is automatically excluded.</p>
            <div className="max-h-[280px] overflow-y-auto space-y-1 pr-1">
              {features.map(col => {
                const excluded = preprocessConfig.excludeFeatures?.includes(col);
                return (
                  <label key={col} className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-md cursor-pointer transition-all ${excluded ? 'opacity-50 bg-muted/20 line-through' : 'hover:bg-muted/30'}`} data-testid={`feat-${col}`}>
                    <input type="checkbox" checked={!excluded} onChange={() => toggleExclude(col)} className="accent-emerald-500 rounded" />
                    <span className="text-sm truncate">{col}</span>
                    <Badge variant="outline" className="ml-auto text-[9px] shrink-0">{numericCols.includes(col) ? 'Num' : 'Cat'}</Badge>
                  </label>
                );
              })}
            </div>
            {features.length > 5 && (
              <div className="flex gap-2 mt-3">
                <Button variant="outline" size="sm" className="text-xs h-7" onClick={() => setPreprocessConfig(p => ({ ...p, excludeFeatures: [] }))} data-testid="select-all-features">Select All</Button>
                <Button variant="outline" size="sm" className="text-xs h-7" onClick={() => setPreprocessConfig(p => ({ ...p, excludeFeatures: features.filter(c => !numericCols.includes(c)) }))} data-testid="select-numeric-only">Numeric Only</Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Preprocessing Log */}
      {preprocessLog && preprocessLog.length > 0 && (
        <Card data-testid="preprocess-log">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />Last Preprocessing Log
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1.5">
              {preprocessLog.map((entry, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <Badge variant="outline" className="text-[9px] shrink-0">{entry.step}</Badge>
                  <span className="text-muted-foreground">{entry.message}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Proceed to Training CTA */}
      <Card className="border-2 border-primary/30 bg-gradient-to-r from-primary/5 to-violet-500/5" data-testid="proceed-to-training-card">
        <CardContent className="p-5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary to-violet-500 flex items-center justify-center shrink-0">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div className="min-w-0">
                <h3 className="font-bold text-sm">Ready to Train</h3>
                <p className="text-xs text-muted-foreground truncate">
                  {activeSteps.length > 0
                    ? `${activeSteps.length} preprocessing step(s) will be applied during training: ${activeSteps.join(', ')}`
                    : 'No preprocessing configured — raw data will be used for training'}
                </p>
              </div>
            </div>
            <Button onClick={() => setActiveView('analysis')} className="shrink-0 gap-2" data-testid="proceed-to-training-btn">
              Proceed to Training <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
