import React from 'react';
import { motion } from 'framer-motion';
import { Settings2, Droplets, BarChart3, Scissors, Filter, CheckCircle2, Info } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
  const { dataProfile, preprocessConfig, setPreprocessConfig, columns, targetColumn, preprocessLog } = useApp();

  if (!dataProfile) {
    return (
      <motion.div {...fadeInUp} className="p-8 text-center" data-testid="preprocess-empty">
        <Settings2 className="h-16 w-16 mx-auto mb-4 text-muted-foreground/30" />
        <h3 className="text-lg font-bold mb-2">Load a Dataset First</h3>
        <p className="text-sm text-muted-foreground">Upload a CSV or pick a sample dataset to configure preprocessing.</p>
      </motion.div>
    );
  }

  const totalMissing = Object.values(dataProfile.missingCounts || {}).reduce((a, b) => a + b, 0);
  const numericCols = dataProfile.numericColumns || [];
  const features = columns.filter(c => c !== targetColumn);

  const update = (key, value) => setPreprocessConfig(prev => ({ ...prev, [key]: value }));
  const toggleExclude = (col) => {
    setPreprocessConfig(prev => {
      const ex = prev.excludeFeatures || [];
      return { ...prev, excludeFeatures: ex.includes(col) ? ex.filter(c => c !== col) : [...ex, col] };
    });
  };

  const activeSteps = [
    preprocessConfig.missingValues !== 'none' && 'Missing Values',
    preprocessConfig.scaling !== 'none' && 'Feature Scaling',
    preprocessConfig.outlierMethod !== 'none' && 'Outlier Treatment',
    (preprocessConfig.excludeFeatures?.length || 0) > 0 && 'Feature Selection',
  ].filter(Boolean);

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
                  {activeSteps.length > 0 ? `${activeSteps.length} active step(s): ${activeSteps.join(' → ')}` : 'No preprocessing steps configured — using raw data'}
                </p>
              </div>
            </div>
            <Badge variant="outline" className="text-xs">{dataProfile.rowCount?.toLocaleString()} rows · {features.length} features</Badge>
          </div>
        </CardContent>
      </Card>

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
              <span>Scaling is recommended for KNN, SVM, and Ridge. Tree-based models (Decision Tree, Random Forest, Gradient Boosting) are scale-invariant.</span>
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
    </motion.div>
  );
}
