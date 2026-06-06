import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Brain, ArrowLeft, CheckCircle2, AlertCircle, BarChart3, Globe, Shield } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ALGO_NAMES } from '../constants';
import { predictOne, prepareInputForPrediction } from '../utils/mlEngine';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

export default function PublicPredictPage({ deployId, onBack }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({});
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);

  const fetchModelInfo = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/public/model/${deployId}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Model not found' }));
        setError(err.detail);
        return;
      }
      const data = await res.json();
      setModelInfo(data);
      const init = {};
      const allFeatures = [...(data.numeric_cols || []), ...(data.categorical_cols || [])];
      allFeatures.forEach(f => { init[f] = ''; });
      setFormData(init);
    } catch (e) { setError('Failed to load model'); }
    setLoading(false);
  }, [deployId]);

  useEffect(() => { fetchModelInfo(); }, [fetchModelInfo]);

  const handlePredict = async () => {
    setPredicting(true);
    setResult(null);
    try {
      const md = modelInfo?.model_data_full;
      if (md) {
        // Client-side prediction using the JS ML engine
        const row = {};
        (md.numericCols || []).forEach(col => { row[col] = Number(formData[col]) || 0; });
        (md.categoricalCols || []).forEach(col => { row[col] = formData[col] || ''; });
        const fvs = prepareInputForPrediction([row], md);
        if (fvs && fvs.length) {
          const prediction = predictOne(md, fvs[0]);
          setResult({ prediction });
        } else {
          setResult({ error: 'Failed to prepare input' });
        }
      } else {
        // Fallback: server-side prediction
        const res = await fetch(`${API_URL}/api/public/predict/${deployId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: formData }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');
        setResult(data);
      }
    } catch (e) { setResult({ error: e.message }); }
    setPredicting(false);
  };

  // Increment prediction count on backend (fire-and-forget)
  const incrementCount = useCallback(() => {
    fetch(`${API_URL}/api/public/predict/${deployId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: formData }),
    }).catch(() => {});
  }, [deployId, formData]);

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-violet-50 to-fuchsia-50 dark:from-zinc-950 dark:to-zinc-900">
      <div className="animate-pulse text-center">
        <Brain className="h-12 w-12 mx-auto mb-3 text-violet-500 animate-bounce" />
        <p className="text-muted-foreground">Loading model...</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-red-50 to-orange-50 dark:from-zinc-950 dark:to-zinc-900">
      <Card className="w-full max-w-md mx-4">
        <CardContent className="p-8 text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-3 text-destructive" />
          <h2 className="text-lg font-bold mb-2">Model Unavailable</h2>
          <p className="text-sm text-muted-foreground">{error}</p>
          {onBack && <Button className="mt-4" onClick={onBack}><ArrowLeft className="h-4 w-4 mr-2" />Go Back</Button>}
        </CardContent>
      </Card>
    </div>
  );

  const allFeatures = [...(modelInfo?.numeric_cols || []), ...(modelInfo?.categorical_cols || [])];
  const encodingMap = modelInfo?.encoding_map || {};

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50/50 via-white to-fuchsia-50/50 dark:from-zinc-950 dark:via-zinc-900 dark:to-zinc-950" data-testid="public-predict-page">
      {/* Header */}
      <div className="border-b bg-white/80 dark:bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
              <Brain className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg">{modelInfo?.name}</h1>
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Globe className="h-3 w-3" />Public Prediction
                <span>·</span>
                by {modelInfo?.owner}
              </p>
            </div>
          </div>
          <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-0">
            <Shield className="h-3 w-3 mr-1" />Live
          </Badge>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
        {/* Model Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: 'Algorithm', value: ALGO_NAMES[modelInfo?.algorithm] || modelInfo?.algorithm, color: 'text-violet-600' },
            { label: 'Type', value: modelInfo?.problem_type, color: 'text-blue-600' },
            { label: 'Features', value: allFeatures.length, color: 'text-emerald-600' },
            { label: 'Predictions', value: modelInfo?.prediction_count, color: 'text-amber-600' },
          ].map(s => (
            <Card key={s.label}>
              <CardContent className="p-3 text-center">
                <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
                <p className="text-[10px] text-muted-foreground">{s.label}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Metrics */}
        {modelInfo?.metrics && Object.keys(modelInfo.metrics).length > 0 && (
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm flex items-center gap-2"><BarChart3 className="h-4 w-4 text-violet-500" />Model Performance</CardTitle></CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {Object.entries(modelInfo.metrics).map(([k, v]) => (
                  <div key={k} className="px-3 py-1.5 rounded-lg bg-muted/50">
                    <span className="text-xs text-muted-foreground">{k}: </span>
                    <span className="text-sm font-semibold">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Prediction Form */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2"><Send className="h-4 w-4 text-fuchsia-500" />Make a Prediction</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2">
              {/* Numeric features */}
              {(modelInfo?.numeric_cols || []).map(feat => (
                <div key={feat}>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">{feat}</label>
                  <input
                    type="number"
                    step="any"
                    value={formData[feat] || ''}
                    onChange={e => setFormData(prev => ({ ...prev, [feat]: e.target.value }))}
                    placeholder={`Enter ${feat}`}
                    className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-violet-500/30 transition-shadow"
                    data-testid={`input-${feat}`}
                  />
                </div>
              ))}
              {/* Categorical features */}
              {(modelInfo?.categorical_cols || []).map(feat => (
                <div key={feat}>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">{feat}</label>
                  <select
                    value={formData[feat] || ''}
                    onChange={e => setFormData(prev => ({ ...prev, [feat]: e.target.value }))}
                    className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-violet-500/30 transition-shadow"
                    data-testid={`input-${feat}`}
                  >
                    <option value="">-- Select --</option>
                    {(encodingMap[feat] || []).map(val => <option key={val} value={val}>{val}</option>)}
                  </select>
                </div>
              ))}
            </div>
            <Button className="w-full mt-4 gap-2 bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600 h-11"
              onClick={handlePredict} disabled={predicting} data-testid="public-predict-btn">
              {predicting ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />Predicting...</> : <><Send className="h-4 w-4" />Get Prediction</>}
            </Button>
          </CardContent>
        </Card>

        {/* Result */}
        <AnimatePresence>
          {result && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
              <Card className={result.error ? 'border-destructive' : 'border-emerald-300 dark:border-emerald-700'} data-testid="prediction-result">
                <CardContent className="p-6">
                  {result.error ? (
                    <div className="flex items-center gap-3 text-destructive">
                      <AlertCircle className="h-6 w-6" /><div><p className="font-bold">Prediction Failed</p><p className="text-sm">{result.error}</p></div>
                    </div>
                  ) : (
                    <div className="text-center">
                      <CheckCircle2 className="h-10 w-10 text-emerald-500 mx-auto mb-3" />
                      <p className="text-sm text-muted-foreground mb-1">Predicted {modelInfo?.target}</p>
                      <p className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-fuchsia-600 bg-clip-text text-transparent" data-testid="prediction-value">
                        {typeof result.prediction === 'number' ? result.prediction.toFixed(4) : String(result.prediction)}
                      </p>
                      {result.probabilities && (
                        <div className="mt-4 flex flex-wrap justify-center gap-2">
                          {Object.entries(result.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="px-3 py-1.5 rounded-lg bg-muted/50">
                              <span className="text-xs text-muted-foreground">Class {cls}: </span>
                              <span className="text-sm font-semibold">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {modelInfo?.description && (
          <p className="text-xs text-muted-foreground text-center">{modelInfo.description}</p>
        )}
        <p className="text-[10px] text-muted-foreground/50 text-center">Powered by AutoML Master</p>
      </div>
    </div>
  );
}
