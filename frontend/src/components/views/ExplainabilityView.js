import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Eye, Zap, Brain, BarChart3, Target, Layers, Activity, Info, AlertCircle, GitBranch, Sparkles, Lightbulb, ArrowUpRight, ChevronRight, FileText, Download, CheckCircle2, TrendingUp, BarChart2, SplitSquareVertical, Trophy, Table2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell, ReferenceLine
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, CLUSTER_COLORS } from '../../constants';
import { getScoreColor, interpretMetric, arrayMin, arrayMax } from '../../utils/helpers';
import { importanceColor, valueToColor, buildWaterfallData, buildForceData } from '../../explainableAI';
import { MetricTip, HelpTip } from '../SmartTooltip';
import { useApp } from '../../context/AppContext';

export default function ExplainabilityView() {
  const {
    models, unsupervisedResult, setActiveView, xaiTab, setXaiTab, xaiLoading, xaiRow, setXaiRow,
    xaiDepFeature, setXaiDepFeature, shapGlobal, shapBeeswarm, shapLocal, shapDependence,
    limeResult, limeProbs, clusterShap, clusterBeeswarm, shapSummary, featureVsPred,
    clusterComparison, bestXaiModelIdx, selectedModelIdx, setSelectedModelIdx,
    dataProfile, handleRunSHAP, handleRunLIME, handleExplainPrediction,
    handleClusterExplain, getXaiModel, featureInsights, algorithm,
    smartRowSuggestions, targetColumn
  } = useApp();

  return (
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
          <div className="space-y-1.5"><label className="text-sm font-medium"><HelpTip text="Select the trained model you want to explain. The recommended model has the highest performance score.">Model</HelpTip></label>
            {bestXaiModelIdx >= 0 && (selectedModelIdx === -1 || selectedModelIdx === bestXaiModelIdx) && <div className="flex items-center gap-1.5 mb-1.5 px-2 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800" data-testid="xai-model-recommendation">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
              <span className="text-xs text-emerald-700 dark:text-emerald-300">Recommended: <strong>{ALGO_NAMES[models[bestXaiModelIdx].algorithm] || models[bestXaiModelIdx].algorithm}</strong> — best {models[bestXaiModelIdx].problemType === 'classification' ? 'accuracy' : 'R² score'}</span>
            </div>}
            <select value={selectedModelIdx === -1 ? bestXaiModelIdx >= 0 ? bestXaiModelIdx : models.length - 1 : selectedModelIdx} onChange={e => setSelectedModelIdx(Number(e.target.value))} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="xai-model-select">
              {models.map((m, i) => {
                const score = m.problemType === 'classification' ? (m.metrics?.accuracy ?? m.metrics?.f1 ?? 0) : (m.metrics?.r2 ?? 0);
                const isBest = i === bestXaiModelIdx;
                return <option key={m.modelId} value={i}>{isBest ? '\u2B50 ' : ''}{ALGO_NAMES[m.algorithm] || m.algorithm} — {m.problemType} ({(score * 100).toFixed(1)}%){isBest ? ' [Recommended]' : ''}</option>;
              })}
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
            <BarChart layout="vertical" data={shapGlobal.importance} margin={{ left: 30, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Mean |SHAP Value|', position: 'bottom', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={140} />
              <Tooltip content={({ active, payload }) => active && payload?.[0] ? <div className="bg-popover border rounded-lg shadow-lg p-3 text-xs"><p className="font-semibold">{payload[0].payload.feature}</p><p>Mean |SHAP|: <strong className="text-violet-600">{payload[0].payload.importance.toFixed(4)}</strong></p><p className="text-muted-foreground mt-1">Rank #{shapGlobal.importance.indexOf(payload[0].payload) + 1} of {shapGlobal.importance.length}</p></div> : null} />
              <Bar dataKey="importance" radius={[0, 6, 6, 0]} isAnimationActive={false}>
                {shapGlobal.importance.map((_, i) => <Cell key={i} fill={`hsl(${270 - (i / shapGlobal.importance.length) * 60}, 75%, ${50 + i * 2}%)`} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent></Card></motion.div>}

        {/* SHAP Plain-English Explanation */}
        {shapGlobal && shapGlobal.importance?.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="shap-explanation" className="border-violet-200/50 dark:border-violet-800/30 bg-gradient-to-r from-violet-50/50 to-purple-50/50 dark:from-violet-950/10 dark:to-purple-950/10 shadow-sm"><CardHeader>
          <CardTitle className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-amber-500" />What does this mean?</CardTitle>
        </CardHeader><CardContent>
          <div className="space-y-3 text-sm leading-relaxed">
            {(() => {
              const top3 = shapGlobal.importance.slice(0, 3);
              const target = targetColumn || 'the outcome';
              return (<>
                <p>The model relies most heavily on <strong className="text-violet-600 dark:text-violet-400">{top3[0]?.feature}</strong>{top3[1] ? <> and <strong className="text-violet-600 dark:text-violet-400">{top3[1]?.feature}</strong></> : null} to make its predictions about <strong>{target}</strong>.</p>
                {top3.length >= 3 && <p><strong className="text-violet-600 dark:text-violet-400">{top3[2]?.feature}</strong> also plays a notable role, though its influence is smaller compared to the top two drivers.</p>}
                <p className="text-muted-foreground">In simple terms: if you want to change the predicted {target}, focus on the top features — they have the most direct impact on the model's decision.</p>
              </>);
            })()}
          </div>
        </CardContent></Card></motion.div>}

        {/* SHAP Summary Plot (positive/negative) */}
        {shapSummary && <motion.div variants={fadeInUp}><Card data-testid="shap-summary-plot" className="border-pink-200/50 dark:border-pink-800/30 shadow-sm"><CardHeader>
          <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5 text-pink-500" />SHAP Summary Plot</CardTitle>
          <CardDescription>This plot shows the average direction of each feature's impact. Pink bars show the average positive push (increasing prediction), while cyan bars show the average negative pull (decreasing prediction). Together they reveal whether a feature typically pushes predictions up or down.</CardDescription>
        </CardHeader><CardContent>
          <ResponsiveContainer width="100%" height={Math.max(200, shapSummary.length * 40)}>
            <BarChart layout="vertical" data={shapSummary} margin={{ left: 30, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
              <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: 'Mean SHAP Value', position: 'bottom', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={140} />
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

        {/* SHAP Local Plain-English Explanation */}
        {shapLocal && <motion.div variants={fadeInUp}><Card data-testid="shap-local-explanation" className="border-pink-200/50 dark:border-pink-800/30 bg-gradient-to-r from-pink-50/50 to-rose-50/50 dark:from-pink-950/10 dark:to-rose-950/10 shadow-sm"><CardContent className="py-4">
          {(() => {
            const sorted = shapLocal.featureNames.map((f, i) => ({ feature: f, shap: shapLocal.shapValues[i], value: shapLocal.instance?.[i] })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
            const topPos = sorted.filter(s => s.shap > 0).slice(0, 2);
            const topNeg = sorted.filter(s => s.shap < 0).slice(0, 2);
            const target = targetColumn || 'the outcome';
            return (
              <div className="text-sm leading-relaxed space-y-2" data-testid="shap-local-why">
                <p className="font-semibold flex items-center gap-2"><Lightbulb className="h-4 w-4 text-amber-500" />Why did the model predict <strong className="text-primary">{shapLocal.instancePred?.toFixed(3)}</strong> for this record?</p>
                {topPos.length > 0 && <p>The prediction was <span className="text-pink-600 dark:text-pink-400 font-semibold">pushed higher</span> mainly because <strong>{topPos[0].feature}</strong>{topPos[0].value != null ? ` = ${topPos[0].value.toFixed(2)}` : ''} (+{topPos[0].shap.toFixed(3)}){topPos[1] ? <> and <strong>{topPos[1].feature}</strong>{topPos[1].value != null ? ` = ${topPos[1].value.toFixed(2)}` : ''} (+{topPos[1].shap.toFixed(3)})</> : ''}.</p>}
                {topNeg.length > 0 && <p>It was <span className="text-cyan-600 dark:text-cyan-400 font-semibold">pulled lower</span> by <strong>{topNeg[0].feature}</strong>{topNeg[0].value != null ? ` = ${topNeg[0].value.toFixed(2)}` : ''} ({topNeg[0].shap.toFixed(3)}){topNeg[1] ? <> and <strong>{topNeg[1].feature}</strong>{topNeg[1].value != null ? ` = ${topNeg[1].value.toFixed(2)}` : ''} ({topNeg[1].shap.toFixed(3)})</> : ''}.</p>}
                <p className="text-muted-foreground">The base prediction was {shapLocal.basePred?.toFixed(3)}. The features above shifted it to the final value of {shapLocal.instancePred?.toFixed(3)}.</p>
              </div>
            );
          })()}
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

        {/* LIME Plain-English Explanation */}
        {limeResult && <motion.div variants={fadeInUp}><Card data-testid="lime-explanation" className="border-emerald-200/50 dark:border-emerald-800/30 bg-gradient-to-r from-emerald-50/50 to-teal-50/50 dark:from-emerald-950/10 dark:to-teal-950/10 shadow-sm"><CardContent className="py-4">
          {(() => {
            const sorted = limeResult.contributions.slice(0, 10).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
            const topPos = sorted.filter(c => c.contribution > 0).slice(0, 2);
            const topNeg = sorted.filter(c => c.contribution < 0).slice(0, 2);
            const target = targetColumn || 'the outcome';
            return (
              <div className="text-sm leading-relaxed space-y-2" data-testid="lime-why">
                <p className="font-semibold flex items-center gap-2"><Lightbulb className="h-4 w-4 text-amber-500" />Why this prediction? (LIME explanation)</p>
                <p>For this specific record, the model predicted <strong className="text-emerald-600">{limeResult.prediction?.toFixed(3)}</strong> for <strong>{target}</strong>.</p>
                {topPos.length > 0 && <p>The biggest factors <span className="text-emerald-600 font-semibold">supporting</span> this prediction: <strong>{topPos[0].feature}</strong> (+{topPos[0].contribution.toFixed(3)}){topPos[1] ? <> and <strong>{topPos[1].feature}</strong> (+{topPos[1].contribution.toFixed(3)})</> : ''}.</p>}
                {topNeg.length > 0 && <p>Factors <span className="text-rose-500 font-semibold">working against</span> the prediction: <strong>{topNeg[0].feature}</strong> ({topNeg[0].contribution.toFixed(3)}){topNeg[1] ? <> and <strong>{topNeg[1].feature}</strong> ({topNeg[1].contribution.toFixed(3)})</> : ''}.</p>}
                <p className="text-muted-foreground">LIME creates a simple approximation of the model's behavior around this data point, making complex model decisions easy to understand.</p>
              </div>
            );
          })()}
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
  );
}
