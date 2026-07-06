import React from 'react';
import { motion } from 'framer-motion';
import { Database, Zap, TrendingUp, Activity, Brain, Trophy, BookOpen, Lightbulb, ArrowUpRight, Clock, Shield, Sparkles, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, ALGO_COLORS } from '../../constants';
import { MetricTip } from '../SmartTooltip';
import { useApp } from '../../context/AppContext';

export default function DashboardView() {
  const {
    models, stats, dataProfile, datasetScan, showGuide, setShowGuide,
    setActiveView, businessInterpretation, topModels, StatCard,
    leaderboardEntries, historyList, historyLoading, handleLoadSnapshot
  } = useApp();

  // Top 5 from leaderboard
  const topLeaderboard = [...leaderboardEntries]
    .sort((a, b) => {
      const sa = a.problem_type === 'classification' ? (a.metrics?.f1 || a.metrics?.accuracy || 0) : (a.metrics?.r2 || 0);
      const sb = b.problem_type === 'classification' ? (b.metrics?.f1 || b.metrics?.accuracy || 0) : (b.metrics?.r2 || 0);
      return sb - sa;
    })
    .slice(0, 5);

  return (
  <motion.div key="dashboard" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="dashboard-view">
    {models.length === 0 ? (
      <motion.div variants={fadeInUp} className="space-y-6">
        <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
          <Database className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
          <h3 className="text-lg font-semibold mb-2" data-testid="dashboard-empty-title">{dataProfile ? 'No Models Trained Yet' : 'No Dataset Uploaded Yet'}</h3>
          <p className="text-muted-foreground mb-6 max-w-md mx-auto text-sm" data-testid="dashboard-empty-msg">{dataProfile ? 'Dataset loaded. Go to Analysis to select a target variable and train your first model.' : 'Upload a CSV file or pick a sample dataset. The system will guide you through each step.'}</p>
          <Button size="lg" onClick={() => setActiveView('analysis')} data-testid="train-first-model-btn"><Zap className="h-4 w-4 mr-2" />{dataProfile ? 'Train Your First Model' : 'Upload Dataset'}</Button>
          {!showGuide && <div className="mt-4"><Button variant="link" size="sm" onClick={() => setShowGuide(true)} className="text-blue-600" data-testid="show-guide-from-dashboard"><BookOpen className="h-3.5 w-3.5 mr-1.5 inline" />New here? Open the Getting Started Guide</Button></div>}
        </CardContent></Card>
        {datasetScan && (
          <Card data-testid="dataset-health-empty"><CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" />Dataset Health</CardTitle></CardHeader><CardContent>
            <div className={`p-3 rounded-lg border-2 flex items-center gap-3 ${datasetScan.ready ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20' : 'border-orange-500 bg-orange-50 dark:bg-orange-950/20'}`}>
              {datasetScan.ready ? <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" /> : <XCircle className="h-5 w-5 text-orange-600 shrink-0" />}
              <p className="font-medium text-sm">{datasetScan.ready ? 'Dataset Ready for Training' : 'Dataset Needs Cleaning'}</p>
              <Badge className="ml-auto" variant={datasetScan.score >= 80 ? 'default' : datasetScan.score >= 50 ? 'secondary' : 'destructive'}>{datasetScan.score}/100</Badge>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-sm">
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Rows</p><p className="font-bold text-lg">{datasetScan.rows}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Columns</p><p className="font-bold text-lg">{datasetScan.columns}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Missing</p><p className="font-bold text-lg">{datasetScan.totalMissing}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Outliers</p><p className="font-bold text-lg">{datasetScan.totalOutliers}</p></div>
            </div>
          </CardContent></Card>
        )}

        {/* Saved Analyses (from previous sessions) */}
        {historyList.length > 0 && (
          <Card data-testid="dashboard-saved-analyses">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-base"><Clock className="h-4 w-4 text-blue-500" />Saved Analyses</CardTitle>
                  <CardDescription className="text-xs">{historyList.length} saved analysis session{historyList.length !== 1 ? 's' : ''} — click to resume</CardDescription>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setActiveView('history')} className="text-xs" data-testid="view-all-history">
                  View All <ArrowUpRight className="h-3 w-3 ml-1" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                {historyList.slice(0, 5).map((snap, i) => (
                  <div key={snap.snapshot_id || snap._id || i}
                    className="flex items-center gap-3 p-2.5 rounded-lg border hover:bg-accent/50 transition-colors cursor-pointer"
                    onClick={() => handleLoadSnapshot(snap.snapshot_id || snap._id)}
                    data-testid={`saved-analysis-${i}`}>
                    <div className="h-8 w-8 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center shrink-0">
                      <Database className="h-4 w-4 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{snap.dataset_name || snap.name || 'Unnamed Analysis'}</p>
                      <p className="text-xs text-muted-foreground">
                        {snap.created_at ? new Date(snap.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}
                        {snap.state?.trainingResult?.bestModel?.algorithm && ` · ${ALGO_NAMES[snap.state.trainingResult.bestModel.algorithm] || snap.state.trainingResult.bestModel.algorithm}`}
                      </p>
                    </div>
                    <ArrowUpRight className="h-4 w-4 text-muted-foreground shrink-0" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </motion.div>
    ) : (<>
      {/* Stat Cards */}
      <motion.div variants={staggerContainer} className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <StatCard title="Total Models" value={stats.totalModels} metricValue={stats.totalModels} icon={Database} />
        <StatCard title="Avg Score" value={`${(stats.avgScore * 100).toFixed(1)}%`} metricValue={`${(stats.avgScore * 100).toFixed(0)}%`} icon={TrendingUp} />
        <StatCard title="Best Algorithm" value={stats.bestAlgoByScore} icon={Trophy} />
        <StatCard title="Highest Score" value={`${(stats.highestScore * 100).toFixed(1)}%`} metricValue={`${(stats.highestScore * 100).toFixed(0)}%`} icon={Sparkles} />
        <StatCard title="Last Training" value={stats.lastTraining ? new Date(stats.lastTraining).toLocaleDateString() : '--'} icon={Clock} />
      </motion.div>

      {/* Quick Insights */}
      <motion.div variants={fadeInUp} className="grid gap-4 md:grid-cols-3" data-testid="quick-insights">
        <Card><CardContent className="p-4 flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center shrink-0"><Trophy className="h-5 w-5 text-emerald-600" /></div>
          <div><p className="text-xs text-muted-foreground">Best Performing</p><p className="font-semibold text-sm" data-testid="insight-best">{stats.bestAlgoByScore}</p></div>
        </CardContent></Card>
        <Card><CardContent className="p-4 flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center shrink-0"><Activity className="h-5 w-5 text-blue-600" /></div>
          <div><p className="text-xs text-muted-foreground">Most Used</p><p className="font-semibold text-sm" data-testid="insight-most-used">{stats.mostUsedAlgo}</p></div>
        </CardContent></Card>
        <Card><CardContent className="p-4 flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center shrink-0"><Sparkles className="h-5 w-5 text-amber-600" /></div>
          <div><p className="text-xs text-muted-foreground"><MetricTip metricKey="accuracy" value={stats.highestScore}>Highest Accuracy</MetricTip></p><p className="font-semibold text-sm" data-testid="insight-highest">{(stats.highestScore * 100).toFixed(1)}%</p></div>
        </CardContent></Card>
      </motion.div>

      {/* Business Interpretation */}
      {businessInterpretation && <motion.div variants={fadeInUp}><Card className="border-2 border-violet-500/30" data-testid="business-interpretation-card">
        <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2"><Lightbulb className="h-5 w-5 text-violet-500" />Business Interpretation</CardTitle>
          <CardDescription>Plain-English summary of what your data means</CardDescription></CardHeader>
        <CardContent><div className="space-y-2" data-testid="business-interpretation-text">
          {businessInterpretation.map((line, i) => <p key={i} className="text-sm leading-relaxed text-muted-foreground">{line}</p>)}
        </div></CardContent>
      </Card></motion.div>}

      {/* Dataset Health + Model Leaderboard */}
      <div className="grid gap-6 lg:grid-cols-2">
        <motion.div variants={fadeInUp}><Card className="h-full" data-testid="dataset-health-widget"><CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" />Dataset Health</CardTitle></CardHeader><CardContent>
          {datasetScan ? (<div className="space-y-4">
            <div className={`p-3 rounded-lg border-2 flex items-center gap-3 ${datasetScan.ready ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20' : 'border-orange-500 bg-orange-50 dark:bg-orange-950/20'}`} data-testid="dataset-readiness">
              {datasetScan.ready ? <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" /> : <XCircle className="h-5 w-5 text-orange-600 shrink-0" />}
              <p className="font-medium text-sm">{datasetScan.ready ? 'Dataset Ready for Training' : 'Dataset Needs Cleaning'}</p>
              <Badge className="ml-auto" variant={datasetScan.score >= 80 ? 'default' : datasetScan.score >= 50 ? 'secondary' : 'destructive'} data-testid="health-score">{datasetScan.score}/100</Badge>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Rows</p><p className="font-bold text-lg" data-testid="scan-rows">{datasetScan.rows}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Columns</p><p className="font-bold text-lg" data-testid="scan-cols">{datasetScan.columns}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Missing Values</p><p className="font-bold text-lg" data-testid="scan-missing">{datasetScan.totalMissing}</p></div>
              <div className="bg-muted/50 rounded-lg p-3"><p className="text-xs text-muted-foreground">Outliers</p><p className="font-bold text-lg" data-testid="scan-outliers">{datasetScan.totalOutliers}</p></div>
            </div>
            {datasetScan.warnings.length > 0 && <div className="space-y-1.5">{datasetScan.warnings.map((w, i) => <div key={i} className="flex items-center gap-2 text-xs text-orange-600"><AlertCircle className="h-3 w-3 shrink-0" />{w}</div>)}</div>}
          </div>) : (
            <div className="text-center py-8"><Shield className="h-10 w-10 text-muted-foreground/30 mx-auto mb-3" /><p className="text-sm text-muted-foreground">Load a dataset in Analysis to see health metrics</p></div>
          )}
        </CardContent></Card></motion.div>

        <motion.div variants={fadeInUp}><Card className="h-full" data-testid="model-leaderboard"><CardHeader><CardTitle className="flex items-center gap-2"><Trophy className="h-5 w-5" />Top Models</CardTitle><CardDescription>Top 5 by performance score</CardDescription></CardHeader><CardContent>
          <div className="space-y-2">{topModels.map((model, idx) => {
            const score = model.problemType === 'classification' ? (model.metrics?.accuracy || 0) : (model.metrics?.r2 || 0);
            return (<div key={model.modelId} className="flex items-center gap-3 p-2.5 rounded-lg border hover:bg-accent/50 transition-colors" data-testid={`top-model-${idx}`}>
              <Badge variant={idx === 0 ? 'default' : 'secondary'} className="shrink-0 w-7 justify-center">{idx + 1}</Badge>
              <div className="flex-1 min-w-0"><p className="font-medium text-sm truncate">{ALGO_NAMES[model.algorithm] || model.algorithm}</p><p className="text-xs text-muted-foreground">{model.problemType}{model.evalMode === 'cv' ? ' (CV)' : ''}</p></div>
              <span className="font-mono text-sm font-semibold shrink-0" style={{color: ALGO_COLORS[model.algorithm]}}>{(score * 100).toFixed(1)}%</span>
            </div>);
          })}</div>
        </CardContent></Card></motion.div>
      </div>

      {/* Charts Row */}
      <div className="grid gap-6 lg:grid-cols-2">
        <motion.div variants={fadeInUp}><Card className="h-[400px]" data-testid="model-performance-chart"><CardHeader><CardTitle>Model Performance</CardTitle><CardDescription>Score per trained model</CardDescription></CardHeader><CardContent>
          <ResponsiveContainer width="100%" height={270}><BarChart data={models.map((m, i) => ({
            name: `${ALGO_NAMES[m.algorithm] || m.algorithm}`.substring(0, 12), score: +((m.problemType === 'classification' ? (m.metrics?.accuracy || 0) : (m.metrics?.r2 || 0)) * 100).toFixed(1)
          }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-30} textAnchor="end" height={80} tick={{fontSize: 10}} /><YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} /><Tooltip formatter={v => `${v}%`} /><Bar dataKey="score" radius={[4, 4, 0, 0]}>{models.map((m, i) => <Cell key={i} fill={ALGO_COLORS[m.algorithm] || '#6b7280'} />)}</Bar></BarChart></ResponsiveContainer>
        </CardContent></Card></motion.div>

        <motion.div variants={fadeInUp}><Card className="h-[400px]" data-testid="algorithm-usage-chart"><CardHeader><CardTitle>Algorithm Usage</CardTitle><CardDescription>Training frequency per algorithm</CardDescription></CardHeader><CardContent>
          <ResponsiveContainer width="100%" height={270}><PieChart><Pie data={(() => {
            const c = {}; models.forEach(m => { const nm = ALGO_NAMES[m.algorithm] || m.algorithm; c[nm] = (c[nm] || 0) + 1; });
            return Object.entries(c).map(([name, value], i) => ({ name, value, fill: Object.values(ALGO_COLORS)[i % Object.values(ALGO_COLORS).length] }));
          })()} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({name, percent}) => `${name} ${(percent*100).toFixed(0)}%`} /><Tooltip /></PieChart></ResponsiveContainer>
        </CardContent></Card></motion.div>
      </div>

      {/* Training Timeline */}
      <motion.div variants={fadeInUp}><Card data-testid="training-timeline"><CardHeader><CardTitle>Training Timeline</CardTitle><CardDescription>Cumulative model count over time</CardDescription></CardHeader><CardContent>
        <ResponsiveContainer width="100%" height={250}><LineChart data={(() => {
          const dc = {}; models.forEach(m => { const d = m.createdAt ? new Date(m.createdAt).toLocaleDateString() : 'Today'; dc[d] = (dc[d] || 0) + 1; });
          let cum = 0; return Object.entries(dc).map(([date, count]) => { cum += count; return { date, count, cumulative: cum }; });
        })()}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="date" /><YAxis /><Tooltip /><Legend /><Line type="monotone" dataKey="cumulative" name="Total Models" stroke="hsl(var(--primary))" strokeWidth={2} dot={{fill:'hsl(var(--primary))'}} /><Line type="monotone" dataKey="count" name="Trained" stroke="#16a34a" strokeWidth={2} dot={{fill:'#16a34a'}} /></LineChart></ResponsiveContainer>
      </CardContent></Card></motion.div>

      {/* Recent Models Table */}
      <motion.div variants={fadeInUp}><Card data-testid="recent-models-table"><CardHeader><CardTitle>Recent Models</CardTitle><CardDescription>Latest trained models</CardDescription></CardHeader><CardContent>
        <div className="rounded-md border overflow-auto"><table className="w-full text-sm">
          <thead><tr className="border-b bg-muted/50">
            <th className="p-3 text-left font-medium">Algorithm</th>
            <th className="p-3 text-left font-medium">Problem Type</th>
            <th className="p-3 text-left font-medium">Target</th>
            <th className="p-3 text-left font-medium">Eval Mode</th>
            <th className="p-3 text-right font-medium">Score</th>
            <th className="p-3 text-right font-medium">Date</th>
          </tr></thead>
          <tbody>{models.slice(-10).reverse().map((model, idx) => {
            const score = model.problemType === 'classification' ? (model.metrics?.accuracy || 0) : (model.metrics?.r2 || 0);
            return (<tr key={model.modelId} className="border-b last:border-0 hover:bg-accent/50 transition-colors" data-testid={`recent-model-row-${idx}`}>
              <td className="p-3"><div className="flex items-center gap-2"><div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center shrink-0"><Brain className="h-3.5 w-3.5 text-primary" /></div><span className="font-medium">{ALGO_NAMES[model.algorithm] || model.algorithm}</span></div></td>
              <td className="p-3"><Badge variant="outline">{model.problemType || '—'}</Badge></td>
              <td className="p-3 text-xs font-mono">{model.targetColumn || '—'}</td>
              <td className="p-3"><Badge variant="secondary">{model.evalMode === 'cv' ? '5-Fold CV' : 'Train/Test'}</Badge></td>
              <td className="p-3 text-right font-mono font-semibold">{(score * 100).toFixed(1)}%</td>
              <td className="p-3 text-right text-muted-foreground text-xs">{model.createdAt ? new Date(model.createdAt).toLocaleDateString() : '—'}</td>
            </tr>);
          })}</tbody>
        </table></div>
      </CardContent></Card></motion.div>
    </>)}

    {/* ==================== LEADERBOARD WIDGET ==================== */}
    {topLeaderboard.length > 0 && (
      <motion.div variants={fadeInUp}>
        <Card data-testid="dashboard-leaderboard-widget">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2 text-base"><Trophy className="h-4 w-4 text-amber-500" />All-Time Leaderboard</CardTitle>
                <CardDescription className="text-xs">{leaderboardEntries.length} models across sessions</CardDescription>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setActiveView('leaderboard')} className="text-xs" data-testid="view-full-leaderboard">
                View All <ArrowUpRight className="h-3 w-3 ml-1" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="space-y-2">
              {topLeaderboard.map((entry, i) => {
                const score = entry.problem_type === 'classification' ? (entry.metrics?.f1 || entry.metrics?.accuracy || 0) : (entry.metrics?.r2 || 0);
                return (
                  <div key={entry.model_id + i} className={`flex items-center gap-3 p-2 rounded-lg ${i === 0 ? 'bg-amber-50/80 dark:bg-amber-950/20' : 'bg-muted/30'}`} data-testid={`lb-widget-${i}`}>
                    <span className={`text-sm font-bold w-6 text-center ${i === 0 ? 'text-amber-500' : i === 1 ? 'text-slate-400' : i === 2 ? 'text-amber-700' : 'text-muted-foreground'}`}>
                      {i === 0 ? <Trophy className="h-4 w-4 mx-auto" /> : `#${i + 1}`}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{ALGO_NAMES[entry.algorithm] || entry.algorithm}</p>
                      <p className="text-xs text-muted-foreground truncate">{entry.dataset_name || 'Unknown'}</p>
                    </div>
                    <Badge variant={i === 0 ? 'default' : 'secondary'} className="text-xs shrink-0">
                      {(score * 100).toFixed(1)}%
                    </Badge>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )}
  </motion.div>
  );
}
