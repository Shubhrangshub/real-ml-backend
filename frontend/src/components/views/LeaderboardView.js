import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Trophy, Brain, TrendingUp, Clock, Trash2, Target, Activity,
  Zap, ChevronRight, ArrowUpRight, AlertCircle, BarChart3, Shield, Layers
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, AreaChart, Area
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, ALGO_COLORS } from '../../constants';
import { getScoreColor } from '../../utils/helpers';
import { useApp } from '../../context/AppContext';

const ALGO_CHART_COLORS = {
  linear_regression: '#3b82f6', ridge_regression: '#6366f1', logistic_regression: '#8b5cf6',
  decision_tree: '#10b981', random_forest: '#059669', gradient_boosting: '#f59e0b',
  knn: '#ec4899', svm: '#ef4444', naive_bayes: '#06b6d4', baseline: '#94a3b8',
  auto: '#a855f7',
};

export default function LeaderboardView() {
  const {
    leaderboardEntries, leaderboardLoading, fetchLeaderboard,
    deleteLeaderboardEntry, clearLeaderboard, setActiveView
  } = useApp();

  const [sortBy, setSortBy] = useState('score');
  const [sortDir, setSortDir] = useState('desc');
  const [filterAlgo, setFilterAlgo] = useState('all');
  const [filterType, setFilterType] = useState('all');

  // Unique algorithms and types
  const algorithms = useMemo(() => [...new Set(leaderboardEntries.map(e => e.algorithm))], [leaderboardEntries]);
  const problemTypes = useMemo(() => [...new Set(leaderboardEntries.map(e => e.problem_type))], [leaderboardEntries]);

  // Filtered & sorted entries
  const sortedEntries = useMemo(() => {
    let entries = [...leaderboardEntries];
    if (filterAlgo !== 'all') entries = entries.filter(e => e.algorithm === filterAlgo);
    if (filterType !== 'all') entries = entries.filter(e => e.problem_type === filterType);

    entries.sort((a, b) => {
      const getScore = (e) => {
        if (e.problem_type === 'classification') return e.metrics?.f1 || e.metrics?.accuracy || 0;
        return e.metrics?.r2 || 0;
      };
      let va, vb;
      if (sortBy === 'score') { va = getScore(a); vb = getScore(b); }
      else if (sortBy === 'time') { va = a.duration_sec || 0; vb = b.duration_sec || 0; }
      else if (sortBy === 'date') { va = new Date(a.created_at).getTime(); vb = new Date(b.created_at).getTime(); }
      else { va = 0; vb = 0; }
      return sortDir === 'desc' ? vb - va : va - vb;
    });
    return entries;
  }, [leaderboardEntries, filterAlgo, filterType, sortBy, sortDir]);

  // ==================== TIMELINE: Best score per session/day ====================
  const timelineData = useMemo(() => {
    if (leaderboardEntries.length === 0) return [];
    const byDate = {};
    leaderboardEntries.forEach(e => {
      const date = new Date(e.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      const score = e.problem_type === 'classification'
        ? (e.metrics?.f1 || e.metrics?.accuracy || 0)
        : (e.metrics?.r2 || 0);
      if (!byDate[date] || score > byDate[date].best) {
        byDate[date] = { best: score, algo: e.algorithm, count: (byDate[date]?.count || 0) + 1 };
      } else {
        byDate[date].count++;
      }
    });
    return Object.entries(byDate).map(([date, data]) => ({
      date,
      best: +(data.best * 100).toFixed(1),
      algo: ALGO_NAMES[data.algo] || data.algo,
      models: data.count,
    }));
  }, [leaderboardEntries]);

  // ==================== PER-ALGORITHM TREND ====================
  const algoTrendData = useMemo(() => {
    if (leaderboardEntries.length === 0) return [];
    // Group by algorithm, then track performance over runs
    const algoRuns = {};
    [...leaderboardEntries].reverse().forEach(e => {
      const algo = ALGO_NAMES[e.algorithm] || e.algorithm;
      if (!algoRuns[algo]) algoRuns[algo] = [];
      const score = e.problem_type === 'classification'
        ? (e.metrics?.f1 || e.metrics?.accuracy || 0) * 100
        : (e.metrics?.r2 || 0) * 100;
      algoRuns[algo].push(score);
    });
    // Build chart data: each point is a run index
    const maxRuns = Math.max(...Object.values(algoRuns).map(r => r.length));
    const data = [];
    for (let i = 0; i < maxRuns; i++) {
      const point = { run: `Run ${i + 1}` };
      for (const [algo, runs] of Object.entries(algoRuns)) {
        if (i < runs.length) point[algo] = +runs[i].toFixed(1);
      }
      data.push(point);
    }
    return data;
  }, [leaderboardEntries]);

  const algoNames = useMemo(() => {
    const names = new Set();
    leaderboardEntries.forEach(e => names.add(ALGO_NAMES[e.algorithm] || e.algorithm));
    return [...names];
  }, [leaderboardEntries]);

  // ==================== STATS ====================
  const stats = useMemo(() => {
    if (leaderboardEntries.length === 0) return null;
    const scores = leaderboardEntries.map(e =>
      e.problem_type === 'classification' ? (e.metrics?.f1 || e.metrics?.accuracy || 0) : (e.metrics?.r2 || 0)
    );
    const best = Math.max(...scores);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const totalTime = leaderboardEntries.reduce((s, e) => s + (e.duration_sec || 0), 0);
    const datasets = new Set(leaderboardEntries.map(e => e.dataset_name)).size;
    return { total: leaderboardEntries.length, best, avg, totalTime, datasets, algorithms: algorithms.length };
  }, [leaderboardEntries, algorithms]);

  // Rank badge helper
  const getRankBadge = (rank) => {
    if (rank === 1) return <span className="inline-flex items-center gap-1 text-amber-500 font-bold"><Trophy className="h-4 w-4" />1st</span>;
    if (rank === 2) return <span className="inline-flex items-center gap-1 text-slate-400 font-semibold">2nd</span>;
    if (rank === 3) return <span className="inline-flex items-center gap-1 text-amber-700 font-semibold">3rd</span>;
    return <span className="text-muted-foreground">{rank}th</span>;
  };

  // Empty state
  if (leaderboardEntries.length === 0 && !leaderboardLoading) {
    return (
      <motion.div key="leaderboard" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="leaderboard-view">
        <Card className="border-2 border-dashed">
          <CardContent className="py-20 text-center">
            <Trophy className="h-16 w-16 text-muted-foreground/40 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Leaderboard Entries Yet</h3>
            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
              Train models and they'll automatically appear here. Track your progress across sessions and find the winning algorithm.
            </p>
            <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="go-train-btn">
              <Zap className="h-4 w-4 mr-2" />Train Your First Model
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div key="leaderboard" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="leaderboard-view">

      {/* ==================== STATS ROW ==================== */}
      {stats && (
        <motion.div variants={fadeInUp}>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            <Card data-testid="stat-total-models">
              <CardContent className="p-4 text-center">
                <Brain className="h-6 w-6 text-primary mx-auto mb-1" />
                <p className="text-2xl font-bold">{stats.total}</p>
                <p className="text-xs text-muted-foreground">Total Models</p>
              </CardContent>
            </Card>
            <Card data-testid="stat-best-score">
              <CardContent className="p-4 text-center">
                <Trophy className="h-6 w-6 text-amber-500 mx-auto mb-1" />
                <p className="text-2xl font-bold text-amber-500">{(stats.best * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Best Score</p>
              </CardContent>
            </Card>
            <Card data-testid="stat-avg-score">
              <CardContent className="p-4 text-center">
                <TrendingUp className="h-6 w-6 text-emerald-500 mx-auto mb-1" />
                <p className="text-2xl font-bold text-emerald-500">{(stats.avg * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Average Score</p>
              </CardContent>
            </Card>
            <Card data-testid="stat-algos-tried">
              <CardContent className="p-4 text-center">
                <Layers className="h-6 w-6 text-violet-500 mx-auto mb-1" />
                <p className="text-2xl font-bold text-violet-500">{stats.algorithms}</p>
                <p className="text-xs text-muted-foreground">Algorithms Tried</p>
              </CardContent>
            </Card>
            <Card data-testid="stat-total-time">
              <CardContent className="p-4 text-center">
                <Clock className="h-6 w-6 text-blue-500 mx-auto mb-1" />
                <p className="text-2xl font-bold text-blue-500">{stats.totalTime.toFixed(1)}s</p>
                <p className="text-xs text-muted-foreground">Total Training</p>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      )}

      {/* ==================== TIMELINE: BEST SCORE PER SESSION ==================== */}
      {timelineData.length > 1 && (
        <motion.div variants={fadeInUp}>
          <Card data-testid="timeline-chart">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" />Best Score Over Time</CardTitle>
              <CardDescription>Your top model performance per session</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tickFormatter={v => v + '%'} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                    formatter={(value, name) => [value + '%', 'Best Score']}
                    labelFormatter={(label) => label}
                  />
                  <Area type="monotone" dataKey="best" stroke="hsl(var(--primary))" fill="url(#scoreGrad)" strokeWidth={2.5} dot={{ r: 5, fill: 'hsl(var(--primary))' }} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ==================== PER-ALGORITHM TREND ==================== */}
      {algoTrendData.length > 1 && algoNames.length > 1 && (
        <motion.div variants={fadeInUp}>
          <Card data-testid="algo-trend-chart">
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5" />Algorithm Performance Trend</CardTitle>
              <CardDescription>How each algorithm performs across training runs</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={algoTrendData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="run" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tickFormatter={v => v + '%'} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                    formatter={(value) => [value + '%']}
                  />
                  <Legend />
                  {algoNames.map((name, i) => {
                    const rawAlgo = Object.entries(ALGO_NAMES).find(([k, v]) => v === name)?.[0] || name;
                    const color = ALGO_CHART_COLORS[rawAlgo] || ALGO_CHART_COLORS[Object.keys(ALGO_CHART_COLORS)[i % Object.keys(ALGO_CHART_COLORS).length]];
                    return (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={name}
                        stroke={color}
                        strokeWidth={2}
                        dot={{ r: 4, fill: color }}
                        connectNulls
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ==================== RANKED TABLE ==================== */}
      <motion.div variants={fadeInUp}>
        <Card data-testid="leaderboard-table-card">
          <CardHeader>
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <CardTitle className="flex items-center gap-2"><Trophy className="h-5 w-5" />Model Rankings</CardTitle>
                <CardDescription>{sortedEntries.length} models ranked by performance</CardDescription>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <select
                  value={filterAlgo}
                  onChange={(e) => setFilterAlgo(e.target.value)}
                  className="rounded-md border border-input bg-background px-2.5 py-1.5 text-xs"
                  data-testid="filter-algo"
                >
                  <option value="all">All Algorithms</option>
                  {algorithms.map(a => <option key={a} value={a}>{ALGO_NAMES[a] || a}</option>)}
                </select>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value)}
                  className="rounded-md border border-input bg-background px-2.5 py-1.5 text-xs"
                  data-testid="filter-type"
                >
                  <option value="all">All Types</option>
                  {problemTypes.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
                <select
                  value={`${sortBy}-${sortDir}`}
                  onChange={(e) => { const [s, d] = e.target.value.split('-'); setSortBy(s); setSortDir(d); }}
                  className="rounded-md border border-input bg-background px-2.5 py-1.5 text-xs"
                  data-testid="sort-select"
                >
                  <option value="score-desc">Best Score First</option>
                  <option value="score-asc">Worst Score First</option>
                  <option value="date-desc">Newest First</option>
                  <option value="date-asc">Oldest First</option>
                  <option value="time-asc">Fastest Training</option>
                  <option value="time-desc">Slowest Training</option>
                </select>
                {leaderboardEntries.length > 0 && (
                  <Button variant="outline" size="sm" onClick={clearLeaderboard} className="text-destructive hover:text-destructive" data-testid="clear-leaderboard-btn">
                    <Trash2 className="h-3.5 w-3.5 mr-1" />Clear All
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg border overflow-auto">
              <table className="w-full text-sm" data-testid="leaderboard-table">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="p-3 text-left font-medium w-16">Rank</th>
                    <th className="p-3 text-left font-medium">Algorithm</th>
                    <th className="p-3 text-left font-medium">Dataset</th>
                    <th className="p-3 text-center font-medium">Type</th>
                    <th className="p-3 text-center font-medium">Score</th>
                    <th className="p-3 text-center font-medium">Accuracy/R²</th>
                    <th className="p-3 text-center font-medium">Time</th>
                    <th className="p-3 text-center font-medium">Date</th>
                    <th className="p-3 text-center font-medium w-12"></th>
                  </tr>
                </thead>
                <tbody>
                  {sortedEntries.map((entry, idx) => {
                    const rank = idx + 1;
                    const score = entry.problem_type === 'classification'
                      ? (entry.metrics?.f1 || entry.metrics?.accuracy || 0)
                      : (entry.metrics?.r2 || 0);
                    const primaryMetric = entry.problem_type === 'classification'
                      ? entry.metrics?.accuracy
                      : entry.metrics?.r2;
                    return (
                      <motion.tr
                        key={entry.model_id + idx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.03 }}
                        className={`border-b last:border-0 hover:bg-accent/50 transition-colors ${rank === 1 ? 'bg-amber-50/50 dark:bg-amber-950/10' : ''}`}
                        data-testid={`lb-row-${idx}`}
                      >
                        <td className="p-3">{getRankBadge(rank)}</td>
                        <td className="p-3">
                          <div className="flex items-center gap-2">
                            <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: ALGO_CHART_COLORS[entry.algorithm] || '#94a3b8' }} />
                            <span className="font-medium">{ALGO_NAMES[entry.algorithm] || entry.algorithm}</span>
                          </div>
                        </td>
                        <td className="p-3 text-xs text-muted-foreground max-w-[150px] truncate">
                          {entry.dataset_name || '—'}
                        </td>
                        <td className="p-3 text-center">
                          <Badge variant="outline" className="text-xs">{entry.problem_type}</Badge>
                        </td>
                        <td className="p-3 text-center">
                          <span className={`font-bold text-sm ${getScoreColor(score)}`}>
                            {(score * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="p-3 text-center font-mono text-xs">
                          {primaryMetric !== undefined ? (primaryMetric * 100).toFixed(2) + '%' : '—'}
                        </td>
                        <td className="p-3 text-center text-xs text-muted-foreground">
                          {entry.duration_sec ? entry.duration_sec.toFixed(2) + 's' : '—'}
                        </td>
                        <td className="p-3 text-center text-xs text-muted-foreground">
                          {new Date(entry.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                        </td>
                        <td className="p-3 text-center">
                          <Button variant="ghost" size="sm" className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                            onClick={() => deleteLeaderboardEntry(entry.model_id)} data-testid={`lb-delete-${idx}`}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ==================== ALGORITHM DISTRIBUTION ==================== */}
      <motion.div variants={fadeInUp}>
        <Card data-testid="algo-distribution-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Algorithm Distribution</CardTitle>
            <CardDescription>Number of models trained per algorithm and their average scores</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={(() => {
                const groups = {};
                leaderboardEntries.forEach(e => {
                  const name = ALGO_NAMES[e.algorithm] || e.algorithm;
                  if (!groups[name]) groups[name] = { name, count: 0, totalScore: 0, algo: e.algorithm };
                  groups[name].count++;
                  const score = e.problem_type === 'classification' ? (e.metrics?.f1 || e.metrics?.accuracy || 0) : (e.metrics?.r2 || 0);
                  groups[name].totalScore += score;
                });
                return Object.values(groups).map(g => ({
                  name: g.name,
                  count: g.count,
                  avgScore: +(g.totalScore / g.count * 100).toFixed(1),
                  algo: g.algo,
                })).sort((a, b) => b.avgScore - a.avgScore);
              })()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis yAxisId="left" label={{ value: 'Avg Score %', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Count', angle: 90, position: 'insideRight', style: { fontSize: 11 } }} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="avgScore" name="Avg Score %" radius={[4, 4, 0, 0]}>
                  {(() => {
                    const groups = {};
                    leaderboardEntries.forEach(e => { const name = ALGO_NAMES[e.algorithm] || e.algorithm; groups[name] = e.algorithm; });
                    return Object.entries(groups).map(([name, algo]) => (
                      <Cell key={name} fill={ALGO_CHART_COLORS[algo] || '#94a3b8'} />
                    ));
                  })()}
                </Bar>
                <Bar yAxisId="right" dataKey="count" name="Models Trained" fill="hsl(var(--muted-foreground))" opacity={0.3} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
