import React from 'react';
import { motion } from 'framer-motion';
import { BarChart2, Zap, TrendingUp, BarChart3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { staggerContainer, fadeInUp } from '../../constants';
import { arrayMin, arrayMax } from '../../utils/helpers';
import { HelpTip } from '../SmartTooltip';
import { useApp } from '../../context/AppContext';

export default function DataExplorerView() {
  const {
    dataProfile, setActiveView, histogramCol, setHistogramCol, histogramData,
    corrVarX, setCorrVarX, corrVarY, setCorrVarY, corrMatrix, dataPreview
  } = useApp();

  return (
  <motion.div key="explore" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="explore-view">
    {!dataProfile ? (
      <Card className="border-2 border-dashed"><CardContent className="py-16 text-center">
        <BarChart2 className="h-14 w-14 text-muted-foreground/30 mx-auto mb-5" />
        <h3 className="text-lg font-semibold mb-2">No Data Loaded</h3>
        <p className="text-muted-foreground text-sm mb-4">Please upload a dataset in the Analysis tab first, then come back to explore your data visually.</p>
        <Button onClick={() => setActiveView('analysis')} size="lg" data-testid="explore-go-analysis"><Zap className="h-4 w-4 mr-2" />Go to Analysis</Button>
      </CardContent></Card>
    ) : (<>
      {/* Histogram */}
      <motion.div variants={fadeInUp}>
      <Card data-testid="histogram-card"><CardHeader>
        <CardTitle className="flex items-center gap-2"><BarChart2 className="h-5 w-5" /><HelpTip text="Histograms show the distribution of values in a numeric column. Each bar represents a range, and the bar height shows how many data points fall in that range. Look for skewed distributions, outliers, or multimodal patterns.">Feature Histograms</HelpTip></CardTitle>
        <CardDescription>Select a numeric feature to see its distribution.</CardDescription>
      </CardHeader><CardContent className="space-y-4">
        <select value={histogramCol} onChange={e => setHistogramCol(e.target.value)} className="w-full max-w-xs rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="histogram-col-select">
          <option value="">-- Select Column --</option>
          {dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        {histogramData.length > 0 && (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="bin" angle={-30} textAnchor="end" height={60} tick={{fontSize: 10}} />
              <YAxis tick={{fontSize: 11}} />
              <Tooltip formatter={(v) => [`${v} rows`, 'Count']} />
              <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
        {histogramCol && histogramData.length > 0 && (() => {
          const vals = dataProfile.rows.map(r => r[histogramCol]).filter(v => typeof v === 'number');
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const sorted = [...vals].sort((a, b) => a - b);
          const median = sorted.length % 2 === 0 ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 : sorted[Math.floor(sorted.length / 2)];
          const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
          return (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Count</p><p className="text-lg font-bold">{vals.length}</p></div>
              <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Mean</p><p className="text-lg font-bold">{mean.toFixed(2)}</p></div>
              <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Median</p><p className="text-lg font-bold">{median.toFixed(2)}</p></div>
              <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Std Dev</p><p className="text-lg font-bold">{std.toFixed(2)}</p></div>
              <div className="rounded-lg border p-3 bg-muted/30"><p className="text-xs text-muted-foreground">Range</p><p className="text-lg font-bold">{arrayMin(vals).toFixed(1)} – {arrayMax(vals).toFixed(1)}</p></div>
            </div>
          );
        })()}
      </CardContent></Card>
      </motion.div>

      {/* Correlation Heatmap */}
      {corrMatrix.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="explore-correlation-heatmap"><CardHeader><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Correlation Heatmap</CardTitle><CardDescription>Pairwise correlations between all numeric variables. Blue = positive, red = negative.</CardDescription></CardHeader>
        <CardContent><div className="overflow-auto">
          <div className="inline-grid gap-px" style={{gridTemplateColumns: `120px repeat(${dataProfile.numericColumns.length}, minmax(60px, 1fr))`}}>
            <div />
            {dataProfile.numericColumns.map(col => <div key={col} className="text-xs font-mono p-2 text-center truncate" title={col}>{col}</div>)}
            {corrMatrix.map((row, ri) => (<React.Fragment key={ri}>
              <div className="text-xs font-mono p-2 text-right truncate" title={row.feature}>{row.feature}</div>
              {dataProfile.numericColumns.map((col, ci) => { const val = row[col] || 0; const bg = val > 0 ? `rgba(37, 99, 235, ${Math.abs(val) * 0.8})` : `rgba(220, 38, 38, ${Math.abs(val) * 0.8})`; return <div key={ci} className="p-2 text-center text-xs font-mono rounded-sm border border-muted/30" style={{backgroundColor: bg, color: Math.abs(val) > 0.4 ? 'white' : 'inherit'}} title={`${row.feature} vs ${col}: ${val.toFixed(3)}`}>{val.toFixed(2)}</div>; })}
            </React.Fragment>))}
          </div>
        </div></CardContent></Card></motion.div>}

      {/* Scatter Plot */}
      <motion.div variants={fadeInUp}><Card data-testid="explore-scatter"><CardHeader><CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" />Scatter Plot</CardTitle><CardDescription>Select two numeric variables to visualize their relationship.</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5"><label className="text-sm font-medium">X Variable</label><select value={corrVarX} onChange={e => setCorrVarX(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="explore-var-x"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div className="space-y-1.5"><label className="text-sm font-medium">Y Variable</label><select value={corrVarY} onChange={e => setCorrVarY(e.target.value)} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="explore-var-y"><option value="">-- Select --</option>{dataProfile.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
          </div>
          {corrVarX && corrVarY && (() => {
            const pairs = dataProfile.rows.filter(r => typeof r[corrVarX] === 'number' && typeof r[corrVarY] === 'number').map(r => ({ x: r[corrVarX], y: r[corrVarY] }));
            const n = pairs.length; if (n < 3) return <p className="text-sm text-muted-foreground">Insufficient data for these variables.</p>;
            const mx = pairs.reduce((s, p) => s + p.x, 0) / n, my = pairs.reduce((s, p) => s + p.y, 0) / n;
            let sx = 0, sy = 0, sxy = 0; pairs.forEach(p => { sx += (p.x - mx) ** 2; sy += (p.y - my) ** 2; sxy += (p.x - mx) * (p.y - my); });
            const r = (sx > 0 && sy > 0) ? sxy / Math.sqrt(sx * sy) : 0;
            const abs = Math.abs(r); const dir = r >= 0 ? 'Positive' : 'Negative';
            const str = abs >= 0.8 ? 'Strong' : abs >= 0.5 ? 'Moderate' : abs >= 0.3 ? 'Weak' : 'Very Weak';
            return (<>
              <div className={`p-4 rounded-lg border-2 ${abs >= 0.5 ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20' : abs >= 0.3 ? 'border-amber-300 bg-amber-50 dark:bg-amber-950/20' : 'border-muted bg-muted/30'}`} data-testid="explore-corr-result">
                <div className="flex items-center justify-between"><div><p className="text-sm font-medium">Correlation Coefficient</p><p className="text-3xl font-bold">{r.toFixed(4)}</p></div><Badge variant={abs >= 0.5 ? 'default' : 'secondary'}>{str} {dir}</Badge></div>
              </div>
              <ResponsiveContainer width="100%" height={350}><ScatterChart><CartesianGrid strokeDasharray="3 3" opacity={0.3} /><XAxis dataKey="x" name={corrVarX} type="number" tick={{fontSize: 11}} label={{value: corrVarX, position: 'bottom', fontSize: 11}} /><YAxis dataKey="y" name={corrVarY} type="number" tick={{fontSize: 11}} label={{value: corrVarY, angle: -90, position: 'left', fontSize: 11}} /><ZAxis range={[60, 60]} /><Tooltip /><Scatter name="Data" data={pairs} fill="hsl(var(--primary))" /></ScatterChart></ResponsiveContainer>
            </>);
          })()}
        </CardContent></Card></motion.div>
    </>)}
  </motion.div>
  );
}
