import React from 'react';
import { motion } from 'framer-motion';
import {
  Layers, ShieldAlert, BarChart3, Brain, Upload, Download, Trash2, Eye,
  Database, Zap
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';
import { staggerContainer, fadeInUp, ALGO_NAMES, CLUSTER_COLORS } from '../../constants';
import { useApp } from '../../context/AppContext';

export function ClustersView() {
  const {
    dataProfile, numClusters, setNumClusters, clusterResult,
    handleClustering, DataUploadMini
  } = useApp();

  return (
    <motion.div key="clusters" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="clusters-view">
      {!dataProfile ? <DataUploadMini /> : (<>
        <motion.div variants={fadeInUp}><Card data-testid="cluster-config-card"><CardHeader><CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5" />Configuration</CardTitle><CardDescription>{dataProfile.numericColumns.length} numeric features, {dataProfile.rowCount} rows</CardDescription></CardHeader>
          <CardContent><div className="flex items-center gap-6 flex-wrap">
            <div className="space-y-2 flex-1 min-w-[200px]"><label className="text-sm font-medium">Clusters (k): <span className="text-primary font-bold">{numClusters}</span></label><input type="range" min={2} max={Math.min(10, dataProfile.rowCount - 1)} value={numClusters} onChange={(e) => setNumClusters(Number(e.target.value))} className="w-full accent-primary" data-testid="cluster-k-slider" /></div>
            <Button onClick={handleClustering} size="lg" className="h-12" data-testid="run-clustering-btn"><Layers className="h-4 w-4 mr-2" />Run K-Means</Button>
          </div></CardContent></Card></motion.div>
        {clusterResult && (<>
          <motion.div variants={fadeInUp}><div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5">{clusterResult.clusterStats.map((cs) => <Card key={cs.clusterId} data-testid={`cluster-stat-${cs.clusterId}`}><CardContent className="p-4 text-center"><div className="h-4 w-full rounded-full mb-3" style={{ backgroundColor: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length], opacity: 0.3 }} /><p className="text-sm text-muted-foreground">Cluster {cs.clusterId}</p><p className="text-3xl font-bold" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>{cs.size}</p><p className="text-xs text-muted-foreground">points</p></CardContent></Card>)}</div></motion.div>
          <motion.div variants={fadeInUp}><Card data-testid="cluster-scatter-chart"><CardHeader><CardTitle>Visualization</CardTitle><CardDescription>{clusterResult.xFeature} vs {clusterResult.yFeature}</CardDescription></CardHeader><CardContent><ResponsiveContainer width="100%" height={400}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" name={clusterResult.xFeature} type="number" /><YAxis dataKey="y" name={clusterResult.yFeature} type="number" /><ZAxis range={[60, 60]} /><Tooltip /><Legend />{Array.from({ length: clusterResult.k }, (_, i) => <Scatter key={i} name={`Cluster ${i}`} data={clusterResult.points.filter(p => p.cluster === i)} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />)}</ScatterChart></ResponsiveContainer></CardContent></Card></motion.div>
          <motion.div variants={fadeInUp}><Card><CardHeader><CardTitle>Size Distribution</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={250}><BarChart data={clusterResult.clusterStats.map(cs => ({ name: `Cluster ${cs.clusterId}`, size: cs.size }))}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="size" radius={[4, 4, 0, 0]}>{clusterResult.clusterStats.map((_, i) => <Cell key={i} fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]} />)}</Bar></BarChart></ResponsiveContainer></CardContent></Card></motion.div>
          <motion.div variants={fadeInUp}><Card data-testid="cluster-means-table"><CardHeader><CardTitle>Cluster Centers</CardTitle></CardHeader><CardContent><div className="rounded-md border overflow-auto"><table className="w-full text-sm"><thead><tr className="border-b bg-muted/50"><th className="p-3 text-left font-medium">Feature</th>{clusterResult.clusterStats.map(cs => <th key={cs.clusterId} className="p-3 text-center font-medium" style={{ color: CLUSTER_COLORS[cs.clusterId % CLUSTER_COLORS.length] }}>C{cs.clusterId}</th>)}</tr></thead><tbody>{clusterResult.features.map((feat, fi) => <tr key={fi} className="border-b last:border-0"><td className="p-3 font-mono text-xs">{feat}</td>{clusterResult.clusterStats.map(cs => <td key={cs.clusterId} className="p-3 text-center text-xs">{cs.means[fi]?.mean?.toFixed(2)}</td>)}</tr>)}</tbody></table></div></CardContent></Card></motion.div>
        </>)}
      </>)}
    </motion.div>
  );
}

export function AnomaliesView() {
  const {
    dataProfile, anomalyMethod, setAnomalyMethod, anomalyThreshold, setAnomalyThreshold,
    anomalyResult, handleAnomalyDetection, DataUploadMini
  } = useApp();

  return (
    <motion.div key="anomalies" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="anomalies-view">
      {!dataProfile ? <DataUploadMini /> : (<>
        <motion.div variants={fadeInUp}><Card data-testid="anomaly-config-card"><CardHeader><CardTitle className="flex items-center gap-2"><ShieldAlert className="h-5 w-5" />Configuration</CardTitle></CardHeader>
          <CardContent><div className="flex items-center gap-6 flex-wrap">
            <div className="space-y-2"><label className="text-sm font-medium">Method</label><select value={anomalyMethod} onChange={(e) => setAnomalyMethod(e.target.value)} className="rounded-md border border-input bg-background px-3 py-2 text-sm" data-testid="anomaly-method-select"><option value="zscore">Z-Score</option><option value="iqr">IQR</option></select></div>
            {anomalyMethod === 'zscore' && <div className="space-y-2"><label className="text-sm font-medium">Threshold: <span className="text-primary font-bold">{anomalyThreshold}</span></label><input type="range" min={1.5} max={4} step={0.5} value={anomalyThreshold} onChange={(e) => setAnomalyThreshold(Number(e.target.value))} className="w-40 accent-primary" data-testid="anomaly-threshold-slider" /></div>}
            <Button onClick={handleAnomalyDetection} size="lg" className="h-12" data-testid="run-anomaly-btn"><ShieldAlert className="h-4 w-4 mr-2" />Detect Anomalies</Button>
          </div></CardContent></Card></motion.div>
        {anomalyResult && (<>
          <motion.div variants={fadeInUp}><div className="grid gap-4 md:grid-cols-3"><Card data-testid="anomaly-count-card"><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Anomalies</p><p className="text-4xl font-bold text-destructive">{anomalyResult.totalAnomalies}</p></CardContent></Card><Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Normal</p><p className="text-4xl font-bold text-primary">{anomalyResult.totalRows - anomalyResult.totalAnomalies}</p></CardContent></Card><Card><CardContent className="p-6 text-center"><p className="text-sm text-muted-foreground">Rate</p><p className="text-4xl font-bold">{(anomalyResult.totalAnomalies / anomalyResult.totalRows * 100).toFixed(1)}%</p></CardContent></Card></div></motion.div>
          {anomalyResult.xFeature && <motion.div variants={fadeInUp}><Card data-testid="anomaly-scatter-chart"><CardHeader><CardTitle>Normal vs Anomaly</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={400}><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" /><YAxis dataKey="y" type="number" /><ZAxis range={[60, 60]} /><Tooltip /><Legend /><Scatter name="Normal" data={anomalyResult.normalPoints} fill="hsl(var(--primary))" /><Scatter name="Anomaly" data={anomalyResult.anomalyPoints} fill="hsl(var(--destructive))" /></ScatterChart></ResponsiveContainer></CardContent></Card></motion.div>}
          <motion.div variants={fadeInUp}><Card data-testid="anomaly-per-column"><CardHeader><CardTitle>Per Column</CardTitle></CardHeader><CardContent><ResponsiveContainer width="100%" height={250}><BarChart data={Object.entries(anomalyResult.anomalies).map(([col, items]) => ({ name: col, count: items.length })).filter(d => d.count > 0)}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" angle={-30} textAnchor="end" height={80} /><YAxis /><Tooltip /><Bar dataKey="count" fill="hsl(var(--destructive))" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></CardContent></Card></motion.div>
          {anomalyResult.anomalyRowIndices.length > 0 && <motion.div variants={fadeInUp}><Card data-testid="anomaly-rows-table"><CardHeader><CardTitle>Anomalous Rows</CardTitle></CardHeader><CardContent><div className="rounded-md border overflow-auto max-h-80"><table className="w-full text-sm"><thead><tr className="border-b bg-destructive/10 sticky top-0"><th className="p-2 text-left font-medium">Row</th>{dataProfile.numericColumns.slice(0, 6).map(col => <th key={col} className="p-2 text-left font-medium">{col}</th>)}</tr></thead><tbody>{anomalyResult.anomalyRowIndices.slice(0, 20).map(ri => <tr key={ri} className="border-b last:border-0 bg-destructive/5"><td className="p-2 font-mono text-xs font-bold">{ri + 1}</td>{dataProfile.numericColumns.slice(0, 6).map(col => { const isA = anomalyResult.anomalies[col]?.some(a => a.index === ri); return <td key={col} className={`p-2 text-xs ${isA ? 'text-destructive font-bold' : ''}`}>{typeof dataProfile.rows[ri]?.[col] === 'number' ? dataProfile.rows[ri][col].toFixed(2) : '-'}</td>; })}</tr>)}</tbody></table></div></CardContent></Card></motion.div>}
        </>)}
      </>)}
    </motion.div>
  );
}

export function ModelsView() {
  const {
    models, setActiveView, handleDeleteModel, handleDownloadModel, handleImportModel
  } = useApp();

  return (
    <motion.div key="models" variants={fadeInUp} initial="initial" animate="animate" exit="exit" data-testid="models-view">
      <Card><CardHeader><div className="flex items-center justify-between"><div><CardTitle className="flex items-center gap-2"><BarChart3 className="h-5 w-5" />Model Library</CardTitle></div><div className="flex items-center gap-2">
        <div className="relative"><input type="file" accept=".json" onChange={handleImportModel} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" data-testid="import-model-input" /><Button variant="outline" size="sm" data-testid="import-model-btn"><Upload className="h-4 w-4 mr-2" />Import Model</Button></div>
        <Badge variant="secondary" className="text-lg px-4 py-2" data-testid="models-count-badge">{models.length} Models</Badge>
      </div></div></CardHeader>
        <CardContent>{models.length === 0 ? <div className="text-center py-12" data-testid="empty-models"><Database className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" /><h3 className="text-lg font-medium mb-2">No Models Yet</h3><Button onClick={() => setActiveView('analysis')} size="lg"><Zap className="h-4 w-4 mr-2" />Train Your First Model</Button></div>
        : <div className="rounded-md border"><table className="w-full" data-testid="models-table"><thead><tr className="border-b bg-muted/50"><th className="p-4 text-left text-sm font-medium">Model ID</th><th className="p-4 text-left text-sm font-medium">Algorithm</th><th className="p-4 text-left text-sm font-medium">Type</th><th className="p-4 text-left text-sm font-medium">Created</th><th className="p-4 text-left text-sm font-medium">Actions</th></tr></thead>
          <tbody>{models.map((model, idx) => <motion.tr key={model.modelId} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: idx * 0.05 }} className="border-b last:border-0 hover:bg-accent/50 transition-colors" data-testid={`model-row-${idx}`}><td className="p-4"><div className="flex items-center gap-2"><div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center"><Brain className="h-4 w-4 text-primary" /></div><code className="text-xs font-mono">{model.modelId.substring(0, 12)}...</code></div></td><td className="p-4"><Badge variant="outline">{ALGO_NAMES[model.algorithm] || model.algorithm}</Badge></td><td className="p-4 text-sm">{model.problemType}</td><td className="p-4 text-sm text-muted-foreground">{new Date(model.createdAt).toLocaleDateString()}</td><td className="p-4"><div className="flex gap-2"><Button variant="ghost" size="sm" onClick={() => setActiveView('predict')} data-testid={`use-model-${idx}`}><Eye className="h-4 w-4" /></Button><Button variant="ghost" size="sm" onClick={() => handleDownloadModel(model.modelId)} className="text-primary" data-testid={`download-model-${idx}`}><Download className="h-4 w-4" /></Button><Button variant="ghost" size="sm" onClick={() => handleDeleteModel(model.modelId)} data-testid={`delete-model-${idx}`}><Trash2 className="h-4 w-4 text-destructive" /></Button></div></td></motion.tr>)}</tbody></table></div>
        }</CardContent></Card>
    </motion.div>
  );
}
