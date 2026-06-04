import React from 'react';
import { motion } from 'framer-motion';
import { History, Play, Share2, Trash2, Save, ArrowUpRight, Clock } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { staggerContainer, fadeInUp, ALGO_NAMES } from '../../constants';
import { useApp } from '../../context/AppContext';

export default function HistoryView() {
  const {
    trainingResult, unsupervisedResult, viewOnlyMode, historyList, historyLoading,
    setShareUrl, setShareCopyStatus, handleSaveAnalysis, handleLoadSnapshot,
    handleDeleteSnapshot, fetchHistory, safeCopyToClipboard
  } = useApp();

  return (
  <motion.div key="history" variants={staggerContainer} initial="initial" animate="animate" exit="exit" className="space-y-6" data-testid="history-view">
    <motion.div variants={fadeInUp}><Card><CardHeader>
      <div className="flex items-center justify-between">
        <div><CardTitle className="flex items-center gap-2"><History className="h-5 w-5" />Saved Analyses</CardTitle>
          <CardDescription>Your analysis sessions are saved here. Click to restore, share, or delete.</CardDescription></div>
        <div className="flex items-center gap-2">
          {(trainingResult || unsupervisedResult) && !viewOnlyMode && <Button size="sm" onClick={() => handleSaveAnalysis()} data-testid="save-current-analysis-btn"><Save className="h-4 w-4 mr-2" />Save Current Analysis</Button>}
          <Button variant="outline" size="sm" onClick={fetchHistory} data-testid="refresh-history-btn"><ArrowUpRight className="h-4 w-4 mr-2" />Refresh</Button>
        </div>
      </div>
    </CardHeader><CardContent>
      {historyLoading ? (
        <div className="py-12 text-center"><div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent mx-auto mb-3" /><p className="text-sm text-muted-foreground">Loading history...</p></div>
      ) : historyList.length === 0 ? (
        <div className="py-12 text-center" data-testid="history-empty">
          <History className="h-14 w-14 text-muted-foreground/30 mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Saved Analyses</h3>
          <p className="text-muted-foreground text-sm mb-4">Train a model and click "Save" to store your analysis for later.</p>
        </div>
      ) : (
        <div className="space-y-3" data-testid="history-list">
          {historyList.map((snap, idx) => (
            <div key={snap.snapshot_id || idx} className="group rounded-lg border p-4 hover:bg-accent/30 transition-colors" data-testid={`history-item-${idx}`}>
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold text-sm truncate">{snap.name}</h4>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${snap.problem_type === 'classification' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' : snap.problem_type === 'regression' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-400'}`}>{snap.problem_type || 'unknown'}</span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{new Date(snap.created_at).toLocaleString()}</span>
                    <span>{snap.row_count} rows, {snap.col_count} cols</span>
                    {snap.target_column && <span>Target: <strong>{snap.target_column}</strong></span>}
                  </div>
                  {snap.models_summary?.length > 0 && <div className="flex flex-wrap gap-1.5 mt-2">
                    {snap.models_summary.slice(0, 5).map((m, mi) => (
                      <span key={mi} className="text-[10px] px-2 py-0.5 rounded-full bg-muted border">{ALGO_NAMES[m.algorithm] || m.algorithm}: {m.score?.toFixed(3)}</span>
                    ))}
                  </div>}
                </div>
                <div className="flex items-center gap-1.5 ml-4 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button variant="default" size="sm" onClick={() => handleLoadSnapshot(snap.snapshot_id)} data-testid={`load-snapshot-${idx}`}><Play className="h-3.5 w-3.5 mr-1.5" />Restore</Button>
                  <Button variant="outline" size="sm" onClick={async () => { const url = `${window.location.origin}${window.location.pathname}?snapshot=${snap.snapshot_id}`; setShareUrl(url); const ok = await safeCopyToClipboard(url); setShareCopyStatus(ok ? 'copied' : 'manual'); }} data-testid={`share-snapshot-${idx}`}><Share2 className="h-3.5 w-3.5" /></Button>
                  <Button variant="outline" size="sm" className="text-destructive hover:bg-destructive/10" onClick={() => handleDeleteSnapshot(snap.snapshot_id)} data-testid={`delete-snapshot-${idx}`}><Trash2 className="h-3.5 w-3.5" /></Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </CardContent></Card></motion.div>
  </motion.div>
  );
}
