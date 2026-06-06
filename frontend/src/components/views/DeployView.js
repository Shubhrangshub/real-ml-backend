import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Rocket, Globe, Copy, Trash2, Eye, EyeOff, Clock, Activity,
  Link2, Code, ExternalLink, ChevronDown, CheckCircle2, AlertCircle, Send, RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import { fadeInUp, ALGO_NAMES } from '../../constants';
import { useApp } from '../../context/AppContext';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';
function getToken() { return localStorage.getItem('automl_token') || ''; }

async function deployFetch(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { 'Authorization': `Bearer ${getToken()}`, 'Content-Type': 'application/json', ...options.headers },
  });
  if (!res.ok) { const err = await res.json().catch(() => ({ detail: 'Failed' })); throw new Error(err.detail); }
  return res.json();
}

export default function DeployView() {
  const { models, trainingResult } = useApp();
  const [deployments, setDeployments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deployingId, setDeployingId] = useState(null);
  const [showApi, setShowApi] = useState(null);

  const fetchDeployments = useCallback(async () => {
    try { const d = await deployFetch('/api/deploy'); setDeployments(d.deployments || []); }
    catch (e) { toast.error(e.message); }
    setLoading(false);
  }, []);

  useEffect(() => { fetchDeployments(); }, [fetchDeployments]);

  const handleDeploy = async (model) => {
    setDeployingId(model.modelId);
    try {
      const res = await deployFetch('/api/deploy', {
        method: 'POST',
        body: JSON.stringify({
          model_id: model.modelId,
          name: `${ALGO_NAMES[model.algorithm] || model.algorithm} — ${model.targetColumn || 'model'}`,
          description: `${model.problemType} model trained on ${model.modelData?.featureNames?.length || '?'} features`,
          model_data: model,
        }),
      });
      toast.success(`Model deployed! ID: ${res.deploy_id}`);
      fetchDeployments();
    } catch (e) { toast.error(e.message); }
    setDeployingId(null);
  };

  const handleToggle = async (dep) => {
    try {
      await deployFetch(`/api/deploy/${dep.deploy_id}`, {
        method: 'PATCH', body: JSON.stringify({ enabled: !dep.enabled }),
      });
      toast.success(dep.enabled ? 'Model disabled' : 'Model enabled');
      fetchDeployments();
    } catch (e) { toast.error(e.message); }
  };

  const handleDelete = async (dep) => {
    try {
      await deployFetch(`/api/deploy/${dep.deploy_id}`, { method: 'DELETE' });
      toast.success('Deployment removed');
      fetchDeployments();
    } catch (e) { toast.error(e.message); }
  };

  const copyUrl = (deployId) => {
    const url = `${window.location.origin}/predict/${deployId}`;
    navigator.clipboard.writeText(url).then(() => toast.success('Link copied!')).catch(() => toast.error('Copy failed'));
  };

  const copyApiExample = (dep) => {
    const example = `curl -X POST "${API_URL}/api/public/predict/${dep.deploy_id}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {}}'`;
    navigator.clipboard.writeText(example).then(() => toast.success('API example copied!')).catch(() => toast.error('Copy failed'));
  };

  // Available models to deploy (not yet deployed)
  const deployedIds = new Set(deployments.map(d => d.model_id));
  const undeployed = models.filter(m => !deployedIds.has(m.modelId));

  return (
    <motion.div variants={fadeInUp} initial="initial" animate="animate" className="space-y-6" data-testid="deploy-view">
      {/* Deploy New Model */}
      {undeployed.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base"><Rocket className="h-5 w-5 text-violet-500" />Deploy a Model</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">Select a trained model to deploy. It will get a public URL for anyone to make predictions.</p>
            <div className="grid gap-2 md:grid-cols-2">
              {undeployed.map(model => (
                <div key={model.modelId} className="flex items-center justify-between p-3 rounded-xl border hover:bg-accent/20 transition-colors" data-testid={`deploy-model-${model.modelId}`}>
                  <div>
                    <span className="font-semibold text-sm">{ALGO_NAMES[model.algorithm] || model.algorithm}</span>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {model.problemType} · {model.modelData?.featureNames?.length || model.modelData?.numericCols?.length + model.modelData?.categoricalCols?.length || '?'} features
                      {model.metrics && Object.entries(model.metrics).slice(0, 2).map(([k, v]) => (
                        <span key={k} className="ml-2">{k}: {typeof v === 'number' ? v.toFixed(3) : v}</span>
                      ))}
                    </div>
                  </div>
                  <Button size="sm" className="gap-1.5 bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600"
                    onClick={() => handleDeploy(model)} disabled={deployingId === model.modelId} data-testid={`deploy-btn-${model.modelId}`}>
                    {deployingId === model.modelId ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
                    Deploy
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Deployments */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base"><Globe className="h-5 w-5 text-emerald-500" />Active Deployments</CardTitle>
            <Badge variant="secondary">{deployments.filter(d => d.enabled).length} active</Badge>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-12"><RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" /></div>
          ) : deployments.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Globe className="h-12 w-12 mx-auto mb-3 opacity-30" />
              <p className="font-medium">No deployments yet</p>
              <p className="text-sm mt-1">Deploy a model above to get a public prediction link.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {deployments.map((dep, idx) => (
                <motion.div key={dep.deploy_id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`group rounded-xl border p-4 transition-all ${dep.enabled ? 'hover:shadow-md hover:bg-accent/10' : 'opacity-60 bg-muted/30'}`}
                  data-testid={`deployment-${idx}`}>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-sm">{dep.name}</span>
                        <Badge className={`text-[10px] px-1.5 py-0 border-0 ${dep.enabled ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}>
                          {dep.enabled ? 'Live' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{new Date(dep.created_at).toLocaleDateString()}</span>
                        <span className="flex items-center gap-1"><Activity className="h-3 w-3" />{dep.prediction_count || 0} predictions</span>
                        <code className="font-mono text-[10px] bg-muted px-1.5 py-0.5 rounded">{dep.deploy_id}</code>
                      </div>

                      {/* Public URL */}
                      <div className="flex items-center gap-2 mt-2 p-2 rounded-lg bg-muted/50">
                        <Link2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                        <code className="text-xs font-mono flex-1 truncate">{window.location.origin}/predict/{dep.deploy_id}</code>
                        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => copyUrl(dep.deploy_id)} data-testid={`copy-url-${idx}`}>
                          <Copy className="h-3 w-3" />
                        </Button>
                        <a href={`/predict/${dep.deploy_id}`} target="_blank" rel="noopener noreferrer">
                          <Button variant="ghost" size="icon" className="h-6 w-6"><ExternalLink className="h-3 w-3" /></Button>
                        </a>
                      </div>

                      {/* API example toggle */}
                      <button className="flex items-center gap-1 text-[10px] text-violet-500 mt-2 hover:underline" onClick={() => setShowApi(showApi === dep.deploy_id ? null : dep.deploy_id)} data-testid={`api-toggle-${idx}`}>
                        <Code className="h-3 w-3" />REST API {showApi === dep.deploy_id ? '▲' : '▼'}
                      </button>
                      <AnimatePresence>
                        {showApi === dep.deploy_id && (
                          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                            <div className="mt-2 p-3 rounded-lg bg-zinc-950 text-zinc-100 text-[11px] font-mono leading-relaxed relative">
                              <pre className="whitespace-pre-wrap">{`# Make a prediction via REST API
curl -X POST "${API_URL}/api/public/predict/${dep.deploy_id}" \\
  -H "Content-Type: application/json" \\
  -d '{"features": {"feature1": "value1", "feature2": 42}}'

# Python
import requests
r = requests.post("${API_URL}/api/public/predict/${dep.deploy_id}",
    json={"features": {"feature1": "value1"}})
print(r.json())`}</pre>
                              <Button variant="ghost" size="icon" className="absolute top-2 right-2 h-6 w-6 text-zinc-400 hover:text-white" onClick={() => copyApiExample(dep)}>
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <div className="flex gap-1 shrink-0">
                      <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => handleToggle(dep)} title={dep.enabled ? 'Disable' : 'Enable'} data-testid={`toggle-deploy-${idx}`}>
                        {dep.enabled ? <EyeOff className="h-4 w-4 text-amber-500" /> : <Eye className="h-4 w-4 text-emerald-500" />}
                      </Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => handleDelete(dep)} title="Delete deployment" data-testid={`delete-deploy-${idx}`}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
