import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2, X, Rocket, Sparkles, PartyPopper, ChevronRight, ChevronLeft,
  Zap, Target, Play, Eye, Save, BarChart3, SlidersHorizontal, Sliders,
  Settings2, RocketIcon, FileDown,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

// ========================= GUIDE STEPS =========================
const GUIDE_STEPS = [
  { icon: Zap, title: 'Load a Dataset', desc: 'Go to Analysis and upload a CSV or pick a sample dataset to get started.' },
  { icon: Target, title: 'Pick a Target', desc: 'Select which column you want to predict. The system auto-detects classification vs regression.' },
  { icon: Play, title: 'Train Models', desc: 'Click Train — the engine runs multiple algorithms and ranks them on a leaderboard.' },
  { icon: BarChart3, title: 'View Results', desc: 'Check the Dashboard for an overview, or dive into Predictions, Compare, and Data Explorer tabs.' },
  { icon: Eye, title: 'Explain Models', desc: 'Use SHAP and LIME in the Explainability tab to understand what drives each prediction.' },
  { icon: Settings2, title: 'Preprocess Data', desc: 'Go to Preprocess to handle missing values, scale features, and apply smart recommendations.' },
  { icon: SlidersHorizontal, title: 'Tune Hyperparameters', desc: 'Use the Tune tab to optimize any model with Grid, Random, or Bayesian search.' },
  { icon: Sliders, title: 'What-If Analysis', desc: 'Tweak feature values in the What-If tab and see how predictions change in real time.' },
  { icon: RocketIcon, title: 'Deploy Models', desc: 'Deploy your best model as a REST API with a public prediction link.' },
  { icon: FileDown, title: 'Export Reports', desc: 'Download PDF reports from the toolbar or from any saved analysis in History.' },
  { icon: Save, title: 'Save & Share', desc: 'Save your work in History. Share a link so others can view your analysis.' },
];

// ========================= SIMPLE GUIDE DIALOG =========================
function GuideDialog({ isOpen, onClose, setActiveView }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (isOpen) setStep(0);
  }, [isOpen]);

  if (!isOpen) return null;

  const current = GUIDE_STEPS[step];
  const Icon = current.icon;
  const isLast = step === GUIDE_STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" data-testid="guide-dialog">
      {/* Backdrop — click to close */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />

      {/* Dialog */}
      <motion.div
        key="guide"
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.2 }}
        className="relative z-10 w-full max-w-md mx-4 rounded-2xl bg-white dark:bg-zinc-900 border shadow-2xl overflow-hidden"
        data-testid="guide-dialog-content"
      >
        {/* Progress bar */}
        <div className="h-1.5 bg-gray-100 dark:bg-zinc-800">
          <motion.div
            className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500"
            animate={{ width: `${((step + 1) / GUIDE_STEPS.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-5">
            <span className="text-xs text-muted-foreground font-medium">
              Step {step + 1} of {GUIDE_STEPS.length}
            </span>
            <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors" data-testid="guide-close">
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Step content */}
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="mb-6"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shrink-0">
                  <Icon className="h-5 w-5 text-white" />
                </div>
                <h3 className="font-bold text-base">{current.title}</h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed pl-[52px]">{current.desc}</p>
            </motion.div>
          </AnimatePresence>

          {/* Step indicators */}
          <div className="flex items-center justify-center gap-1.5 mb-5">
            {GUIDE_STEPS.map((_, i) => (
              <button
                key={i}
                onClick={() => setStep(i)}
                className={`h-1.5 rounded-full transition-all duration-300 ${i === step ? 'w-6 bg-violet-500' : i < step ? 'w-1.5 bg-violet-300 dark:bg-violet-700' : 'w-1.5 bg-gray-200 dark:bg-zinc-700'}`}
                data-testid={`guide-dot-${i}`}
              />
            ))}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <Button
              variant="ghost" size="sm" className="h-8 text-xs"
              onClick={() => step === 0 ? onClose() : setStep(s => s - 1)}
              data-testid="guide-prev"
            >
              {step === 0 ? 'Skip' : <><ChevronLeft className="h-3.5 w-3.5 mr-1" />Back</>}
            </Button>
            <Button
              size="sm"
              className="h-8 text-xs bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600"
              onClick={() => isLast ? onClose() : setStep(s => s + 1)}
              data-testid="guide-next"
            >
              {isLast ? <><PartyPopper className="h-3.5 w-3.5 mr-1" />Done!</> : <>Next<ChevronRight className="h-3.5 w-3.5 ml-1" /></>}
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// ========================= MILESTONES =========================
const MILESTONES = [
  { id: 'load', label: 'Load a dataset', icon: '1' },
  { id: 'train', label: 'Train models', icon: '2' },
  { id: 'predict', label: 'Make predictions', icon: '3' },
  { id: 'explain', label: 'Explain a model (SHAP/LIME)', icon: '4' },
  { id: 'save', label: 'Save an analysis', icon: '5' },
];

// ========================= PROGRESS PILL =========================
function ProgressPill({ milestones, completed, onStartGuide }) {
  const [expanded, setExpanded] = useState(false);
  const completedCount = completed.filter(Boolean).length;
  const total = milestones.length;
  const progress = completedCount / total;
  const allDone = completedCount === total;

  // Don't render if dismissed permanently
  const [dismissed, setDismissed] = useState(() => {
    try { return localStorage.getItem('automl_milestones_dismissed') === 'true'; } catch { return false; }
  });

  if (dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try { localStorage.setItem('automl_milestones_dismissed', 'true'); } catch {}
  };

  return (
    <div className="fixed bottom-6 right-6 z-40" data-testid="progress-pill">
      <AnimatePresence mode="wait">
        {!expanded ? (
          <motion.button
            key="pill"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setExpanded(true)}
            className="flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-white dark:bg-zinc-900 border shadow-lg hover:shadow-xl transition-shadow"
            data-testid="progress-pill-collapsed"
          >
            <div className="relative h-8 w-8">
              <svg className="h-8 w-8 -rotate-90" viewBox="0 0 32 32">
                <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" className="text-gray-100 dark:text-zinc-800" strokeWidth="3" />
                <circle cx="16" cy="16" r="13" fill="none" stroke="url(#pill-gradient)" strokeWidth="3"
                  strokeDasharray={`${progress * 81.68} 81.68`} strokeLinecap="round" />
                <defs><linearGradient id="pill-gradient" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#8B5CF6" /><stop offset="100%" stopColor="#D946EF" /></linearGradient></defs>
              </svg>
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold">{completedCount}</span>
            </div>
            <span className="text-sm font-medium whitespace-nowrap pr-1">
              {allDone ? 'All done!' : `${completedCount}/${total} complete`}
            </span>
            {allDone && <Sparkles className="h-4 w-4 text-amber-500" />}
          </motion.button>
        ) : (
          <motion.div
            key="expanded"
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            className="w-72 rounded-2xl bg-white dark:bg-zinc-900 border shadow-2xl overflow-hidden"
            data-testid="progress-pill-expanded"
          >
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Rocket className="h-4 w-4 text-violet-500" />
                  <span className="font-bold text-sm">Your Progress</span>
                </div>
                <button onClick={() => setExpanded(false)} className="text-muted-foreground hover:text-foreground" data-testid="progress-pill-close">
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="h-2 rounded-full bg-gray-100 dark:bg-zinc-800 mb-4 overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress * 100}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>

              <div className="space-y-2 mb-4">
                {milestones.map((m, idx) => {
                  const done = completed[idx];
                  return (
                    <div key={m.id} className={`flex items-center gap-2.5 p-2 rounded-lg transition-colors ${done ? 'bg-emerald-50 dark:bg-emerald-950/20' : 'bg-gray-50 dark:bg-zinc-800/50'}`} data-testid={`milestone-${m.id}`}>
                      {done
                        ? <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0" />
                        : <span className="h-5 w-5 rounded-full bg-gray-200 dark:bg-zinc-700 text-xs font-bold flex items-center justify-center shrink-0 text-muted-foreground">{m.icon}</span>
                      }
                      <span className={`text-xs ${done ? 'line-through text-muted-foreground' : 'font-medium'}`}>{m.label}</span>
                    </div>
                  );
                })}
              </div>

              {allDone ? (
                <div className="text-center p-3 rounded-xl bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-950/20 dark:to-fuchsia-950/20 mb-2">
                  <PartyPopper className="h-6 w-6 mx-auto mb-1 text-amber-500" />
                  <p className="text-xs font-semibold">You're an AutoML pro!</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">You've explored all the core features.</p>
                </div>
              ) : (
                <Button variant="outline" size="sm" className="w-full text-xs h-8" onClick={onStartGuide} data-testid="open-guide-btn">
                  <Rocket className="h-3.5 w-3.5 mr-1.5" />View Getting Started Guide
                </Button>
              )}

              <button onClick={handleDismiss} className="w-full text-[10px] text-muted-foreground hover:text-foreground mt-2 text-center transition-colors" data-testid="dismiss-pill-btn">
                Dismiss permanently
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ========================= MAIN EXPORT =========================
export default function OnboardingGuide({ setActiveView, csvText, trainingResult, unsupervisedResult, predictionHistory, shapGlobal, limeResult, hasSavedOnce }) {
  const [guideOpen, setGuideOpen] = useState(false);

  // Auto-open guide for first-time users
  useEffect(() => {
    try {
      const seen = localStorage.getItem('automl_tour_seen');
      if (!seen) {
        const timer = setTimeout(() => setGuideOpen(true), 1500);
        return () => clearTimeout(timer);
      }
    } catch {}
  }, []);

  const handleCloseGuide = useCallback(() => {
    setGuideOpen(false);
    try { localStorage.setItem('automl_tour_seen', 'true'); } catch {}
  }, []);

  const handleOpenGuide = useCallback(() => {
    setGuideOpen(true);
  }, []);

  const milestoneCompleted = [
    !!csvText,
    !!(trainingResult || unsupervisedResult),
    predictionHistory?.length > 0,
    !!(shapGlobal || limeResult),
    !!hasSavedOnce,
  ];

  return (
    <>
      <GuideDialog
        isOpen={guideOpen}
        onClose={handleCloseGuide}
        setActiveView={setActiveView}
      />
      <ProgressPill
        milestones={MILESTONES}
        completed={milestoneCompleted}
        onStartGuide={handleOpenGuide}
      />
    </>
  );
}
