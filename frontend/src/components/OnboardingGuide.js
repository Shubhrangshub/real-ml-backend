import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, ChevronRight, ChevronLeft, X, Rocket, Sparkles, PartyPopper } from 'lucide-react';
import { Button } from '@/components/ui/button';

const TOUR_STEPS = [
  {
    target: '[data-testid="app-sidebar"]',
    title: 'Welcome to AutoML!',
    desc: 'This sidebar is your main navigation. Each tab unlocks a different ML capability.',
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-analysis"]',
    title: 'Start with Analysis',
    desc: 'Upload your own CSV or pick a sample dataset to get started in seconds.',
    position: 'right',
    view: 'analysis',
  },
  {
    target: '[data-testid="sample-dataset-0"]',
    title: 'Try a Sample Dataset',
    desc: 'Click any sample to instantly load it — no upload needed. Perfect for exploring.',
    position: 'bottom',
    view: 'analysis',
    waitForSelector: true,
  },
  {
    target: '[data-testid="target-column-select"]',
    title: 'Pick What to Predict',
    desc: 'Select your target column — the variable you want the ML model to learn and predict.',
    position: 'bottom',
    view: 'analysis',
    requiresData: true,
  },
  {
    target: '[data-testid="start-training-btn"]',
    title: 'Train Models',
    desc: 'One click trains 8+ algorithms simultaneously. Results appear in seconds — all in your browser!',
    position: 'bottom',
    view: 'analysis',
    requiresData: true,
  },
  {
    target: '[data-testid="nav-dashboard"]',
    title: 'Your Command Center',
    desc: 'The Dashboard shows your best models, key metrics, recent history, and leaderboard at a glance.',
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-explainability"]',
    title: 'Understand Your Models',
    desc: 'SHAP & LIME explain WHY predictions happen — turn black-box models into actionable insights.',
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-leaderboard"]',
    title: 'Track Performance',
    desc: 'Every model is auto-saved to the leaderboard. Compare algorithms and track trends over time.',
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-history"]',
    title: 'Never Lose Work',
    desc: "Analyses are auto-saved when you switch datasets. Load any past session from History — it's all here.",
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-whatif"]',
    title: 'What-If Analyzer',
    desc: 'Tweak feature values side-by-side and instantly see how predictions change. Great for scenario planning!',
    position: 'right',
    view: null,
  },
  {
    target: '[data-testid="nav-deploy"]',
    title: 'Deploy Your Model',
    desc: 'Deploy any trained model and get a public link. Anyone can make predictions without logging in — plus a REST API for developers.',
    position: 'right',
    view: null,
  },
];

const MILESTONES = [
  { id: 'dataset', label: 'Load a dataset', icon: '1' },
  { id: 'train', label: 'Train a model', icon: '2' },
  { id: 'predict', label: 'Make a prediction', icon: '3' },
  { id: 'explain', label: 'Explore explainability', icon: '4' },
  { id: 'save', label: 'Save an analysis', icon: '5' },
];

// ========================= SPOTLIGHT TOUR =========================
function SpotlightTour({ isActive, onClose, steps, setActiveView }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [spotlightRect, setSpotlightRect] = useState(null);
  const [tooltipStyle, setTooltipStyle] = useState({});
  const rafRef = useRef(null);

  const positionSpotlight = useCallback(() => {
    const step = steps[currentStep];
    if (!step) return;

    // Navigate to correct view if needed
    if (step.view && setActiveView) {
      setActiveView(step.view);
    }

    // Small delay for DOM to update after view change
    const findElement = () => {
      const el = document.querySelector(step.target);
      if (!el) {
        // Element not found yet, retry
        rafRef.current = requestAnimationFrame(findElement);
        return;
      }

      const rect = el.getBoundingClientRect();
      const padding = 8;
      setSpotlightRect({
        top: rect.top - padding,
        left: rect.left - padding,
        width: rect.width + padding * 2,
        height: rect.height + padding * 2,
      });

      // Calculate tooltip position
      const tooltip = {};
      const tooltipWidth = 320;
      const tooltipHeight = 160;

      if (step.position === 'right') {
        tooltip.top = Math.max(16, rect.top);
        tooltip.left = rect.right + 20;
        if (tooltip.left + tooltipWidth > window.innerWidth - 16) {
          tooltip.left = rect.left - tooltipWidth - 20;
        }
      } else if (step.position === 'bottom') {
        tooltip.top = rect.bottom + 16;
        tooltip.left = Math.max(16, rect.left);
        if (tooltip.left + tooltipWidth > window.innerWidth - 16) {
          tooltip.left = window.innerWidth - tooltipWidth - 16;
        }
        if (tooltip.top + tooltipHeight > window.innerHeight - 16) {
          tooltip.top = rect.top - tooltipHeight - 16;
        }
      } else if (step.position === 'left') {
        tooltip.top = rect.top;
        tooltip.left = rect.left - tooltipWidth - 20;
      } else {
        tooltip.top = rect.bottom + 16;
        tooltip.left = rect.left;
      }

      setTooltipStyle(tooltip);
    };

    // Delay to let view transitions settle
    setTimeout(findElement, 200);

    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [currentStep, steps, setActiveView]);

  useEffect(() => {
    if (!isActive) return;
    positionSpotlight();

    const handleResize = () => positionSpotlight();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isActive, positionSpotlight]);

  if (!isActive) return null;

  const step = steps[currentStep];
  const isLast = currentStep === steps.length - 1;
  const isFirst = currentStep === 0;

  return (
    <div className="fixed inset-0 z-[100]" data-testid="spotlight-tour">
      {/* Dark overlay with spotlight cutout */}
      <div className="absolute inset-0" onClick={onClose}>
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <mask id="spotlight-mask">
              <rect width="100%" height="100%" fill="white" />
              {spotlightRect && (
                <rect
                  x={spotlightRect.left} y={spotlightRect.top}
                  width={spotlightRect.width} height={spotlightRect.height}
                  rx="12" fill="black"
                />
              )}
            </mask>
          </defs>
          <rect width="100%" height="100%" fill="rgba(0,0,0,0.65)" mask="url(#spotlight-mask)" />
        </svg>
      </div>

      {/* Spotlight border glow */}
      {spotlightRect && (
        <div
          className="absolute rounded-xl border-2 border-violet-400 shadow-[0_0_20px_rgba(139,92,246,0.4)] pointer-events-none transition-all duration-300"
          style={{
            top: spotlightRect.top, left: spotlightRect.left,
            width: spotlightRect.width, height: spotlightRect.height,
          }}
        />
      )}

      {/* Tooltip */}
      <motion.div
        key={currentStep}
        initial={{ opacity: 0, y: 10, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.2 }}
        className="absolute z-[101] w-80 rounded-2xl bg-white dark:bg-zinc-900 border shadow-2xl overflow-hidden"
        style={tooltipStyle}
        onClick={e => e.stopPropagation()}
        data-testid="tour-tooltip"
      >
        {/* Progress bar */}
        <div className="h-1 bg-gray-100 dark:bg-zinc-800">
          <div
            className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>

        <div className="p-5">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="h-6 w-6 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white text-xs font-bold flex items-center justify-center">
                {currentStep + 1}
              </span>
              <h3 className="font-bold text-sm">{step?.title}</h3>
            </div>
            <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors" data-testid="tour-close">
              <X className="h-4 w-4" />
            </button>
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed mb-4">{step?.desc}</p>

          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">{currentStep + 1} of {steps.length}</span>
            <div className="flex gap-2">
              {!isFirst && (
                <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={() => setCurrentStep(prev => prev - 1)} data-testid="tour-prev">
                  <ChevronLeft className="h-3.5 w-3.5 mr-1" />Back
                </Button>
              )}
              {isFirst && (
                <Button variant="ghost" size="sm" className="h-8 text-xs text-muted-foreground" onClick={onClose} data-testid="tour-skip">
                  Skip tour
                </Button>
              )}
              <Button size="sm" className="h-8 text-xs bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600"
                onClick={() => isLast ? onClose() : setCurrentStep(prev => prev + 1)} data-testid="tour-next">
                {isLast ? <><PartyPopper className="h-3.5 w-3.5 mr-1" />Got it!</> : <>Next<ChevronRight className="h-3.5 w-3.5 ml-1" /></>}
              </Button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

// ========================= PROGRESS PILL =========================
function ProgressPill({ milestones, completed, onStartTour }) {
  const [expanded, setExpanded] = useState(false);
  const completedCount = completed.filter(Boolean).length;
  const total = milestones.length;
  const allDone = completedCount === total;
  const progress = completedCount / total;

  // Auto-hide after all milestones done (with celebration delay)
  const [celebrated, setCelebrated] = useState(false);
  useEffect(() => {
    if (allDone && !celebrated) {
      setCelebrated(true);
      setTimeout(() => setExpanded(true), 500);
    }
  }, [allDone, celebrated]);

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
    <div className="fixed bottom-6 right-6 z-50" data-testid="progress-pill">
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
            {/* Circular progress */}
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

              {/* Progress bar */}
              <div className="h-2 rounded-full bg-gray-100 dark:bg-zinc-800 mb-4 overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress * 100}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>

              {/* Milestones */}
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
                <Button variant="outline" size="sm" className="w-full text-xs h-8" onClick={onStartTour} data-testid="retake-tour-btn">
                  <Rocket className="h-3.5 w-3.5 mr-1.5" />Retake the tour
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
  const [tourActive, setTourActive] = useState(false);

  // Auto-start tour for first-time users
  useEffect(() => {
    try {
      const seen = localStorage.getItem('automl_tour_seen');
      if (!seen) {
        // Small delay so the app renders first
        const timer = setTimeout(() => setTourActive(true), 1500);
        return () => clearTimeout(timer);
      }
    } catch {}
  }, []);

  const handleCloseTour = useCallback(() => {
    setTourActive(false);
    try { localStorage.setItem('automl_tour_seen', 'true'); } catch {}
  }, []);

  const handleStartTour = useCallback(() => {
    setTourActive(true);
  }, []);

  // Filter tour steps based on current data state
  const availableSteps = TOUR_STEPS.filter(s => {
    if (s.requiresData && !csvText) return false;
    if (s.waitForSelector && !csvText) return true; // show even without data
    return true;
  });

  // Milestone completion tracking
  const milestoneCompleted = [
    !!csvText,
    !!(trainingResult || unsupervisedResult),
    predictionHistory?.length > 0,
    !!(shapGlobal || limeResult),
    !!hasSavedOnce,
  ];

  return (
    <>
      <SpotlightTour
        isActive={tourActive}
        onClose={handleCloseTour}
        steps={availableSteps}
        setActiveView={setActiveView}
      />
      <ProgressPill
        milestones={MILESTONES}
        completed={milestoneCompleted}
        onStartTour={handleStartTour}
      />
    </>
  );
}
