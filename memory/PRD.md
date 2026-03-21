# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard (AutoML Master) in React. All ML analysis runs directly in the browser using JavaScript libraries — no Python backend or server-side ML needed.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Custom client-side JS (supervised in App.js, unsupervised in unsupervisedML.js)
- **XAI Engine**: explainableAI.js (SHAP approximation + LIME)
- **Backend**: FastAPI (minimal, serves health check)
- **Database**: MongoDB (minimal use)
- **All ML**: Runs in-browser — no server calls for training/prediction/explanation

## Core Files
- `/app/frontend/src/App.js` — Main UI + supervised ML + all features (~3300 lines)
- `/app/frontend/src/unsupervisedML.js` — Unsupervised ML engine
- `/app/frontend/src/explainableAI.js` — SHAP & LIME computation engine
- `/app/frontend/src/index.css` — CSS with light/dark mode variables

## Completed Features

### Phase 1-8 (Previous)
- [x] Supervised learning (11 algorithms), Unsupervised learning, Dataset Scanner, K-Fold CV
- [x] UI/UX overhaul, Algorithm leaderboard, Predictions, Cluster predictions

### Phase 9 (Feb 2026)
- [x] Batch Predictions, Model Export/Import, PDF Reports, Data Explorer, Dark Mode

### Phase 10 (Feb 2026)
- [x] Metric Hover Tooltips, Explainable AI (SHAP + LIME + Cluster Explanations)

### Phase 11 (Feb 2026)
- [x] Performance Optimization (arrayMin/arrayMax, KNN, RF, DT optimizations)

### Phase 12 (Feb 2026)
- [x] Session Persistence (full state → localStorage, Clear Session button)

### Phase 13 (Feb 2026)
- [x] Enhanced XAI Dashboard (7 new charts, vibrant colors, section dividers, descriptions)

### Phase 14 (Feb 2026)
- [x] **Smart Guided Help System**:
  - **HelpTip Component**: Reusable (?) hover tooltip for all major inputs (Target Variable, Algorithm, Evaluation Mode, Record Row, Data Explorer columns)
  - **Getting Started Guide**: Collapsible 7-step panel (Upload → Review → Target → Train → Results → Predict → XAI) with automatic progress tracking (green checkmarks for completed steps)
  - **Target Variable Auto-Suggestion**: `suggestedTarget` useMemo analyzes column types & variability, shows banner with "Suggested target: [name]" and "Use this target" button
  - **Algorithm Descriptions**: ALGO_DESCRIPTIONS constant with 1-2 line explanations for all 10 algorithms, shown dynamically below the algorithm selector
  - **Smart XAI Row Suggestions**: `smartRowSuggestions` useMemo provides 4 chips: Random sample, Highest prediction, Lowest prediction, Representative (median) — click to auto-fill row input
  - **XAI Method Descriptions**: Pre-computation help text for SHAP, LIME, and Cluster explanation tabs
  - **Error Prevention**: Better empty-state messages guiding users to upload data/train first
  - **Enhanced Descriptions**: Updated page header subtitles, upload area guidance, prediction form guidance
  - **Guide Access**: "Help Guide" button in header + "New here? Open Guide" link on empty dashboard

## Testing Status
- Iteration 16: 24/24 tests passed (100% — Help System)
- Iteration 15: 14/14 tests passed (100% — XAI enhancement)
- Iteration 14: 22/22 tests passed (100% — session persistence)
- Iteration 13: 27/27 tests passed (100% — performance optimization)

## Backlog
- [ ] P1: Counterfactual Explanations ("what would need to change" for different prediction)
- [ ] P1: Shareable Report Link (unique URL for sharing analysis reports)
- [ ] P2: Refactor App.js into modular components (~3300+ lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
