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
- `/app/frontend/src/App.js` — Main UI + supervised ML + all features
- `/app/frontend/src/unsupervisedML.js` — Unsupervised ML engine
- `/app/frontend/src/explainableAI.js` — SHAP & LIME computation engine
- `/app/frontend/src/index.css` — CSS with light/dark mode variables

## Completed Features

### Phase 1-8 (Previous)
- [x] Supervised learning (11 algorithms)
- [x] Unsupervised learning (K-Means, DBSCAN, Hierarchical, PCA, t-SNE, etc.)
- [x] Dataset Scanner with 12 quality checks
- [x] K-Fold Cross Validation
- [x] UI/UX overhaul with MetricCards, tooltips, charts
- [x] Algorithm leaderboard and comparison
- [x] Single predictions with probability display
- [x] Cluster predictions for unsupervised models

### Phase 9 (Feb 2026)
- [x] Batch Predictions from CSV Upload
- [x] Model Export/Import
- [x] Export Report as PDF (jsPDF)
- [x] Data Explorer tab (histograms, correlation heatmap, scatter)
- [x] Dark Mode Toggle

### Phase 10 (Feb 2026)
- [x] **Metric Hover Tooltips** — All metrics have plain-English tooltip explanations via MetricTip component
- [x] **Explainable AI (XAI) Module**:
  - SHAP Analysis tab: Global Feature Importance, Beeswarm Plot, Waterfall Plot, Dependence Plot, Force Plot
  - LIME Explanation tab: Feature Contribution Chart, Prediction Probability Chart, SHAP vs LIME comparison
  - Cluster Explanation tab: Cluster Feature Influence, Cluster SHAP Distribution
  - Row selector for individual explanations
  - "Explain This Prediction" button (computes both SHAP + LIME)

### Phase 11 (Feb 2026)
- [x] **Performance Optimization** — All ML engine code optimized without changing algorithms or outputs:
  - Replaced Math.min/max(...bigArray) with loop-based arrayMin/arrayMax/arrayMinMax
  - Optimized Decision Tree findBestSplit, KNN prediction, Random Forest voting
  - Optimized extractImportance with Map-based lookup
  - Reuse parsed dataProfile.rows in handleTrain
  - Optimized explainableAI.js: shared background/basePred, Float64Array scratch

### Phase 12 (Feb 2026)
- [x] **Session Persistence** — Full application state persisted to localStorage:
  - Saved state: dataset (csvText), target column, algorithm, eval mode, cleaning log, training results, models, predictions, SHAP/LIME results, unsupervised results, UI selections
  - Automatic restore on app load — skips "No Dataset Uploaded" screen
  - Debounced save (500ms) after every state change
  - Models also saved separately to `automl_models` for backward compat
  - "Clear Session" button in header (visible when data is loaded) resets all state and localStorage
  - dataProfile and columns re-derived from csvText on restore

## Testing Status
- Iteration 14: 22/22 tests passed (100% — session persistence)
- Iteration 13: 27/27 tests passed (100% — performance optimization regression)

## Backlog
- [ ] P1: Counterfactual Explanations ("what would need to change" for different prediction)
- [ ] P1: Shareable Report Link (unique URL for sharing analysis reports)
- [ ] P2: Refactor App.js into modular components (~2800+ lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
