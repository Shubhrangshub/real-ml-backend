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
- [x] **Metric Hover Tooltips** — All metrics (R², MAE, RMSE, Accuracy, Precision, Recall, F1, Silhouette, Davies-Bouldin, Calinski-Harabasz, CV Score, SHAP Value, Base Value) have plain-English tooltip explanations via MetricTip component
- [x] **Explainable AI (XAI) Module**:
  - SHAP Analysis tab: Global Feature Importance, Beeswarm Plot, Waterfall Plot (per-row), Dependence Plot (per-feature), Force Plot
  - LIME Explanation tab: Feature Contribution Chart, Prediction Probability Chart, SHAP vs LIME side-by-side comparison
  - Cluster Explanation tab: Cluster Feature Influence, Cluster SHAP Distribution
  - Row selector for individual explanations
  - "Explain This Prediction" button (computes both SHAP + LIME)
  - Performance optimization: sampling for datasets > 5000 rows

## Testing Status
- Iteration 12: 17/17 tests passed (100% frontend)
- All XAI visualizations verified working for supervised and unsupervised

## Backlog
- [ ] P1: Refactor App.js into modular components (~2700 lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
