# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard (AutoML Master) in React. All ML analysis runs directly in the browser using JavaScript libraries — no Python backend or server-side ML needed.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Custom client-side JS (supervised in App.js, unsupervised in unsupervisedML.js)
- **Backend**: FastAPI (minimal, serves health check)
- **Database**: MongoDB (minimal use)
- **All ML**: Runs in-browser — no server calls for training/prediction

## Core Files
- `/app/frontend/src/App.js` — Main UI + supervised ML engine + all features
- `/app/frontend/src/unsupervisedML.js` — Unsupervised ML engine (clustering, PCA, t-SNE, anomaly detection)
- `/app/frontend/src/index.css` — CSS with light/dark mode variables
- `/app/frontend/tailwind.config.js` — Tailwind config with dark mode class strategy

## Completed Features
### Phase 1-8 (Previous)
- [x] Supervised learning (11 algorithms: Decision Tree, Random Forest, KNN, Naive Bayes, Logistic Regression, SVM, Neural Network, Gradient Boosting, AdaBoost, Extra Trees, Ridge/Lasso)
- [x] Unsupervised learning (K-Means, DBSCAN, Hierarchical, PCA, t-SNE, Isolation Forest, LOF, Optimal K)
- [x] Dataset Scanner with 12 quality checks
- [x] K-Fold Cross Validation
- [x] UI/UX overhaul with MetricCards, tooltips, charts
- [x] Algorithm leaderboard and comparison
- [x] Single predictions with probability display
- [x] Cluster predictions for unsupervised models

### Phase 9 (Feb 2026)
- [x] **Batch Predictions from CSV Upload** — Upload CSV, predict all rows, display results table, export as CSV
- [x] **Model Export/Import** — Download models as JSON, import from JSON files in Model Library
- [x] **Export Report as PDF** — Generate PDF with dataset info, metrics, leaderboard, feature importance (jsPDF + autotable)
- [x] **Data Explorer tab** — Histograms with stats, correlation heatmap, scatter plot with correlation coefficient
- [x] **Dark Mode Toggle** — Sun/moon toggle in header, persists in localStorage, uses Tailwind dark class

## Testing Status
- Iteration 11: 24/24 tests passed (100% frontend)
- All 5 new features verified working

## Backlog
- [ ] P1: Refactor App.js into modular components (2400+ lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
