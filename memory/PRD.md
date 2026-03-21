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
- `/app/frontend/src/App.js` — Main UI + supervised ML + all features (~3100 lines)
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
- [x] Metric Hover Tooltips
- [x] Explainable AI (XAI) Module (SHAP + LIME + Cluster Explanations)

### Phase 11 (Feb 2026)
- [x] Performance Optimization (arrayMin/arrayMax, KNN bounded insertion, RF voting, DT sort optimization)

### Phase 12 (Feb 2026)
- [x] Session Persistence (full state saved to localStorage, restored on load, Clear Session button)

### Phase 13 (Feb 2026)
- [x] **Enhanced XAI Dashboard** — Complete overhaul of explainability module:
  - **Scalable computation**: Stratified sampling (cap 800 from 2000+) for large datasets
  - **SHAP Tab — Global Explainability**:
    - Global Feature Importance (enhanced vibrant gradient colors, rich tooltips with rank)
    - SHAP Summary Plot (NEW — positive/negative mean SHAP per feature, stacked bars)
    - Beeswarm Plot (enhanced — blue→violet→pink gradient legend bar, larger dots)
  - **SHAP Tab — Local Explainability**:
    - Instance Details Panel (NEW — table showing feature values + SHAP contributions + impact bars)
    - Waterfall Plot (enhanced — running total in tooltip)
    - Force Plot (enhanced — 5-color gradients for pos/neg, hover scale animations)
  - **SHAP Tab — Feature Analysis**:
    - Dependence Plot (enhanced — custom tooltips, indigo scatter)
    - Feature vs Prediction Scatter (NEW — feature value vs model prediction)
  - **LIME Tab — Local Explanation**:
    - Instance Explanation Panel (NEW — prediction, intercept, R² fit, weight/contribution table)
    - Feature Contribution Chart (enhanced — emerald/rose colors, rich tooltips)
    - Positive vs Negative Feature Impact (NEW — split view with gradient progress bars)
    - Probability Distribution (enhanced — 8-color palette)
    - SHAP vs LIME Comparison (enhanced — bordered panels with violet/emerald themes)
  - **Cluster Tab — Cluster Explainability**:
    - Cluster Feature Influence (enhanced — blue gradient bars)
    - Cluster Comparison Bar Chart (NEW — grouped bars per feature across all clusters + overall)
    - PCA Scatter Plot (NEW — color-coded by cluster with variance % labels)
    - Feature Distribution per Cluster (NEW — mini bar charts per feature per cluster)
    - Cluster SHAP Distribution beeswarm (enhanced)
  - **UI Enhancements**: Gradient section dividers, gradient action buttons (violet for SHAP, emerald for LIME, blue for Clusters), 2-3 sentence plain-English descriptions for every chart, colorful themed card borders
  - **Performance**: Stratified sampling, new state persisted in session, `xaiCacheRef` for future cache support

## Testing Status
- Iteration 15: 14/14 tests passed (100% — XAI enhancement regression)
- Iteration 14: 22/22 tests passed (100% — session persistence)
- Iteration 13: 27/27 tests passed (100% — performance optimization)

## Backlog
- [ ] P1: Counterfactual Explanations ("what would need to change" for different prediction)
- [ ] P1: Shareable Report Link (unique URL for sharing analysis reports)
- [ ] P2: Refactor App.js into modular components (~3100+ lines)
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P3: Dataset preprocessing pipeline UI
