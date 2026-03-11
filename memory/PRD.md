# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (11 algorithms + K-Fold CV)
- **Storage**: React state + localStorage persistence
- **Backend**: DEPRECATED (not used)

## Core Features Implemented

### Phase 1 - Bug Fixes (Completed)
- [x] Fix Predictions tab, model persistence, dashboard metrics, trend indicators

### Phase 2 - Universal Dashboard (Completed)
- [x] Universal Analysis Engine, K-Means Clustering, Anomaly Detection, localStorage Persistence, Sample datasets

### Phase 3 - Supervised Learning Engine (Completed - Feb 2026)
- [x] Auto task detection, 80/20 train-test split
- [x] Regression: Linear, Ridge | Classification: Logistic, Decision Tree
- [x] Metrics: R², MAE, RMSE, Accuracy, Precision, Recall, Confusion Matrix
- [x] Algorithm Leaderboard, Feature Importance, Data Leakage Prevention

### Phase 4 - Advanced Algorithms & Visual Explainability (Completed - Feb 2026)
- [x] Regression: Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor
- [x] Classification: Random Forest Classifier, KNN, SVM (Linear), Naive Bayes (Gaussian)
- [x] F1 Score (Macro + Per-Class)
- [x] Model Comparison Bar Chart, Confusion Matrix Heatmap, Actual vs Predicted with reference line
- [x] Graph explanations on all visualizations, human-readable algorithm names (ALGO_NAMES)

### Phase 5 - K-Fold Cross Validation (Completed - Feb 2026)
- [x] Reusable `kFoldCrossValidation(X, y, k, trainFn, problemType)` function
- [x] `buildModelForAlgo` helper for DRY model training
- [x] Evaluation Mode selector: Train/Test Split (Fast) vs 5-Fold CV (Recommended)
- [x] CV applied to all 11 supervised algorithms (5 regression + 6 classification)
- [x] CV scores displayed in leaderboard (green "CV: XX.XX%" text)
- [x] Best model selected by CV score when CV mode is active
- [x] Cross-Validation Performance Chart (dual bars: CV Score + Test Score)
- [x] Chart explanation text for CV
- [x] Split info banner updated for CV mode
- [x] Default remains Train/Test Split for fast execution

## Algorithms Available
### Regression (Auto trains all)
Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, Baseline

### Classification (Auto trains all)
Logistic Regression, Decision Tree, Random Forest, KNN, SVM (Linear), Naive Bayes, Baseline

## Test Results
- Phase 3: 25/25 passed (iteration_4.json)
- Phase 4: 18/20 passed, 2 partial (iteration_5.json)
- Phase 5: 20/20 passed (iteration_6.json)

## Tech Stack
React 18, Tailwind CSS, Shadcn UI, Recharts, Framer Motion, simple-statistics, ml-kmeans, danfojs

## Key File
`/app/frontend/src/App.js` — All application logic (~1100 lines)

## Upcoming Tasks
- [ ] P1: Refactor App.js into modular components

## Future/Backlog
- [ ] Unsupervised: Hierarchical Clustering, DBSCAN, PCA
- [ ] Batch predictions from CSV upload
- [ ] Export reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab
- [ ] Model comparison dashboard (dedicated page)
