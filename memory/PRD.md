# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (Linear Regression, Ridge Regression, Logistic Regression, Decision Tree, K-Means, Anomaly Detection)
- **Storage**: React state + localStorage persistence
- **Backend**: DEPRECATED (not used)

## Core Features Implemented

### Phase 1 - Bug Fixes (Completed)
- [x] Fix Predictions tab navigation (no redirect, shows warning if no model)
- [x] Fix model persistence with setModels(prev => [...prev, newModel])
- [x] Fix Total Models dashboard metric (uses models.length)
- [x] Fix trend indicators (show N/A when value is 0)
- [x] Fix Predictions page auto-uses latest model

### Phase 2 - Universal Dashboard (Completed)
- [x] Universal Analysis Engine: Auto-profiles datasets on upload, detects column types
- [x] K-Means Clustering Dashboard with scatter visualization
- [x] Anomaly Detection Module with Z-Score and IQR methods
- [x] localStorage Persistence with race condition fix
- [x] Sample datasets (Loan Approval, House Prices, Insurance Costs, TV Shows)

### Phase 3 - Supervised Learning Engine (Completed - Feb 2026)
- [x] Automatic task detection: Regression vs Classification
- [x] 80/20 Train-Test Split with Fisher-Yates shuffle
- [x] Regression algorithms: Linear Regression, Ridge Regression
- [x] Classification algorithms: Logistic Regression, Decision Tree
- [x] Regression metrics: R² Score, MAE, RMSE
- [x] Classification metrics: Accuracy, Precision, Recall, Confusion Matrix, Per-Class Metrics
- [x] Algorithm Leaderboard ranking models by test performance
- [x] Predicted vs Actual scatter chart (regression)
- [x] Residual Plot (regression)
- [x] Feature Importance chart
- [x] Data Leakage Prevention (auto-removes ID/date columns)
- [x] Train vs Test comparison table (overfitting detection)
- [x] Baseline model for comparison
- [x] Model stored for predictions after training

## Sidebar Navigation (6 tabs)
1. Dashboard - ML operations overview
2. Analysis - Upload data, auto-profile, train models
3. Predictions - Generate predictions from trained models
4. Clusters - K-Means clustering dashboard
5. Anomalies - Outlier detection with Z-score/IQR
6. Model Library - Manage trained models (view, download, delete)

## Test Results
- Phase 1: 21/21 tests passed (iteration_2.json)
- Phase 2: 21/21 tests passed
- Phase 3: 25/25 tests passed (iteration_4.json)

## Tech Stack
- React 18, Tailwind CSS, Shadcn UI, Recharts, Framer Motion
- simple-statistics, ml-kmeans, danfojs (installed)

## Key File
- `/app/frontend/src/App.js` - All application logic (~854 lines)

## Upcoming Tasks (P1 - Refactoring)
- [ ] Refactor App.js into modular components:
  - AnalysisTab.js
  - SupervisedLearning.js
  - ClusteringTab.js
  - AnomaliesTab.js
  - PredictionsTab.js
  - ModelLibrary.js
  - Dashboard.js

## Future Tasks (P2 - Advanced Algorithms)
- [ ] Random Forest
- [ ] Gradient Boosting
- [ ] KNN
- [ ] SVM
- [ ] Hierarchical Clustering
- [ ] DBSCAN
- [ ] PCA

## Backlog
- [ ] Batch predictions from CSV upload
- [ ] Export training/analysis reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
- [ ] Model comparison dashboard
- [ ] Text processing (TF-IDF) for text-heavy datasets
