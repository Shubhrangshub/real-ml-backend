# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (Linear/Ridge/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, K-Means, Anomaly Detection)
- **Storage**: React state + localStorage persistence
- **Backend**: DEPRECATED (not used)

## Core Features Implemented

### Phase 1 - Bug Fixes (Completed)
- [x] Fix Predictions tab navigation
- [x] Fix model persistence
- [x] Fix Total Models dashboard metric
- [x] Fix trend indicators
- [x] Fix Predictions page auto-uses latest model

### Phase 2 - Universal Dashboard (Completed)
- [x] Universal Analysis Engine: Auto-profiles datasets
- [x] K-Means Clustering Dashboard
- [x] Anomaly Detection Module (Z-Score/IQR)
- [x] localStorage Persistence with race condition fix
- [x] Sample datasets

### Phase 3 - Supervised Learning Engine (Completed - Feb 2026)
- [x] Automatic task detection: Regression vs Classification
- [x] 80/20 Train-Test Split
- [x] Regression: Linear Regression, Ridge Regression
- [x] Classification: Logistic Regression, Decision Tree
- [x] Regression metrics: R², MAE, RMSE
- [x] Classification metrics: Accuracy, Precision, Recall, Confusion Matrix
- [x] Algorithm Leaderboard, Feature Importance, Data Leakage Prevention

### Phase 4 - Advanced Algorithms & Visual Explainability (Completed - Feb 2026)
- [x] **New Regression Algorithms:** Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor
- [x] **New Classification Algorithms:** Random Forest Classifier, KNN Classifier, SVM (Linear), Naive Bayes (Gaussian)
- [x] **F1 Score** (Macro + Per-Class) added to classification metrics
- [x] **Model Comparison Bar Chart** with color-coded algorithm bars and explanation
- [x] **Confusion Matrix Heatmap** with green diagonal (correct) / red off-diagonal (errors) + explanation
- [x] **Actual vs Predicted Scatter Plot** with perfect-fit reference line + explanation
- [x] **Residual Plot** with y=0 reference line + explanation
- [x] **Feature Importance Chart** with explanation (especially for tree-based models)
- [x] **Graph Explanations** on every visualization (3-4 line plain-English descriptions)
- [x] **Algorithm Dropdown** with optgroups (Regression, Classification, Both)
- [x] **Human-readable Algorithm Names** across all views (ALGO_NAMES constant)
- [x] **Best Model Highlighting** in leaderboard with colored border and label
- [x] Auto mode trains ALL relevant algorithms and selects the best

## Algorithms Available
### Regression (Auto trains all)
- Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, Baseline

### Classification (Auto trains all)
- Logistic Regression, Decision Tree, Random Forest, KNN, SVM (Linear), Naive Bayes, Baseline

## Sidebar Navigation (6 tabs)
1. Dashboard - ML operations overview
2. Analysis - Upload data, auto-profile, train models
3. Predictions - Generate predictions from trained models
4. Clusters - K-Means clustering dashboard
5. Anomalies - Outlier detection with Z-score/IQR
6. Model Library - Manage trained models

## Test Results
- Phase 1: 21/21 tests passed
- Phase 2: 21/21 tests passed
- Phase 3: 25/25 tests passed (iteration_4.json)
- Phase 4: 18/20 tests passed, 2 partial (iteration_5.json)

## Tech Stack
- React 18, Tailwind CSS, Shadcn UI, Recharts, Framer Motion
- simple-statistics, ml-kmeans, danfojs

## Key File
- `/app/frontend/src/App.js` - All application logic (~950+ lines)

## Upcoming Tasks
- [ ] P1: Refactor App.js into modular components (AnalysisTab, ClusteringTab, etc.)

## Future/Backlog
- [ ] Unsupervised: Hierarchical Clustering, DBSCAN, PCA
- [ ] Batch predictions from CSV upload
- [ ] Export reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
- [ ] Model comparison dashboard (dedicated page)
- [ ] Text processing (TF-IDF)
