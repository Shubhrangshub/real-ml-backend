# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (11 algorithms + K-Fold CV + Dataset Scanner)
- **Storage**: React state + localStorage persistence
- **Backend**: DEPRECATED (not used)

## Core Features Implemented

### Phase 1-3 — Foundation (Completed)
- Bug fixes, Universal Dashboard, Supervised Learning Engine (Linear/Ridge/Logistic/Decision Tree)
- Auto task detection, 80/20 train-test split, evaluation metrics (R², MAE, RMSE, Accuracy, Precision, Recall, Confusion Matrix)

### Phase 4 — Advanced Algorithms & Visual Explainability (Completed)
- 7 new algorithms: Random Forest (Reg/Clf), Gradient Boosting, KNN, SVM, Naive Bayes
- F1 Score, Model Comparison Chart, Confusion Matrix Heatmap, graph explanations

### Phase 5 — K-Fold Cross Validation (Completed)
- 5-fold CV for all 11 supervised algorithms
- Evaluation mode toggle (Train/Test Split vs CV)
- CV Performance Chart, leaderboard ranked by CV score

### Phase 6 — Dashboard Overhaul + Dataset Scanner (Completed - Feb 2026)
- [x] **5 Stat Cards**: Total Models, Avg Score, Best Algorithm, Highest Score, Last Training
- [x] **3 Quick Insight Cards**: Best Performing, Most Used, Highest Accuracy
- [x] **Dataset Health Widget**: Rows, columns, missing values, outliers, readiness score (0-100), ready/needs-cleaning indicator
- [x] **Dataset Scanner** (`scanDataset` function): Missing values, duplicates, outliers (IQR), constant columns, high correlations (>0.9), target validation, class imbalance, feature scaling check, size warning, health score
- [x] **Model Leaderboard Panel**: Top 5 models by performance score
- [x] **3 Recharts Charts**: Model Performance Bar, Algorithm Usage Pie, Training Timeline Line
- [x] **Recent Models Table**: Algorithm, Problem Type, Target, Eval Mode, Score, Date
- [x] **Empty State**: Friendly "No Models Yet" with CTA button
- [x] **Auto-updates**: Dashboard reacts to models state changes in real-time
- [x] **Model storage enhanced**: Now includes evalMode + targetColumn per model

## All Algorithms
### Regression: Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
### Classification: Logistic, Decision Tree, Random Forest, KNN, SVM, Naive Bayes

## Test Results
- Phase 3: 25/25 passed (iteration_4.json)
- Phase 4: 18/20 passed (iteration_5.json)
- Phase 5: 20/20 passed (iteration_6.json)
- Phase 6: 20/20 passed (iteration_7.json)

## Key File
`/app/frontend/src/App.js` — ~1320 lines

## Upcoming Tasks
- [ ] P1: Refactor App.js into modular components

## Future/Backlog
- [ ] Full Dataset Scanner tab with auto-clean buttons (remove duplicates, fill missing, normalize)
- [ ] Unsupervised: Hierarchical Clustering, DBSCAN, PCA
- [ ] Batch predictions from CSV upload
- [ ] Export reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
