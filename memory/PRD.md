# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (11 algorithms + K-Fold CV + Dataset Scanner)
- **Storage**: React state + localStorage persistence
- **Backend**: DEPRECATED (not used)

## Core Features Implemented

### Phase 1-3 -- Foundation (Completed)
- Bug fixes, Universal Dashboard, Supervised Learning Engine (Linear/Ridge/Logistic/Decision Tree)
- Auto task detection, 80/20 train-test split, evaluation metrics (R2, MAE, RMSE, Accuracy, Precision, Recall, Confusion Matrix)

### Phase 4 -- Advanced Algorithms & Visual Explainability (Completed)
- 7 new algorithms: Random Forest (Reg/Clf), Gradient Boosting, KNN, SVM, Naive Bayes
- F1 Score, Model Comparison Chart, Confusion Matrix Heatmap, graph explanations

### Phase 5 -- K-Fold Cross Validation (Completed)
- 5-fold CV for all 11 supervised algorithms
- Evaluation mode toggle (Train/Test Split vs CV)
- CV Performance Chart, leaderboard ranked by CV score

### Phase 6 -- Dashboard Overhaul + Dataset Scanner (Completed - Feb 2026)
- 5 Stat Cards, 3 Quick Insight Cards, Dataset Health Widget
- Dataset Scanner with 12 quality checks, health score, one-click cleaning
- Model Leaderboard, Charts (Performance, Algorithm Usage, Timeline), Recent Models Table

### Phase 7 -- UI/UX & Explainability Overhaul (Completed - Feb 2026)
- [x] Color-coded MetricCards (green/sky/amber/red) based on score thresholds with quality labels (Excellent/Good/Fair/Needs Work)
- [x] Hover tooltips on all metrics (R2, MAE, RMSE, Accuracy, Precision, Recall, F1) with plain-English explanations
- [x] Plain-English summary in training results header (problem type, sample count, best algorithm, score)
- [x] Improved chart headers with CardDescription explaining what each visualization shows
- [x] Overfitting detection in Train vs Test comparison (Gap column, red highlight if >15pp)
- [x] Dynamic prediction form replacing JSON textarea (labeled number inputs for numeric, dropdowns for categorical)
- [x] Enhanced prediction result card with large value display and Input Summary badges
- [x] Color-coded leaderboard scores using getScoreColor
- [x] Improved chart styling: reference lines, better colors, opacity adjustments
- [x] Updated Confusion Matrix header with clearer explanation

## All Algorithms
### Regression: Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
### Classification: Logistic, Decision Tree, Random Forest, KNN, SVM, Naive Bayes

## Test Results
- Phase 3: 25/25 passed (iteration_4.json)
- Phase 4: 18/20 passed (iteration_5.json)
- Phase 5: 20/20 passed (iteration_6.json)
- Phase 6: 20/20 passed (iteration_7.json)
- Phase 7: 10/10 passed (iteration_9.json)

## Key File
`/app/frontend/src/App.js` -- ~1632 lines

## Upcoming Tasks
- [ ] P1: Refactor App.js into modular components (Dashboard, Analysis, Scanner, Predictions, etc.)

## Future/Backlog
- [ ] Unsupervised: Hierarchical Clustering, DBSCAN, PCA
- [ ] Batch predictions from CSV upload
- [ ] Export reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
