# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS - 11 supervised + 4 clustering + 2 dim reduction + 2 anomaly detection algorithms
- **Storage**: React state + localStorage persistence
- **Key Files**: `/app/frontend/src/App.js` (UI + supervised), `/app/frontend/src/unsupervisedML.js` (unsupervised engine)

## Core Features Implemented

### Phase 1-3 - Foundation (Completed)
- Universal Dashboard, Supervised Learning Engine, auto task detection, 80/20 split, evaluation metrics

### Phase 4 - Advanced Algorithms & Visual Explainability (Completed)
- 7 new supervised algorithms, F1 Score, Model Comparison Chart, Confusion Matrix Heatmap

### Phase 5 - K-Fold Cross Validation (Completed)
- 5-fold CV for all supervised algorithms, CV Performance Chart, leaderboard

### Phase 6 - Dashboard Overhaul + Dataset Scanner (Completed)
- Live stats, Dataset Scanner with 12 quality checks, one-click cleaning, training gate

### Phase 7 - UI/UX & Explainability Overhaul (Completed)
- Color-coded MetricCards with tooltips, plain-English summaries, dynamic prediction form, improved chart styling

### Phase 8 - Full Unsupervised Learning System (Completed - Mar 2026)
- [x] **Clustering Algorithms**: K-Means (enhanced), Hierarchical (Agglomerative), DBSCAN, Gaussian Mixture Models
- [x] **Dimensionality Reduction**: PCA (2D projection with variance explained), t-SNE (browser-optimized)
- [x] **Anomaly Detection**: Isolation Forest, Local Outlier Factor (LOF)
- [x] **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, Inertia
- [x] **Automatic Optimal K Detection**: Elbow Method + Silhouette analysis (K=2 to 10)
- [x] **Data Preprocessing Pipeline**: Missing value imputation, z-score standardization
- [x] **Algorithm Leaderboard**: All 4 clustering algorithms ranked by Silhouette Score with full metrics
- [x] **7 Chart Types**: PCA Scatter, t-SNE, Cluster Distribution, Elbow Method, Silhouette by K, Anomaly Detection, Cluster Profiles
- [x] **Cluster Interpretation System**: Per-cluster feature averages, key distinguishing features, auto-generated English descriptions
- [x] **Cluster Prediction Form**: Dynamic form to assign new data points to clusters with nearest centroid + interpretation
- [x] **Terminology Guide**: 15 ML concept cards (K-Means, DBSCAN, PCA, Silhouette Score, etc.)
- [x] **Preprocessing Summary Panel**: Dataset size, features, missing values, scaling method
- [x] **Auto-detect Mode**: Target = "__none__" triggers unsupervised pipeline; target selected = supervised pipeline

## All Algorithms
### Supervised (11)
- Regression: Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting
- Classification: Logistic, Decision Tree, Random Forest, KNN, SVM, Naive Bayes

### Unsupervised (8)
- Clustering: K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models
- Dimensionality Reduction: PCA, t-SNE
- Anomaly Detection: Isolation Forest, Local Outlier Factor

## Test Results
- Phase 7 UI/UX: 10/10 passed (iteration_9.json)
- Phase 8 Unsupervised: 15/15 passed (iteration_10.json)

## Upcoming Tasks
- [ ] P1: Refactor App.js into modular components

## Future/Backlog
- [ ] Batch predictions from CSV upload
- [ ] Export reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
- [ ] Model export/import
