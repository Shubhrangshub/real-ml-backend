# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard in React. All ML analysis runs directly in the browser using JavaScript libraries. No Python backend or MongoDB.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Client-side JS (Normal equation linear regression, Logistic regression, K-Means, Anomaly detection)
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
- [x] **Universal Analysis Engine**: Auto-profiles datasets on upload, detects column types (numeric/categorical), suggests ML task (regression/classification/clustering), shows data profile card with stats table
- [x] **K-Means Clustering Dashboard**: New "Clusters" tab with k slider (2-10), scatter visualization with color-coded clusters, cluster size distribution chart, cluster centers table. Uses ml-kmeans library with standardized features
- [x] **Anomaly Detection Module**: New "Anomalies" tab with Z-Score and IQR methods, threshold slider, scatter plot (normal vs anomaly), per-column anomaly chart, anomalous rows table with highlighted values
- [x] **localStorage Persistence**: Models saved to localStorage on train, restored on page load. Fixed race condition with hasLoadedFromStorage flag
- [x] Renamed "Train Models" to "Analysis" tab
- [x] Added "No target (use Clustering)" option in target dropdown
- [x] Mini data upload component in Clusters/Anomalies tabs for quick sample loading

## Sidebar Navigation (6 tabs)
1. Dashboard - ML operations overview
2. Analysis - Upload data, auto-profile, train models
3. Predictions - Generate predictions from trained models
4. Clusters - K-Means clustering dashboard
5. Anomalies - Outlier detection with Z-score/IQR
6. Model Library - Manage trained models (view, download, delete)

## Test Results
- Phase 1: 21/21 tests passed (iteration_2.json)
- Phase 2: 21/21 tests passed (iteration_3.json)

## Tech Stack
- React 18, Tailwind CSS, Shadcn UI, Recharts, Framer Motion
- simple-statistics, ml-kmeans, ml-regression, danfojs (installed)

## Key File
- `/app/frontend/src/App.js` - All application logic (~1130 lines)

## Upcoming / Backlog
- [ ] More ML algorithms (Decision Tree, Random Forest, Gradient Boosting)
- [ ] Text processing (TF-IDF) for text-heavy datasets
- [ ] Batch predictions from CSV upload
- [ ] Export training/analysis reports as PDF
- [ ] Dark mode toggle
- [ ] Data visualization tab (histograms, correlation matrix)
- [ ] Model comparison dashboard
