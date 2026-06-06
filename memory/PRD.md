# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), predictions, model deployment, and What-If analysis — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & 5 large sample datasets (1000-1500 rows, 8-12 features each)
- [x] Dataset profiling, scanner, cleaning
- [x] 10+ ML algorithms, auto selection, K-fold cross-validation
- [x] Single & batch predictions, unsupervised learning, anomaly detection

### Explainability
- [x] SHAP, LIME, feature importance, business interpretation, Decision Tree viz

### Auth & UX
- [x] JWT session auth, Google OAuth, Forgot/Reset Password
- [x] Compare Models, Dark/Light mode, History, Model import/export
- [x] Secure serialization (Pickle HMAC)

### Model Leaderboard
- [x] Auto-save, sidebar tab, dashboard widget, CRUD APIs

### Dataset Context Management
- [x] Full state reset, auto-save with toast, dataset name badge

### Admin Dashboard
- [x] Analytics, User Management, Activity Log, System Controls

### Onboarding Guide
- [x] 11-step spotlight tour (including What-If & Deploy), progress pill

### Model Deployment ✅ TESTED
- [x] Deploy trained models to get a public prediction URL
- [x] Public prediction page — client-side prediction using JS ML engine
- [x] REST API endpoint for programmatic access (Python reimplementation of JS prediction logic)
- [x] Enable/disable (revoke) deployments (disabled models return 403)
- [x] Delete deployments, Prediction counter per deployment
- [x] API: POST /api/deploy, GET /api/deploy, PATCH/DELETE /api/deploy/{id}
- [x] Public: GET /api/public/model/{id}, POST /api/public/predict/{id}

### What-If Analyzer ✅ TESTED
- [x] Side-by-side baseline vs. modified scenario comparison
- [x] Auto-populated feature sliders from dataset statistics
- [x] Real-time prediction comparison with diff and % change
- [x] Uses prepareInputForPrediction + predictOne for correct data encoding

### Automated PDF Report Generation (Jun 2026) ✅ TESTED
- [x] Professional multi-page PDF report (jsPDF + jspdf-autotable)
- [x] Cover page with violet gradient, dataset info, user, and date
- [x] Table of Contents (dynamic based on available data)
- [x] Executive Summary — narrative description of the entire analysis
- [x] Dataset Overview — metric cards (rows, features, numeric/categorical, completeness) + feature table
- [x] Analysis Configuration — supervised/unsupervised, classification/regression/clustering, target variable, eval mode, train/test split details
- [x] Model Leaderboard & Ranking — table + horizontal bar chart, best model highlighted
- [x] Best Model Performance — metric cards + train vs test comparison + overfitting analysis + residual stats
- [x] Model Comparison — all trained models side by side
- [x] Feature Importance & Explainability — SHAP bar chart + table
- [x] LIME Local Interpretation — contribution table with direction coloring
- [x] Unsupervised Analysis Results — clustering algorithm comparison + cluster profiles
- [x] Anomaly Detection Results
- [x] Prediction History table
- [x] Global Leaderboard entries
- [x] Model Deployments table (status, prediction count)
- [x] Conclusions & Recommendations — data-driven narrative with actionable insights
- [x] Page footers with page numbers

### Large Sample Datasets
- [x] Loan Approval: 1,200 rows, 12 features (classification)
- [x] House Prices: 1,000 rows, 12 features (regression)
- [x] Insurance Costs: 1,100 rows, 9 features (regression)
- [x] Customer Churn: 1,500 rows, 11 features (classification)
- [x] Customer Segmentation: 1,000 rows, 10 features (unsupervised)

## File Structure
```
src/
├── App.js, AuthPage.js, constants.js
├── context/AppContext.js
├── utils/helpers.js, mlEngine.js, datasetUtils.js, reportGenerator.js
├── components/
│   ├── SmartTooltip.js, OnboardingGuide.js, PublicPredictPage.js
│   └── views/
│       ├── DashboardView.js, AnalysisView.js, PredictView.js
│       ├── ExplainabilityView.js, DataExplorerView.js
│       ├── CompareModelsView.js, LeaderboardView.js
│       ├── HistoryView.js, SmallViews.js, AdminView.js
│       ├── DeployView.js, WhatIfView.js
```

## DB Collections
- users, user_sessions, analysis_snapshots, password_reset_tokens
- leaderboard_entries, activity_log, deployed_models

## Backlog
- [ ] P1: React Hook dependency & array key optimization
- [ ] P1: Real-time Collaborative Sessions
- [ ] P2: Backend server.py refactoring (split into modules)
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: Interactive Tutorial Mode
- [ ] P3: Dataset preprocessing pipeline UI
- [ ] P3: Insecure token storage fix (localStorage → httpOnly cookies)
