# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), predictions, model deployment, What-If analysis, preprocessing pipelines, and hyperparameter tuning — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & 5 large sample datasets (1000-1500 rows)
- [x] Dataset profiling, scanner, cleaning
- [x] 10+ ML algorithms, auto selection, K-fold cross-validation
- [x] Single & batch predictions, unsupervised learning, anomaly detection

### Explainability
- [x] SHAP, LIME, feature importance, business interpretation, Decision Tree viz

### Auth & UX
- [x] JWT session auth, Google OAuth, Forgot/Reset Password
- [x] Compare Models, Dark/Light mode, History, Model import/export

### Model Leaderboard, Admin Dashboard, Onboarding Guide
- [x] All fully implemented and tested

### Model Deployment ✅ TESTED
- [x] Deploy trained models to get a public prediction URL
- [x] Public prediction page with client-side prediction
- [x] REST API endpoint with Python reimplementation of JS prediction logic
- [x] Enable/disable/delete deployments

### What-If Analyzer ✅ TESTED
- [x] Side-by-side baseline vs. modified scenario comparison

### Automated PDF Report ✅ TESTED
- [x] Multi-page PDF with cover, TOC, executive summary, dataset overview
- [x] Model leaderboard, best model metrics, feature importance charts
- [x] SHAP/LIME tables, preprocessing config, tuning results, deployments
- [x] Conclusions & recommendations

### Data Preprocessing Pipeline (Jun 2026) ✅ TESTED
- [x] **Missing Values**: auto (median/mode), mean, median, mode, zero, drop rows, none
- [x] **Feature Scaling**: none, standardize (z-score), min-max normalize
- [x] **Outlier Treatment**: none, clip (winsorize), remove — configurable IQR threshold
- [x] **Feature Selection**: toggle individual features, select all/numeric only
- [x] Preprocessing config stored in state, applied during training pipeline
- [x] Scale params stored in modelData for prediction compatibility
- [x] Preprocessing log displayed after training
- [x] Integrated with PDF report (preprocessing pipeline section)
- [x] Integrated with onboarding guide (new tour step)
- [x] Backend public predict handles scaled models correctly

### Advanced Hyperparameter Tuning (Jun 2026) ✅ TESTED
- [x] **Tunable algorithms**: Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Ridge
- [x] **Search strategies**: Random Search (20 trials), Grid Search (exhaustive)
- [x] **Per-algorithm parameters**: maxDepth, minSamples, nTrees, learningRate, k, C, lambda, epochs
- [x] **Results dashboard**: Original vs Tuned score comparison, improvement metric
- [x] **Top 10 Trials table**: ranked by score with parameter values
- [x] **Apply Tuned Model**: adds optimized model to session (only when improvement > 0)
- [x] Progress bar during tuning
- [x] Integrated with PDF report (tuning results section)
- [x] Integrated with onboarding guide (new tour step)

## File Structure
```
src/
├── App.js, AuthPage.js, constants.js
├── context/AppContext.js
├── utils/helpers.js, mlEngine.js, datasetUtils.js, reportGenerator.js, preprocessUtils.js
├── components/
│   ├── SmartTooltip.js, OnboardingGuide.js, PublicPredictPage.js
│   └── views/
│       ├── DashboardView.js, AnalysisView.js, PredictView.js
│       ├── ExplainabilityView.js, DataExplorerView.js
│       ├── CompareModelsView.js, LeaderboardView.js
│       ├── HistoryView.js, SmallViews.js, AdminView.js
│       ├── DeployView.js, WhatIfView.js
│       ├── PreprocessView.js, TuneView.js
```

## DB Collections
- users, user_sessions, analysis_snapshots, password_reset_tokens
- leaderboard_entries, activity_log, deployed_models

## Backlog
- [ ] P1: React Hook dependency & array key optimization
- [ ] P1: Real-time Collaborative Sessions
- [ ] P2: Backend server.py refactoring (split into modules)
- [ ] P2: Interactive Tutorial Mode expansion
- [ ] P3: Insecure token storage fix (localStorage → httpOnly cookies)
