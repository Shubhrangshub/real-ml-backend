# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), and predictions — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & sample datasets (Loan Approval, House Prices, Insurance Costs, Customer Churn, Customer Segmentation)
- [x] Dataset profiling & auto-summary
- [x] Dataset scanner (quality score, missing values, outliers, duplicates)
- [x] Data cleaning (fill missing, remove duplicates, remove outliers, drop constants, normalize)
- [x] Auto algorithm selection
- [x] 10+ ML algorithms (Linear/Ridge/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes, Baseline)
- [x] K-fold cross-validation
- [x] Single & batch predictions
- [x] Unsupervised learning (K-Means, Hierarchical, DBSCAN, GMM)
- [x] Anomaly detection (Z-Score, IQR)
- [x] Clustering (K-Means)

### Explainability
- [x] SHAP (Global, Local, Beeswarm, Dependence plots)
- [x] LIME explanations
- [x] Feature importance
- [x] Business interpretation text
- [x] Decision Tree inline visualization

### Auth & UX
- [x] JWT session-based auth (localStorage tokens)
- [x] Google OAuth integration
- [x] Forgot Password / Reset Password flow
- [x] Compare Models view (radar chart, metric table, feature importance, confusion matrix, winner recommendation)
- [x] Dark/Light mode
- [x] Analysis history (save/load/share/delete snapshots)
- [x] Model import/export (download/upload JSON)
- [x] Secure model serialization (Pickle HMAC)
- [x] Meaningful auth error messages
- [x] History deduplication

## Modular Architecture (Refactored Feb 2026)
App.js reduced from 4598 → 1692 lines (63% reduction)

### File Structure
```
src/
├── App.js                          (1692 lines - state + routing)
├── AuthPage.js                     (auth + forgot password)
├── constants.js                    (ALGO_NAMES, GUIDE_STEPS, etc.)
├── context/
│   └── AppContext.js               (React Context for shared state)
├── utils/
│   ├── helpers.js                  (getScoreColor, interpretMetric, etc.)
│   ├── mlEngine.js                 (all ML algorithms, prediction, metrics)
│   └── datasetUtils.js             (CSV parsing, profiling, scanning, cleaning)
├── components/
│   ├── SmartTooltip.js             (SmartTooltip, MetricTip, HelpTip)
│   └── views/
│       ├── DashboardView.js        (168 lines)
│       ├── AnalysisView.js         (492 lines)
│       ├── PredictView.js          (356 lines)
│       ├── ExplainabilityView.js   (903 lines)
│       ├── DataExplorerView.js     (111 lines)
│       ├── CompareModelsView.js     (Compare Models with radar/metrics)
│       ├── HistoryView.js          (70 lines)
│       └── SmallViews.js           (Clusters/Anomalies/Models)
```

## API Endpoints
- POST /api/auth/signup, /api/auth/login, /api/auth/logout, /api/auth/google
- POST /api/auth/forgot-password, /api/auth/reset-password
- GET /api/auth/me
- POST /api/train, /api/predict
- GET /api/snapshots, POST /api/snapshots, DELETE /api/snapshots/{id}
- POST /api/snapshots/{id}/share
- GET /api/download-model/{model_id}

## DB Schema
- users: {email, password_hash, name, picture, auth_provider}
- user_sessions: {session_token, user_id, expires_at}
- analysis_snapshots: {snapshot_id, user_id, data, fingerprint, createdAt}
- password_reset_tokens: {email, token, expires_at, used, created_at}

## Test Credentials
- Email: test@automl.com / Password: Test1234!
- See /app/memory/test_credentials.md for full list

## Known Issues
- Token stored in localStorage (httpOnly cookies cause Kubernetes proxy fetch hangs)
- Unused variable warnings in extracted view components (cosmetic)

## Backlog
- [ ] P1: Real-time Collaborative Sessions
- [ ] P1: Model Deployment API
- [ ] P1: Automated Report Generation (PDF/HTML)
- [ ] P2: Backend server.py refactoring (high complexity functions)
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: "What-If" Analyzer
- [ ] P2: Interactive Tutorial Mode
- [ ] P2: Metric Comparison Radar Chart
- [ ] P2: Performance Benchmark Mode
- [ ] P3: Dataset preprocessing pipeline UI
