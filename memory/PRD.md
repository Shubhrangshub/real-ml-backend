# AutoML Master - Product Requirements Document

## Original Problem Statement
Full-stack AutoML application (React + FastAPI + MongoDB) that enables users to upload CSV datasets, train machine learning models, visualize results, and generate explanations.

## Architecture
- **Frontend:** React SPA (`/app/frontend/src/App.js`) with Shadcn UI, Framer Motion, Recharts, Sonner toasts
- **Backend:** FastAPI (`/app/backend/server.py`) with MongoDB
- **Database:** MongoDB (users, snapshots, models, training_history, user_sessions)
- **ML:** 100% client-side JavaScript + scikit-learn on backend for server-trained models
- **Auth:** Email/password JWT + Google OAuth

## Core Features (Implemented)
- [x] User auth (signup/login/Google OAuth)
- [x] CSV file upload and data profiling
- [x] Automated model training (Auto + individual algorithms)
- [x] Model leaderboard with ranking
- [x] Feature importance charts
- [x] Inline SVG Decision Tree visualization (flowchart)
- [x] Regression visualizations (Actual vs Predicted, Residual Analysis)
- [x] Cross-validation support
- [x] Single + Batch prediction
- [x] Model download (pickle)
- [x] Unsupervised learning (K-Means, DBSCAN)
- [x] SHAP + LIME explainability with plain-English summaries
- [x] Export system (CSV, Share link, Google Sheets) with toast feedback
- [x] History with fingerprint-based deduplication
- [x] Dark/Light mode
- [x] Viewport-aware tooltips (SmartTooltip)

## Code Review Fixes Applied
- [x] **Pickle Security (Critical)**: HMAC-signed serialization (`secure_pickle_dumps`/`secure_pickle_loads`) using SHA-256 — prevents tampering/RCE. Legacy unsigned models still load via fallback.
- [x] **datetime.utcnow() Deprecation**: Replaced 3 instances with `datetime.now(timezone.utc)`
- [x] **Signup Error Handling**: AuthPage.js shows actual errors instead of generic "Connection error"
- [x] **Empty Catch Blocks**: 5 instances in unsupervisedML.js now log `console.warn` with error details

## Code Review Items Deferred (Need Refactor Phase)
- [ ] Break App.js (~4500 lines) into modular components
- [ ] Split server.py into routes/models/logic modules
- [ ] Extract custom hooks (useModelTraining, usePredictions, useDataset)
- [ ] Fix 39 React hook dependency warnings (requires component split first)
- [ ] Replace array index keys with stable IDs (52 instances, mostly Recharts)
- [ ] Wrap 17 expensive JSX computations in useMemo
- [ ] localStorage → httpOnly cookies migration (blocked by K8s proxy architecture — `credentials: 'include'` causes fetch hangs through ingress proxy)

## Code Review False Positives
- `gdata` (line 243): Always defined via `gdata = r.json()` on line 215
- `df` (line 753): Always defined via `pd.read_csv()` on line 534
- `is None` comparisons (16 instances): Correct PEP 8 Python
- Pickle lines 104, 106: Inside `secure_pickle_loads()` AFTER HMAC verification — security gate is the signature check

## Key API Endpoints
- POST /api/auth/signup, /api/auth/login, /api/auth/logout, /api/auth/google
- GET /api/auth/me
- POST /api/train, /api/predict
- POST /api/snapshots, GET /api/snapshots, GET /api/snapshots/{id}, DELETE /api/snapshots/{id}
- POST /api/export/prepare, GET /api/export/download/{token}
- GET /api/models, DELETE /api/models/{id}, GET /api/download-model/{id}

## Test Credentials
- Email: test@automl.com / Password: Test1234!

## Backlog

### P1 - Upcoming
- [ ] Refactor App.js into modular components
- [ ] Refactor server.py into route modules
- [ ] Real-time Collaborative Sessions
- [ ] Model Deployment API

### P2 - Future
- [ ] Automated Report Generation (PDF/HTML)
- [ ] Advanced hyperparameter tuning UI
- [ ] Interactive Tutorial Mode
- [ ] "What-If" Analyzer
- [ ] Counterfactual Explanations

### P3 - Low Priority
- [ ] Dataset preprocessing pipeline UI
- [ ] Metric Comparison Radar Chart
- [ ] Performance Benchmark Mode
