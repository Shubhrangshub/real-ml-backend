# AutoML Master - Product Requirements Document

## Overview
AutoML Master is a full-stack AutoML platform (React + FastAPI + MongoDB) enabling dataset upload, model training, explainable AI (SHAP/LIME), and predictions — all running client-side in the browser.

## Architecture
- **Frontend**: React, TailwindCSS, Framer Motion, Recharts, Context API
- **Backend**: FastAPI (Python), bcrypt auth, session tokens
- **Database**: MongoDB (automl_db)

## Implemented Features

### Core ML
- [x] CSV upload & 5 sample datasets
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
- [x] Full state reset on new dataset, auto-save with toast, dataset name badge

### UI Polish
- [x] Compact header, colorful sample cards, modern Model Library

### Admin Dashboard
- [x] Analytics, User Management, Activity Log, System Controls
- [x] Conditional sidebar tab (admin-only), seeded admin

### Onboarding Guide (Feb 2026)
- [x] 9-step spotlight tour with dark overlay + violet highlight border
- [x] Auto-triggers on first login, re-triggerable via Guide button
- [x] Floating progress pill (bottom-right) tracking 5 milestones:
  1. Load a dataset, 2. Train a model, 3. Make a prediction, 4. Explore explainability, 5. Save an analysis
- [x] Pill collapses to compact "N/5 complete" badge, expands on click
- [x] "Retake the tour" button, "Dismiss permanently" option
- [x] Celebration state when all milestones complete
- [x] Tour steps: Sidebar → Analysis → Sample Datasets → Target Column → Train → Dashboard → Explainability → Leaderboard → History

## File Structure
```
src/
├── App.js, AuthPage.js, constants.js
├── context/AppContext.js
├── utils/helpers.js, mlEngine.js, datasetUtils.js
├── components/
│   ├── SmartTooltip.js
│   ├── OnboardingGuide.js (NEW - tour + progress pill)
│   └── views/
│       ├── DashboardView.js, AnalysisView.js, PredictView.js
│       ├── ExplainabilityView.js, DataExplorerView.js
│       ├── CompareModelsView.js, LeaderboardView.js
│       ├── HistoryView.js, SmallViews.js, AdminView.js
```

## Backlog
- [ ] P1: React Hook dependency issues & array key optimization
- [ ] P1: Real-time Collaborative Sessions
- [ ] P1: Model Deployment API
- [ ] P1: Automated Report Generation (PDF/HTML)
- [ ] P2: Backend server.py refactoring
- [ ] P2: Advanced hyperparameter tuning UI, What-If Analyzer
- [ ] P3: Dataset preprocessing pipeline UI
