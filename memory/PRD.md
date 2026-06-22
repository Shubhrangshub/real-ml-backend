# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a full-stack AutoML application with client-side ML, model training/comparison, deployment, and comprehensive data analysis features.

## Architecture
- **Frontend**: React (TailwindCSS, Framer Motion, Recharts, Shadcn/UI). Context API for state management.
- **Backend**: FastAPI (server.py)
- **Database**: MongoDB
- **ML**: Client-side via `mlEngine.js` (scikit-learn equivalent in JS)
- **PDF**: jspdf + jspdf-autotable

## Core Features (All Completed)
1. Dataset upload/sample loading with auto-profiling
2. Model training (8 algorithms: Random Forest, Gradient Boosting, KNN, SVM, Decision Tree, Naive Bayes, Logistic/Linear Regression)
3. Model Leaderboard with auto-save and ranking
4. SHAP/LIME Explainability with beeswarm plots
5. Model Comparison (radar charts, metric tables)
6. Model Deployment via public REST API
7. What-If Analyzer for predictions
8. Automated PDF Report generation
9. Data Preprocessing Pipeline UI with Smart Recommendations
10. Advanced Hyperparameter Tuning (all 8 algorithms)
11. Admin Dashboard with analytics
12. 13-step Onboarding Dialog Guide
13. Anomaly Detection with narration
14. History with direct PDF download

## QA Bug Fixes (All Complete - June 2026)
### Bugs 1-11 (Fixed in previous session)
- MAJOR-01 through MAJOR-05, MINOR-01 through MINOR-06
- Clamped negative predictions, deduplicated models, auto-selected best models, bad R2 warnings, etc.

### Bugs 1-11 Re-Fixes (June 18, 2026)
- **MAJOR-01** (RE-FIX): Target suggestion now correctly picks 'charges' over 'sex' for insurance datasets. Root cause: high-uniqueness penalty (-20) was incorrectly applied to numeric financial columns. Fix: numeric columns matching financial/target keywords skip the penalty and get a +8 continuous bonus.
- **MAJOR-03** (RE-FIX): Prediction engine now uses the SAME model shown in dropdown. Root cause: `handlePredict` fell back to `models.length - 1` (last trained) instead of best model. Fix: created `getBestModelIdx()` helper used by handlePredict, handleBatchPredict, and smartRowSuggestions.

### Bugs 12-15 (Fixed June 18, 2026)
- **MINOR-07**: Added prominent amber warning banner in Compare tab for models with negative R2 (`data-testid="negative-r2-warning"`)
- **MINOR-08**: Already fixed - Tune tab shows "Keep original model" recommendation
- **MINOR-09**: Added rare-category flagging in What-If dropdown with '(rare)' tag for categories < 2% occurrence
- **MINOR-10**: Already fixed - Baseline entries filtered from Leaderboard

### Additional Bug Fixes
- **MINOR-11**: Deploy tab confirm dialog for negative R2 models
- **MINOR-12**: Data Explorer shows preprocessing notification near correlation heatmap (`data-testid="preprocess-correlation-notice"`)
- **SHAP Chart**: Verified correct - pink (#ec4899) for positive push (right), cyan (#06b6d4) for negative pull (left)

## Key DB Collections
- `users`, `user_sessions`, `analysis_snapshots`, `leaderboard_entries`, `deployed_models`, `activity_log`

## Key API Endpoints
- POST /api/deploy, /api/snapshots, /api/admin/analytics, /api/admin/users, /api/leaderboard

## Pending Tasks
### P1 - Upcoming
- Real-time Collaborative Sessions

### P2/P3 - Future/Backlog
- Export preprocessing pipeline as downloadable Python script
- Refactor App.js (~2000 lines) into smaller hooks/providers
- Refactor server.py for complexity
- Address insecure token storage (localStorage -> httpOnly cookies)

## Test Credentials
- Admin: shubhrangshub@gmail.com / MyNewPass123!
