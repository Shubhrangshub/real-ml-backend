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

### Bugs 1-11 Re-Fixes (June 18-22, 2026)
- **MAJOR-01** (RE-FIX x2): Fixed in TWO places — App.js `suggestedTarget` logic AND `datasetUtils.js` `generateDatasetSummary` `possibleTarget`. Root cause: `datasetUtils.js` blindly picked first categorical column (`sex`) via `break`. Now uses priority system: financial keyword match → target keyword match → categorical with fewest classes → numeric.
- **MAJOR-03** (RE-FIX): Prediction engine now uses `getBestModelIdx()` helper everywhere. Also fixed batch prediction select dropdown fallback. Fixed model auto-selection in WhatIfView (was using `models[-1]` when state was -1).

### Bugs 12-15 Re-Fixes (June 18-22, 2026)
- **MINOR-07**: Amber warning banner in Compare tab for negative R² models (code verified, cannot trigger with current data)
- **MINOR-08**: Already fixed - Tune tab "Keep original model" recommendation
- **MINOR-09**: Rare-category flagging in What-If AND Predictions dropdowns. Categories < 2% occurrence tagged with '(rare)'
- **MINOR-10**: Already fixed - Baseline entries filtered from Leaderboard

### Additional Bug Re-Fixes (June 22, 2026)
- **MINOR-05**: Added rare-category flagging in PredictView categorical dropdowns (consistent with What-If)
- **MINOR-11** (RE-FIX x2): Deploy confirmation dialog now uses React Portal (`createPortal`) to render at document.body level, fixing CSS stacking context issue from parent `motion.div` transform. Works for ALL deployments with extra warning for R² < 0.7.
- **MINOR-12** (RE-FIX): Banner now checks `preprocessConfig` active steps (scaling, outlier, excludeFeatures) in addition to cleaningLog/preprocessLog. "Apply All" now triggers the banner.
- **SHAP Chart**: Verified correct - pink (#ec4899) for positive push (right), cyan (#06b6d4) for negative pull (left)

## Key DB Collections
- `users`, `user_sessions`, `analysis_snapshots`, `leaderboard_entries`, `deployed_models`, `activity_log`

## Key API Endpoints
- POST /api/deploy, /api/snapshots, /api/admin/analytics, /api/admin/users, /api/leaderboard


### Code Quality Fixes (June 22, 2026)
**Phase 1 — Security:**
- Hardened pickle deserialization with `RestrictedUnpickler` — only allows sklearn/numpy/scipy modules
- Legacy unsigned pickle fallback now also uses restricted unpickler
- HMAC signature verification still enforced for model integrity

**Phase 2 — React Hook Stability:**
- Extracted auth hooks from `App()` into `/hooks/useAuth.js` — eliminates conditional hooks errors
- Fixed `dataProfile?.fileName` missing dependency in session-save useEffect
- Webpack now compiles with zero React hook warnings

### Bug Re-Test Final Fixes (June 25, 2026)
- **MINOR-09 (FINAL FIX)**: Prediction Result panel now shows "(rare)" tag on both the predicted value and Input Summary badges for rare categorical values. Uses same threshold (max(2, rowCount*0.02)) as the dropdown.
- **Upload Toast**: Added `toast.success("Dataset uploaded successfully — X rows, Y columns")` to `handleCsvTextChange` for non-cleaning actions. Triggers on file upload, drag-and-drop, and sample dataset loading.

### Employee Dataset Bug Fix (June 25, 2026)
- **EMP-BUG-01 (FIXED)**: Empty-field prediction returning $151,075 instead of 0. Root cause: Linear Regression intercept produced non-zero output even with all-zero inputs. Fix: Added all-fields-empty guard in `handlePredict` that returns 0 directly with a `toast.warning` before calling the ML engine. Works across all datasets (insurance, employee, etc.).

### SHAP Chart & Header Responsive Fix (June 25, 2026)
- **SHAP Summary Plot (FIXED)**: Bars weren't diverging from zero due to `stackId="a"` in Recharts causing negative bars to start from end of positive bars. Fix: Removed `stackId` so bars render as grouped bars independently from zero.
- **Header Toolbar Overlap (FIXED)**: Responsive layout fix — title truncates, dataset badge hidden below xl, button labels icon-only below lg breakpoint.

### Preprocessing UX Improvements (June 25, 2026)
- **"Proceed to Training" button**: Added at bottom of Preprocess tab — shows active step count and navigates to Analysis.
- **Preprocessing nudge**: Added in Analysis tab Model Configuration section before Start Training button. Shows green (active), amber (issues detected), or blue (no config) status with a button to navigate to Preprocess.
- **Preprocessing applied card**: Added in training results section showing what preprocessing steps were executed during training.

### Dashboard History/Leaderboard Fix (July 6, 2026)
- **Dashboard now loads history and leaderboard on initial load** — fetches snapshots and leaderboard entries when user lands on Dashboard.
- **Saved Analyses section**: Shows clickable cards of previous analysis sessions (up to 5) with dataset name, date, and algorithm. Clicking loads the snapshot.
- **Leaderboard widget visible immediately**: All-Time Leaderboard shows top models across sessions on first visit.

## Pending Tasks
### P1 - Upcoming
- Real-time Collaborative Sessions

### P2/P3 - Future/Backlog
- Export preprocessing pipeline as downloadable Python script
- Refactor App.js (~2000 lines) into smaller hooks/providers
- Refactor server.py for complexity
- Address insecure token storage (localStorage -> httpOnly cookies)
- Replace array index keys with stable identifiers in React lists
- Optimize expensive JSX computations with useMemo

## Test Credentials
- Admin: shubhrangshub@gmail.com / MyNewPass123!
