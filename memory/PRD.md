# AutoML Master - Product Requirements Document

## Original Problem Statement
Build a 100% client-side Universal AI Dashboard (AutoML Master) in React. All ML analysis runs directly in the browser.

## Architecture
- **Frontend**: React + Tailwind CSS + Shadcn UI + Recharts + Framer Motion
- **ML Engine**: Custom client-side JS (supervised in App.js, unsupervised in unsupervisedML.js)
- **XAI Engine**: explainableAI.js (SHAP approximation + LIME)
- **Backend**: FastAPI + MongoDB (snapshots API for history & sharing)

## Core Files
- `/app/frontend/src/App.js` — Main UI + supervised ML (~3600 lines)
- `/app/frontend/src/unsupervisedML.js` — Unsupervised ML engine
- `/app/frontend/src/explainableAI.js` — SHAP & LIME computation engine
- `/app/backend/server.py` — FastAPI with snapshot CRUD endpoints

## Completed Features

### Phase 1-10 (Previous)
- [x] Supervised (11 algos) + Unsupervised learning, Dataset Scanner, K-Fold CV
- [x] Batch Predictions, Model Export/Import, PDF Reports, Data Explorer, Dark Mode
- [x] Metric Hover Tooltips, Explainable AI (SHAP + LIME + Cluster Explanations)

### Phase 11 (Feb 2026)
- [x] Performance Optimization

### Phase 12 (Feb 2026)
- [x] Session Persistence (localStorage + Clear Session)

### Phase 13 (Feb 2026)
- [x] Enhanced XAI Dashboard (7 new charts, vibrant colors, descriptions)

### Phase 14 (Feb 2026)
- [x] Smart Guided Help System (HelpTip, Guide Panel, Target Suggestion, Smart Row Suggestions)

### Phase 15 (Feb 2026)
- [x] History & Sharing System (MongoDB snapshots, share URLs, view-only mode, CSV/JSON/Sheets export)

### Phase 16 (Feb 2026)
- [x] **Smart Metric Interpretation System**:
  - `interpretMetric(key, value)` function — context-sensitive plain-English explanations for all metrics based on actual values
  - **R²**: Special handling for <0 ("worse than predicting average"), <0.3, <0.5, <0.7, <0.9, 0.9+
  - **Accuracy**: <50% (poor), <70% (fair + imbalanced dataset warning), <90% (good), 90%+ (excellent)
  - **F1 Score**: <0.5 (poor), <0.7 (moderate), <0.9 (good), 0.9+ (excellent) with precision/recall balance explanation
  - **Precision/Recall**: 4-tier interpretation with false alarm / missed detection context
  - **RMSE/MAE**: Actual error value mentioned, "lower is better" guidance
  - **Silhouette**: Negative (wrong clusters), <0.25 (overlapping), <0.5 (moderate), <0.75 (good), 0.75+ (excellent)
  - **Davies-Bouldin, Calinski-Harabasz, CV Score**: Full threshold-based interpretations
  - **Color coding**: Emerald (Excellent), Sky (Good), Amber (Fair), Red (Poor/Needs Work) + CheckCircle2/AlertCircle icons
  - Interpretation shown BELOW each metric value in MetricCard AND in hover tooltip (MetricTip)
  - MetricTip now accepts optional `value` prop for context-sensitive tooltip content

### Phase 17 (Feb 2026)
- [x] **Simplified Export System**:
  - Removed PDF export (jsPDF/jspdf-autotable) and JSON export
  - Kept: Share Analysis (view-only dashboard via snapshot URL), Export to Google Sheets (CSV download), Download CSV (comprehensive with SHAP, LIME, predictions, metrics)
  - View-only banner: "This is a shared analysis (view-only). Request access to edit."
  - UI: Three buttons — "Share Analysis", "Export to Google Sheets", "Download CSV"
- [x] **Safe Clipboard Share Fix**:
  - `safeCopyToClipboard` utility: tries navigator.clipboard, falls back to textarea+execCommand
  - Share toast now shows selectable input field (not truncated code tag) for manual copy
  - Status feedback: "Link copied to clipboard" (success) / "Copy not supported here. Please copy manually." (fallback)
  - All clipboard usage sites updated (share button, copy button, history share buttons)
  - Fully wrapped in try-catch — no crashes when clipboard API is blocked (iframe/preview safe)
- [x] **Dataset Summary Generator**:
  - Auto-generates plain-English summary after dataset upload
  - Domain detection via keyword matching (finance, health, insurance, real estate, sales, education, HR, etc.)
  - 5-line description: what data is about, data types, likely objective, key variables, data characteristics
  - Focus line: "This dataset mainly focuses on [domain], with key variables like [X, Y, Z]."
  - Key Variables section with type badges, Suggested Target with task type
  - Rendered between Dataset Profile and Dataset Scanner cards
- [x] **XAI Model Recommendation**:
  - Auto-detects and recommends the best model for SHAP/LIME analysis based on score (accuracy for classification, R² for regression)
  - Green recommendation banner above model selector: "Recommended: [model] — best [metric]"
  - Dropdown shows all models with scores and ⭐ [Recommended] tag on the best one
  - Best model auto-selected by default; users can still pick any other model

## Testing Status
- Iteration 21: 12/12 (100% — Dataset Summary)
- Iteration 20: 10/10 (100% — Safe Clipboard Share)
- Iteration 19: 13/13 (100% — Export System Simplification)
- Iteration 18: 13/13 (100% — Metric Interpretation)
- Iteration 17: 29/29 (100% — History & Sharing)
- Iteration 16: 24/24 (100% — Help System)
- Iteration 15: 14/14 (100% — XAI)

## Backlog
- [ ] P0: Refactor App.js into modular components (~3900+ lines)
- [ ] P1: "What-If" Analyzer
- [ ] P1: Auto-save After Training
- [ ] P1: Counterfactual Explanations
- [ ] P2: Real-time model comparison dashboard
- [ ] P2: Advanced hyperparameter tuning UI
- [ ] P2: Interactive Tutorial Mode
- [ ] P2: Metric Comparison Radar Chart
- [ ] P2: Performance Benchmark Mode
- [ ] P3: Dataset preprocessing pipeline UI
