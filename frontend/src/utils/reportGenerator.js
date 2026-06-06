/**
 * AutoML Master — Automated PDF Report Generator
 * Generates a comprehensive, professional PDF report with narration and visuals.
 */
import { jsPDF } from 'jspdf';
import { autoTable, applyPlugin } from 'jspdf-autotable';

// Apply the autoTable plugin to jsPDF so doc.autoTable() works
applyPlugin(jsPDF);

const ALGO_NAMES = {
  linear_regression: 'Linear Regression', ridge_regression: 'Ridge Regression',
  logistic_regression: 'Logistic Regression', decision_tree: 'Decision Tree',
  random_forest: 'Random Forest', gradient_boosting: 'Gradient Boosting',
  knn: 'K-Nearest Neighbors', svm: 'Support Vector Machine',
  naive_bayes: 'Naive Bayes', baseline: 'Baseline', kmeans: 'K-Means',
  dbscan: 'DBSCAN', hierarchical: 'Hierarchical Clustering',
};

// Color palette
const C = {
  primary: [109, 40, 217],     // violet-600
  secondary: [192, 38, 211],   // fuchsia-600
  accent: [16, 185, 129],      // emerald-500
  dark: [24, 24, 27],          // zinc-900
  muted: [113, 113, 122],      // zinc-500
  light: [244, 244, 245],      // zinc-100
  white: [255, 255, 255],
  red: [239, 68, 68],
  amber: [245, 158, 11],
  blue: [59, 130, 246],
};

function fmt(v, d = 4) {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(d);
  return String(v);
}

function pct(v) { return v != null ? (v * 100).toFixed(1) + '%' : '—'; }

// Draw a horizontal bar chart programmatically
function drawBarChart(doc, data, x, y, w, h, title, color = C.primary) {
  if (!data || data.length === 0) return y;
  const maxVal = Math.max(...data.map(d => Math.abs(d.value)), 0.001);
  const barH = Math.min(14, (h - 20) / data.length);
  const labelW = Math.min(w * 0.35, 90);
  const chartW = w - labelW - 35;

  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...C.dark);
  doc.text(title, x, y);
  y += 8;

  data.forEach((d, i) => {
    const cy = y + i * barH;
    // Label
    doc.setFontSize(7);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...C.muted);
    const label = d.label.length > 18 ? d.label.substring(0, 16) + '…' : d.label;
    doc.text(label, x, cy + barH * 0.6);
    // Bar
    const barWidth = Math.max(1, (Math.abs(d.value) / maxVal) * chartW);
    const alpha = 0.7 + 0.3 * (1 - i / data.length);
    doc.setFillColor(color[0], color[1], color[2]);
    doc.setGState(new doc.GState({ opacity: alpha }));
    doc.roundedRect(x + labelW, cy, barWidth, barH - 2, 1.5, 1.5, 'F');
    doc.setGState(new doc.GState({ opacity: 1 }));
    // Value
    doc.setFontSize(6.5);
    doc.setTextColor(...C.dark);
    doc.text(fmt(d.value), x + labelW + barWidth + 3, cy + barH * 0.55);
  });

  return y + data.length * barH + 6;
}

// Draw a metric card
function drawMetricCard(doc, x, y, w, h, label, value, color = C.primary) {
  doc.setFillColor(250, 250, 252);
  doc.roundedRect(x, y, w, h, 3, 3, 'F');
  doc.setDrawColor(230, 230, 235);
  doc.roundedRect(x, y, w, h, 3, 3, 'S');
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...color);
  doc.text(String(value), x + w / 2, y + h * 0.5, { align: 'center' });
  doc.setFontSize(7);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...C.muted);
  doc.text(label, x + w / 2, y + h * 0.8, { align: 'center' });
}

// Ensure enough space, add new page if needed
function ensureSpace(doc, y, needed, margin = 20) {
  if (y + needed > doc.internal.pageSize.getHeight() - margin) {
    doc.addPage();
    return 25;
  }
  return y;
}

// Section header
function sectionHeader(doc, y, title, number) {
  y = ensureSpace(doc, y, 20);
  const pw = doc.internal.pageSize.getWidth();
  doc.setFillColor(...C.primary);
  doc.roundedRect(14, y, pw - 28, 0.7, 0, 0, 'F');
  y += 6;
  doc.setFontSize(13);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...C.primary);
  doc.text(`${number}. ${title}`, 16, y + 5);
  return y + 12;
}

// Narrative text block
function narrate(doc, y, text, maxWidth) {
  y = ensureSpace(doc, y, 12);
  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...C.dark);
  const lines = doc.splitTextToSize(text, maxWidth || (doc.internal.pageSize.getWidth() - 32));
  doc.text(lines, 16, y);
  return y + lines.length * 4.2 + 3;
}

// Footer
function addFooter(doc) {
  const totalPages = doc.internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    const pw = doc.internal.pageSize.getWidth();
    const ph = doc.internal.pageSize.getHeight();
    doc.setFontSize(6.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...C.muted);
    doc.text(`AutoML Master Report — Page ${i} of ${totalPages}`, pw / 2, ph - 8, { align: 'center' });
    doc.text(new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }), pw - 16, ph - 8, { align: 'right' });
  }
}

/**
 * Main export function — generates and downloads the PDF report.
 */
export async function generateReport({
  dataProfile, trainingResult, models, targetColumn, evalMode,
  shapGlobal, limeResult, predictionHistory, unsupervisedResult,
  clusterResult, anomalyResult, leaderboardEntries, deployments,
  authUser,
}) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pw = doc.internal.pageSize.getWidth(); // ~210
  const mw = pw - 32; // usable text width
  const now = new Date();
  const datasetName = dataProfile?.fileName || 'Uploaded Dataset';
  const hasSupervisedResult = !!trainingResult;
  const hasUnsupervisedResult = !!unsupervisedResult;
  const problemType = trainingResult?.problemType || (hasUnsupervisedResult ? 'clustering' : 'N/A');
  const analysisType = hasSupervisedResult ? 'Supervised' : hasUnsupervisedResult ? 'Unsupervised' : 'N/A';
  let sectionNum = 0;

  // ============================== COVER PAGE ==============================
  // Background gradient strip
  doc.setFillColor(...C.primary);
  doc.rect(0, 0, pw, 85, 'F');
  doc.setFillColor(...C.secondary);
  doc.rect(0, 85, pw, 4, 'F');

  // Title
  doc.setFontSize(32);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...C.white);
  doc.text('AutoML Analysis', pw / 2, 35, { align: 'center' });
  doc.setFontSize(14);
  doc.setFont('helvetica', 'normal');
  doc.text('Comprehensive Report', pw / 2, 48, { align: 'center' });

  // Dataset name badge
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  const dsW = doc.getTextWidth(datasetName) + 16;
  doc.setFillColor(255, 255, 255);
  doc.setGState(new doc.GState({ opacity: 0.2 }));
  doc.roundedRect((pw - dsW) / 2, 58, dsW, 12, 4, 4, 'F');
  doc.setGState(new doc.GState({ opacity: 1 }));
  doc.setTextColor(...C.white);
  doc.text(datasetName, pw / 2, 66, { align: 'center' });

  // Info cards below banner
  let cy = 100;
  const cardData = [
    { label: 'Analysis Type', value: analysisType },
    { label: 'Problem Type', value: problemType.charAt(0).toUpperCase() + problemType.slice(1) },
    { label: 'Target Variable', value: targetColumn || 'N/A (Unsupervised)' },
    { label: 'Generated', value: now.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) },
  ];
  const cardW = (mw - 9) / 4;
  cardData.forEach((cd, i) => {
    const cx = 16 + i * (cardW + 3);
    doc.setFillColor(250, 250, 252);
    doc.roundedRect(cx, cy, cardW, 22, 3, 3, 'F');
    doc.setDrawColor(230, 230, 235);
    doc.roundedRect(cx, cy, cardW, 22, 3, 3, 'S');
    doc.setFontSize(7);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...C.muted);
    doc.text(cd.label, cx + cardW / 2, cy + 7, { align: 'center' });
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...C.dark);
    const valText = cd.value.length > 16 ? cd.value.substring(0, 14) + '…' : cd.value;
    doc.text(valText, cx + cardW / 2, cy + 16, { align: 'center' });
  });
  cy += 35;

  // User & date footer on cover
  if (authUser) {
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...C.muted);
    doc.text(`Generated by: ${authUser.name || authUser.email || 'Unknown User'}`, 16, cy);
    doc.text(`Date: ${now.toLocaleString()}`, 16, cy + 5);
  }

  // Table of contents
  cy += 18;
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(...C.dark);
  doc.text('Table of Contents', 16, cy);
  cy += 8;
  const tocItems = ['Executive Summary'];
  if (dataProfile) tocItems.push('Dataset Overview');
  tocItems.push('Analysis Configuration');
  if (hasSupervisedResult) tocItems.push('Model Leaderboard & Ranking', 'Best Model Performance');
  if (models?.length > 1) tocItems.push('Model Comparison');
  if (shapGlobal || models?.some(m => m.featureImportance?.length > 0)) tocItems.push('Feature Importance & Explainability');
  if (limeResult) tocItems.push('LIME Local Interpretation');
  if (hasUnsupervisedResult) tocItems.push('Unsupervised Analysis Results');
  if (clusterResult && !hasUnsupervisedResult) tocItems.push('Clustering Results');
  if (anomalyResult) tocItems.push('Anomaly Detection Results');
  if (predictionHistory?.length > 0) tocItems.push('Prediction History');
  if (leaderboardEntries?.length > 0) tocItems.push('Global Leaderboard');
  if (deployments?.length > 0) tocItems.push('Model Deployments');
  tocItems.push('Conclusions & Recommendations');

  tocItems.forEach((item, i) => {
    doc.setFontSize(8);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...C.muted);
    doc.text(`${i + 1}.`, 18, cy);
    doc.setTextColor(...C.dark);
    doc.text(item, 26, cy);
    cy += 5.5;
  });

  // ============================== PAGE 2+: CONTENT ==============================
  doc.addPage();
  let y = 20;

  // ---- 1. Executive Summary ----
  sectionNum++;
  y = sectionHeader(doc, y, 'Executive Summary', sectionNum);
  {
    const parts = [];
    parts.push(`This report presents a comprehensive analysis of the "${datasetName}" dataset`);
    if (dataProfile) parts.push(` containing ${dataProfile.rowCount?.toLocaleString() || '?'} records and ${dataProfile.columns?.length || dataProfile.headers?.length || '?'} features.`);
    else parts.push('.');

    if (hasSupervisedResult) {
      const best = trainingResult.bestModel;
      const algoName = ALGO_NAMES[best?.algorithm] || best?.algorithm || 'Unknown';
      parts.push(`\n\nA supervised ${problemType} analysis was conducted targeting the "${targetColumn}" variable. `);
      parts.push(`${trainingResult.leaderboard?.length || 1} algorithm(s) were evaluated using ${evalMode === 'cv' ? 'K-Fold Cross-Validation' : 'train/test split (80/20)'}. `);
      parts.push(`The best performing model is ${algoName}`);
      const bestScore = best?.testMetrics || best?.metrics;
      if (bestScore) {
        const mainMetric = problemType === 'classification'
          ? bestScore.accuracy != null ? `accuracy of ${pct(bestScore.accuracy)}` : `F1-score of ${fmt(bestScore.f1, 4)}`
          : `R² score of ${fmt(bestScore.r2, 4)}`;
        parts.push(` achieving a ${mainMetric}.`);
      } else parts.push('.');
    }
    if (hasUnsupervisedResult) {
      const best = unsupervisedResult.bestAlgorithm;
      parts.push(`\n\nAn unsupervised clustering analysis was performed. The best algorithm identified is ${best?.name || 'K-Means'} `);
      parts.push(`with ${best?.k || unsupervisedResult.preprocessing?.featureNames?.length || '?'} clusters and a silhouette score of ${fmt(best?.silhouette, 3)}.`);
    }
    if (anomalyResult) {
      parts.push(`\n\nAnomaly detection identified ${anomalyResult.anomalyCount || '?'} anomalous records out of ${anomalyResult.totalRows || dataProfile?.rowCount || '?'} total rows.`);
    }
    y = narrate(doc, y, parts.join(''), mw);
  }

  // ---- 2. Dataset Overview ----
  if (dataProfile) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Dataset Overview', sectionNum);

    const numCols = dataProfile.numericColumns?.length || 0;
    const catCols = (dataProfile.headers?.length || 0) - numCols;
    const totalMissing = Object.values(dataProfile.missingCounts || {}).reduce((a, b) => a + b, 0);
    const totalCells = (dataProfile.rowCount || 0) * (dataProfile.headers?.length || 1);
    const completeness = totalCells > 0 ? ((1 - totalMissing / totalCells) * 100).toFixed(1) : '100.0';

    // Metric cards
    y = ensureSpace(doc, y, 28);
    const metricsData = [
      { label: 'Rows', value: (dataProfile.rowCount || 0).toLocaleString(), color: C.primary },
      { label: 'Features', value: String(dataProfile.headers?.length || 0), color: C.blue },
      { label: 'Numeric', value: String(numCols), color: C.accent },
      { label: 'Categorical', value: String(catCols), color: C.secondary },
      { label: 'Completeness', value: completeness + '%', color: totalMissing > 0 ? C.amber : C.accent },
    ];
    const mcW = (mw - 12) / metricsData.length;
    metricsData.forEach((m, i) => {
      drawMetricCard(doc, 16 + i * (mcW + 3), y, mcW, 22, m.label, m.value, m.color);
    });
    y += 30;

    y = narrate(doc, y,
      `The dataset contains ${numCols} numeric and ${catCols} categorical feature(s). ` +
      `Data completeness is ${completeness}%` +
      (totalMissing > 0 ? ` with ${totalMissing.toLocaleString()} missing value(s) across the dataset.` : ', with no missing values.'),
      mw
    );

    // Feature types table
    if (dataProfile.headers?.length > 0) {
      y = ensureSpace(doc, y, 30);
      const featureRows = dataProfile.headers.slice(0, 30).map(h => {
        const isNum = dataProfile.numericColumns?.includes(h);
        const missing = dataProfile.missingCounts?.[h] || 0;
        const isTarget = h === targetColumn;
        return [h + (isTarget ? ' (TARGET)' : ''), isNum ? 'Numeric' : 'Categorical', String(missing), missing > 0 ? ((missing / (dataProfile.rowCount || 1)) * 100).toFixed(1) + '%' : '0%'];
      });
      doc.autoTable({
        startY: y, margin: { left: 16, right: 16 },
        head: [['Feature', 'Type', 'Missing', 'Missing %']],
        body: featureRows,
        theme: 'grid',
        styles: { fontSize: 7, cellPadding: 2 },
        headStyles: { fillColor: C.primary, textColor: C.white, fontStyle: 'bold', fontSize: 7 },
        alternateRowStyles: { fillColor: [250, 250, 252] },
      });
      y = doc.lastAutoTable.finalY + 8;
    }
  }

  // ---- 3. Analysis Configuration ----
  sectionNum++;
  y = sectionHeader(doc, y, 'Analysis Configuration', sectionNum);
  {
    const configRows = [
      ['Analysis Type', analysisType],
      ['Problem Type', problemType.charAt(0).toUpperCase() + problemType.slice(1)],
    ];
    if (targetColumn) configRows.push(['Target Variable', targetColumn]);
    if (hasSupervisedResult) {
      configRows.push(['Evaluation Method', evalMode === 'cv' ? 'K-Fold Cross-Validation' : 'Train/Test Split (80/20)']);
      configRows.push(['Algorithms Evaluated', String(trainingResult.leaderboard?.length || 1)]);
      if (trainingResult.totalTime) configRows.push(['Total Training Time', `${trainingResult.totalTime.toFixed(2)} seconds`]);
      if (trainingResult.splitInfo) {
        configRows.push(['Training Samples', String(trainingResult.splitInfo.trainSize || '—')]);
        configRows.push(['Test Samples', String(trainingResult.splitInfo.testSize || '—')]);
      }
      if (trainingResult.dataInfo) {
        configRows.push(['Total Features', String(trainingResult.dataInfo.numFeatures || '—')]);
        if (trainingResult.dataInfo.removedLeakageColumns?.length > 0) {
          configRows.push(['Removed (Data Leakage)', trainingResult.dataInfo.removedLeakageColumns.join(', ')]);
        }
        if (trainingResult.dataInfo.textColumns?.length > 0) {
          configRows.push(['Text Features (TF-IDF)', trainingResult.dataInfo.textColumns.join(', ')]);
        }
      }
    }
    if (hasUnsupervisedResult) {
      configRows.push(['Clustering Algorithms', 'K-Means, DBSCAN, Hierarchical']);
      configRows.push(['Features Used', String(unsupervisedResult.preprocessing?.featureNames?.length || '—')]);
    }

    y = ensureSpace(doc, y, 40);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Parameter', 'Value']],
      body: configRows,
      theme: 'grid',
      styles: { fontSize: 8, cellPadding: 2.5 },
      headStyles: { fillColor: C.primary, textColor: C.white, fontStyle: 'bold' },
      columnStyles: { 0: { fontStyle: 'bold', cellWidth: 55 } },
      alternateRowStyles: { fillColor: [250, 250, 252] },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- 4. Model Leaderboard & Ranking (Supervised) ----
  if (hasSupervisedResult && trainingResult.leaderboard?.length > 0) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Model Leaderboard & Ranking', sectionNum);

    const lb = trainingResult.leaderboard;
    y = narrate(doc, y,
      `${lb.length} algorithm(s) were trained and evaluated. ` +
      `Models are ranked by ${problemType === 'classification' ? 'accuracy' : 'R² score'} on the test set. ` +
      `The top-ranked model is highlighted below.`,
      mw
    );

    const lbRows = lb.map((m, i) => {
      const tm = m.testMetrics || m.metrics || {};
      const score = problemType === 'classification'
        ? (tm.accuracy != null ? pct(tm.accuracy) : fmt(tm.f1))
        : fmt(tm.r2);
      return [
        `#${i + 1}`,
        ALGO_NAMES[m.algorithm] || m.algorithm,
        score,
        m.durationSec ? m.durationSec.toFixed(3) + 's' : '—',
        problemType === 'classification' ? (tm.f1 != null ? fmt(tm.f1) : '—') : (tm.mae != null ? fmt(tm.mae) : '—'),
      ];
    });

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Rank', 'Algorithm', problemType === 'classification' ? 'Accuracy' : 'R² Score', 'Duration', problemType === 'classification' ? 'F1 Score' : 'MAE']],
      body: lbRows,
      theme: 'grid',
      styles: { fontSize: 7.5, cellPadding: 2.5 },
      headStyles: { fillColor: C.primary, textColor: C.white, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [250, 250, 252] },
      didParseCell: (data) => {
        if (data.section === 'body' && data.row.index === 0) {
          data.cell.styles.fillColor = [237, 233, 254]; // violet-50 for winner
          data.cell.styles.fontStyle = 'bold';
        }
      },
    });
    y = doc.lastAutoTable.finalY + 5;

    // Visual bar chart of scores
    const chartData = lb.slice(0, 10).map(m => ({
      label: ALGO_NAMES[m.algorithm] || m.algorithm,
      value: problemType === 'classification'
        ? (m.testMetrics?.accuracy ?? m.metrics?.accuracy ?? 0)
        : (m.testMetrics?.r2 ?? m.metrics?.r2 ?? 0),
    }));
    y = ensureSpace(doc, y, chartData.length * 14 + 20);
    y = drawBarChart(doc, chartData, 16, y, mw, chartData.length * 14 + 15,
      `Algorithm ${problemType === 'classification' ? 'Accuracy' : 'R² Score'} Comparison`, C.primary);
  }

  // ---- 5. Best Model Performance ----
  if (hasSupervisedResult) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Best Model Performance', sectionNum);

    const best = trainingResult.bestModel;
    const algoName = ALGO_NAMES[best?.algorithm] || best?.algorithm || 'Unknown';
    const tm = best?.testMetrics || best?.metrics || {};
    const trainM = best?.trainMetrics || {};

    y = narrate(doc, y,
      `The winning algorithm is ${algoName} (${problemType}). Below are detailed performance metrics on both training and test sets.`,
      mw
    );

    // Metrics cards row
    y = ensureSpace(doc, y, 30);
    const metricPairs = problemType === 'classification'
      ? [['Accuracy', pct(tm.accuracy), C.primary], ['F1 Score', fmt(tm.f1), C.secondary], ['Precision', fmt(tm.precision), C.blue], ['Recall', fmt(tm.recall), C.accent]]
      : [['R² Score', fmt(tm.r2), C.primary], ['MAE', fmt(tm.mae), C.secondary], ['RMSE', fmt(tm.rmse), C.blue], ['MSE', fmt(tm.mse), C.accent]];

    const mcW2 = (mw - 9) / metricPairs.length;
    metricPairs.forEach(([label, value, color], i) => {
      drawMetricCard(doc, 16 + i * (mcW2 + 3), y, mcW2, 22, label, value, color);
    });
    y += 30;

    // Train vs Test comparison table
    const compRows = [];
    if (problemType === 'classification') {
      compRows.push(['Accuracy', pct(trainM.accuracy), pct(tm.accuracy)]);
      compRows.push(['F1 Score', fmt(trainM.f1), fmt(tm.f1)]);
      compRows.push(['Precision', fmt(trainM.precision), fmt(tm.precision)]);
      compRows.push(['Recall', fmt(trainM.recall), fmt(tm.recall)]);
    } else {
      compRows.push(['R²', fmt(trainM.r2), fmt(tm.r2)]);
      compRows.push(['MAE', fmt(trainM.mae), fmt(tm.mae)]);
      compRows.push(['RMSE', fmt(trainM.rmse), fmt(tm.rmse)]);
      compRows.push(['MSE', fmt(trainM.mse), fmt(tm.mse)]);
    }

    y = ensureSpace(doc, y, 35);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Metric', 'Train Set', 'Test Set']],
      body: compRows,
      theme: 'grid',
      styles: { fontSize: 8, cellPadding: 2.5, halign: 'center' },
      headStyles: { fillColor: C.primary, textColor: C.white, fontStyle: 'bold' },
      columnStyles: { 0: { halign: 'left', fontStyle: 'bold' } },
      alternateRowStyles: { fillColor: [250, 250, 252] },
    });
    y = doc.lastAutoTable.finalY + 5;

    // Overfitting analysis
    const trainScore = problemType === 'classification' ? trainM.accuracy : trainM.r2;
    const testScore = problemType === 'classification' ? tm.accuracy : tm.r2;
    if (trainScore != null && testScore != null) {
      const gap = trainScore - testScore;
      let note;
      if (gap > 0.15) note = 'The model shows signs of significant overfitting (train/test gap > 15%). Consider regularization, simpler models, or more training data.';
      else if (gap > 0.05) note = 'There is moderate overfitting observed. The model generalizes reasonably but may benefit from cross-validation tuning.';
      else note = 'The model generalizes well with minimal train/test gap, indicating low overfitting risk.';
      y = narrate(doc, y, `Overfitting Check: ${note}`, mw);
    }

    // Residual stats for regression
    if (problemType === 'regression' && trainingResult.residualStats) {
      const rs = trainingResult.residualStats;
      y = narrate(doc, y,
        `Residual Analysis: Mean residual = ${fmt(rs.mean)}, Std Dev = ${fmt(rs.std)}, Min = ${fmt(rs.min)}, Max = ${fmt(rs.max)}.` +
        (Math.abs(rs.mean || 0) < 0.01 ? ' The near-zero mean indicates unbiased predictions.' : ' Non-zero mean residual suggests systematic prediction bias.'),
        mw
      );
    }
  }

  // ---- 6. Model Comparison (if multiple models) ----
  if (models?.length > 1) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Model Comparison', sectionNum);

    y = narrate(doc, y,
      `${models.length} model(s) are available in the current session. The table below compares their key metrics across different algorithms.`,
      mw
    );

    const compModelRows = models.map(m => {
      const tm = m.metrics || m.testMetrics || {};
      return [
        ALGO_NAMES[m.algorithm] || m.algorithm,
        m.problemType || problemType,
        problemType === 'classification' ? pct(tm.accuracy) : fmt(tm.r2),
        problemType === 'classification' ? fmt(tm.f1) : fmt(tm.mae),
        m.durationSec ? m.durationSec.toFixed(3) + 's' : '—',
        m.evalMode || evalMode || '—',
      ];
    });

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Algorithm', 'Type', problemType === 'classification' ? 'Accuracy' : 'R²', problemType === 'classification' ? 'F1' : 'MAE', 'Duration', 'Eval Mode']],
      body: compModelRows,
      theme: 'grid',
      styles: { fontSize: 7, cellPadding: 2 },
      headStyles: { fillColor: C.secondary, textColor: C.white, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [250, 250, 252] },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- 7. Feature Importance & Explainability ----
  {
    const bestModel = models?.[0];
    const fi = shapGlobal?.importance || bestModel?.featureImportance;
    if (fi?.length > 0) {
      sectionNum++;
      y = sectionHeader(doc, y, 'Feature Importance & Explainability', sectionNum);

      const source = shapGlobal?.importance ? 'SHAP (SHapley Additive exPlanations)' : 'Model-derived feature importance';
      y = narrate(doc, y,
        `Feature importance was computed using ${source}. This shows which features have the greatest impact on model predictions. ` +
        `Higher values indicate stronger influence on the model output.`,
        mw
      );

      // Bar chart
      const chartFI = fi.slice(0, 15).map(f => ({
        label: f.feature || f.name || `Feature ${fi.indexOf(f)}`,
        value: f.importance || f.value || 0,
      }));
      y = ensureSpace(doc, y, chartFI.length * 14 + 20);
      y = drawBarChart(doc, chartFI, 16, y, mw, chartFI.length * 14 + 15, 'Top Features by Importance', C.accent);

      // Table
      const fiRows = fi.slice(0, 20).map((f, i) => [
        `#${i + 1}`,
        f.feature || f.name || `Feature ${i}`,
        fmt(f.importance || f.value, 6),
      ]);
      y = ensureSpace(doc, y, 30);
      doc.autoTable({
        startY: y, margin: { left: 16, right: 16 },
        head: [['Rank', 'Feature', 'Importance Score']],
        body: fiRows,
        theme: 'grid',
        styles: { fontSize: 7.5, cellPadding: 2 },
        headStyles: { fillColor: C.accent, textColor: C.white, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [240, 253, 244] },
      });
      y = doc.lastAutoTable.finalY + 8;
    }
  }

  // ---- 8. LIME Local Interpretation ----
  if (limeResult?.contributions?.length > 0) {
    sectionNum++;
    y = sectionHeader(doc, y, 'LIME Local Interpretation', sectionNum);

    y = narrate(doc, y,
      `LIME (Local Interpretable Model-agnostic Explanations) explains how each feature contributed to an individual prediction. ` +
      `Positive weights push the prediction higher; negative weights push it lower.` +
      (limeResult.prediction != null ? ` The predicted value for this instance is ${fmt(limeResult.prediction)}.` : ''),
      mw
    );

    const limeRows = limeResult.contributions.slice(0, 20).map(c => [
      c.feature,
      fmt(c.weight, 6),
      fmt(c.contribution, 6),
      c.weight > 0 ? 'Positive' : c.weight < 0 ? 'Negative' : 'Neutral',
    ]);
    if (limeResult.intercept != null) limeRows.push(['(Intercept)', fmt(limeResult.intercept, 6), '—', '—']);

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Feature', 'Weight', 'Contribution', 'Direction']],
      body: limeRows,
      theme: 'grid',
      styles: { fontSize: 7.5, cellPadding: 2 },
      headStyles: { fillColor: C.blue, textColor: C.white, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [239, 246, 255] },
      didParseCell: (data) => {
        if (data.section === 'body' && data.column.index === 3) {
          if (data.cell.raw === 'Positive') data.cell.styles.textColor = [16, 185, 129];
          else if (data.cell.raw === 'Negative') data.cell.styles.textColor = [239, 68, 68];
        }
      },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- 9. Unsupervised Analysis Results ----
  if (hasUnsupervisedResult) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Unsupervised Analysis Results', sectionNum);

    const best = unsupervisedResult.bestAlgorithm;
    y = narrate(doc, y,
      `Unsupervised clustering analysis was performed to identify natural groupings in the data. ` +
      `The best-performing algorithm is ${best?.name || 'K-Means'} with ${best?.k || '?'} clusters ` +
      `and a silhouette score of ${fmt(best?.silhouette, 3)} (range: -1 to 1, higher is better).`,
      mw
    );

    // All algorithms comparison
    const algos = unsupervisedResult.results || [];
    if (algos.length > 0) {
      const algoRows = algos.map(a => [
        a.name || '—', String(a.k || '—'), fmt(a.silhouette, 4),
        fmt(a.calinski, 2), fmt(a.davies, 4),
      ]);
      y = ensureSpace(doc, y, 30);
      doc.autoTable({
        startY: y, margin: { left: 16, right: 16 },
        head: [['Algorithm', 'Clusters', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']],
        body: algoRows,
        theme: 'grid',
        styles: { fontSize: 7.5, cellPadding: 2 },
        headStyles: { fillColor: C.secondary, textColor: C.white, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [253, 244, 255] },
      });
      y = doc.lastAutoTable.finalY + 8;
    }

    // Cluster interpretations
    const interp = unsupervisedResult.interpretation?.interpretations;
    if (interp?.length > 0) {
      y = narrate(doc, y, 'Cluster Profiles:', mw);
      interp.forEach((cl, i) => {
        y = ensureSpace(doc, y, 10);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(...C.secondary);
        doc.text(`Cluster ${i} (${cl.size || '?'} samples): ${cl.label || ''}`, 18, y);
        y += 4.5;
        if (cl.description) {
          y = narrate(doc, y, cl.description, mw - 4);
        }
      });
    }
  }

  // ---- 10. Clustering Results (standalone K-Means) ----
  if (clusterResult && !hasUnsupervisedResult) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Clustering Results', sectionNum);
    y = narrate(doc, y,
      `K-Means clustering was applied, producing ${clusterResult.k || '?'} clusters ` +
      `with a silhouette score of ${fmt(clusterResult.silhouette, 3)}.`,
      mw
    );
  }

  // ---- 11. Anomaly Detection Results ----
  if (anomalyResult) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Anomaly Detection Results', sectionNum);
    y = narrate(doc, y,
      `Anomaly detection was run using the ${anomalyResult.method || 'Z-Score'} method ` +
      `(threshold: ${anomalyResult.threshold || '3'}). ` +
      `${anomalyResult.anomalyCount || 0} anomalies were detected out of ${anomalyResult.totalRows || dataProfile?.rowCount || '?'} records ` +
      `(${anomalyResult.totalRows ? ((anomalyResult.anomalyCount / anomalyResult.totalRows) * 100).toFixed(1) : '?'}%).`,
      mw
    );
  }

  // ---- 12. Prediction History ----
  if (predictionHistory?.length > 0) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Prediction History', sectionNum);
    y = narrate(doc, y,
      `${predictionHistory.length} prediction(s) have been made during this session. Below is a summary of the most recent predictions.`,
      mw
    );

    const predRows = predictionHistory.slice(-20).map((p, i) => {
      const inputStr = Object.entries(p.input || {}).slice(0, 3).map(([k, v]) => `${k}=${v}`).join(', ');
      return [
        String(i + 1),
        p.type === 'cluster' ? 'Clustering' : (problemType.charAt(0).toUpperCase() + problemType.slice(1)),
        p.model || '—',
        inputStr + (Object.keys(p.input || {}).length > 3 ? '…' : ''),
        String(p.prediction ?? '—'),
      ];
    });

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['#', 'Type', 'Algorithm', 'Input (Summary)', 'Prediction']],
      body: predRows,
      theme: 'grid',
      styles: { fontSize: 7, cellPadding: 2, overflow: 'ellipsize' },
      headStyles: { fillColor: C.primary, textColor: C.white, fontStyle: 'bold' },
      columnStyles: { 3: { cellWidth: 55 } },
      alternateRowStyles: { fillColor: [250, 250, 252] },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- 13. Global Leaderboard ----
  if (leaderboardEntries?.length > 0) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Global Leaderboard', sectionNum);
    y = narrate(doc, y,
      `The global leaderboard tracks the best models trained across all sessions and datasets. ` +
      `${leaderboardEntries.length} entries are currently saved.`,
      mw
    );

    const lbRows = leaderboardEntries.slice(0, 20).map((e, i) => [
      `#${i + 1}`,
      ALGO_NAMES[e.algorithm] || e.algorithm || '—',
      e.dataset_name || '—',
      e.target_column || '—',
      e.problem_type || '—',
      fmt(e.best_metric_value || Object.values(e.metrics || {})[0], 4),
    ]);

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Rank', 'Algorithm', 'Dataset', 'Target', 'Type', 'Score']],
      body: lbRows,
      theme: 'grid',
      styles: { fontSize: 7, cellPadding: 2, overflow: 'ellipsize' },
      headStyles: { fillColor: C.amber, textColor: C.white, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [255, 251, 235] },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- 14. Model Deployments ----
  if (deployments?.length > 0) {
    sectionNum++;
    y = sectionHeader(doc, y, 'Model Deployments', sectionNum);
    y = narrate(doc, y,
      `${deployments.length} model(s) have been deployed with public prediction URLs. ` +
      `${deployments.filter(d => d.enabled).length} are currently active.`,
      mw
    );

    const depRows = deployments.map(d => [
      d.name || '—',
      d.deploy_id || '—',
      d.enabled ? 'Active' : 'Disabled',
      String(d.prediction_count || 0),
      d.created_at ? new Date(d.created_at).toLocaleDateString() : '—',
    ]);

    y = ensureSpace(doc, y, 30);
    doc.autoTable({
      startY: y, margin: { left: 16, right: 16 },
      head: [['Model Name', 'Deploy ID', 'Status', 'Predictions', 'Created']],
      body: depRows,
      theme: 'grid',
      styles: { fontSize: 7.5, cellPadding: 2 },
      headStyles: { fillColor: C.accent, textColor: C.white, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [240, 253, 244] },
      didParseCell: (data) => {
        if (data.section === 'body' && data.column.index === 2) {
          data.cell.styles.textColor = data.cell.raw === 'Active' ? [16, 185, 129] : [239, 68, 68];
          data.cell.styles.fontStyle = 'bold';
        }
      },
    });
    y = doc.lastAutoTable.finalY + 8;
  }

  // ---- FINAL: Conclusions & Recommendations ----
  sectionNum++;
  y = sectionHeader(doc, y, 'Conclusions & Recommendations', sectionNum);
  {
    const parts = [];
    if (hasSupervisedResult) {
      const best = trainingResult.bestModel;
      const algoName = ALGO_NAMES[best?.algorithm] || best?.algorithm || 'Unknown';
      const tm = best?.testMetrics || best?.metrics || {};
      parts.push(`The analysis evaluated ${trainingResult.leaderboard?.length || 1} algorithms on the "${datasetName}" dataset for a ${problemType} task. `);
      parts.push(`${algoName} emerged as the best performer`);
      if (problemType === 'classification') parts.push(` with ${pct(tm.accuracy)} accuracy and ${fmt(tm.f1)} F1-score.`);
      else parts.push(` with an R² of ${fmt(tm.r2)} and MAE of ${fmt(tm.mae)}.`);

      parts.push('\n\nRecommendations:\n');
      parts.push('- Consider deploying this model via the Deploy tab for production use.\n');
      if (trainingResult.leaderboard?.length > 1) parts.push('- Experiment with hyperparameter tuning on top-performing algorithms for further gains.\n');
      const trainScore = problemType === 'classification' ? best?.trainMetrics?.accuracy : best?.trainMetrics?.r2;
      const testScore = problemType === 'classification' ? tm.accuracy : tm.r2;
      if (trainScore && testScore && trainScore - testScore > 0.1) {
        parts.push('- The model shows some overfitting. Consider regularization, feature selection, or cross-validation.\n');
      }
      parts.push('- Use SHAP/LIME explainability to validate that the model relies on meaningful features.\n');
      parts.push('- Save this analysis to History for future reference and comparison.');
    }
    if (hasUnsupervisedResult) {
      const best = unsupervisedResult.bestAlgorithm;
      parts.push(`Unsupervised clustering identified ${best?.k || '?'} natural groups in the data using ${best?.name || 'K-Means'}. `);
      parts.push(`The silhouette score of ${fmt(best?.silhouette, 3)} indicates ${best?.silhouette > 0.5 ? 'well-defined' : best?.silhouette > 0.25 ? 'moderate' : 'weak'} cluster separation.\n\n`);
      parts.push('Recommendations:\n');
      parts.push('- Use the cluster profiles to build targeted strategies for each segment.\n');
      parts.push('- Run anomaly detection to identify outliers that may not fit any cluster.\n');
      parts.push('- Consider supervised models with cluster labels as an additional feature.');
    }
    if (!hasSupervisedResult && !hasUnsupervisedResult) {
      parts.push('No analysis has been performed yet. Upload a dataset and run either supervised or unsupervised analysis to generate insights.');
    }
    y = narrate(doc, y, parts.join(''), mw);
  }

  // Add footers to all pages
  addFooter(doc);

  // Save the PDF
  const filename = `AutoML_Report_${datasetName.replace(/[^a-zA-Z0-9]/g, '_')}_${now.toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
  return filename;
}
