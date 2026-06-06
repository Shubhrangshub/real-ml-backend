import jsPDF from 'jspdf';
import 'jspdf-autotable';

const VIOLET = [139, 92, 246];
const FUCHSIA = [217, 70, 239];
const DARK = [24, 24, 27];
const GRAY = [113, 113, 122];
const WHITE = [255, 255, 255];

function addHeader(doc, title, y) {
  doc.setFillColor(...VIOLET);
  doc.roundedRect(14, y, doc.internal.pageSize.width - 28, 10, 2, 2, 'F');
  doc.setTextColor(...WHITE);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text(title, 18, y + 7);
  doc.setTextColor(...DARK);
  doc.setFont('helvetica', 'normal');
  return y + 16;
}

function checkPage(doc, y, needed = 30) {
  if (y + needed > doc.internal.pageSize.height - 20) {
    doc.addPage();
    return 20;
  }
  return y;
}

function addKeyValue(doc, key, value, y) {
  y = checkPage(doc, y);
  doc.setFontSize(9);
  doc.setTextColor(...GRAY);
  doc.text(key, 18, y);
  doc.setTextColor(...DARK);
  doc.setFont('helvetica', 'bold');
  doc.text(String(value ?? '—'), 75, y);
  doc.setFont('helvetica', 'normal');
  return y + 6;
}

export function generateReport({ dataProfile, targetColumn, trainingResult, models, shapGlobal, limeResult, predictionHistory, businessInterpretation, cleaningLog }) {
  const doc = new jsPDF('p', 'mm', 'a4');
  const pw = doc.internal.pageSize.width;
  let y = 15;

  // ====== TITLE ======
  doc.setFillColor(...VIOLET);
  doc.rect(0, 0, pw, 40, 'F');
  // Gradient effect
  doc.setFillColor(...FUCHSIA);
  doc.rect(pw * 0.6, 0, pw * 0.4, 40, 'F');

  doc.setTextColor(...WHITE);
  doc.setFontSize(22);
  doc.setFont('helvetica', 'bold');
  doc.text('AutoML Analysis Report', 18, 20);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Generated: ${new Date().toLocaleString()}`, 18, 30);
  if (dataProfile?.fileName) {
    doc.text(`Dataset: ${dataProfile.fileName}`, pw - 18, 30, { align: 'right' });
  }
  doc.setTextColor(...DARK);
  y = 50;

  // ====== DATASET OVERVIEW ======
  y = addHeader(doc, 'Dataset Overview', y);
  y = addKeyValue(doc, 'Dataset', dataProfile?.fileName || 'Uploaded CSV', y);
  y = addKeyValue(doc, 'Rows', dataProfile?.rowCount?.toLocaleString(), y);
  y = addKeyValue(doc, 'Columns', dataProfile?.columnCount, y);
  y = addKeyValue(doc, 'Target Column', targetColumn || '—', y);
  y = addKeyValue(doc, 'Numeric Features', dataProfile?.numericColumns?.length, y);
  y = addKeyValue(doc, 'Categorical Features', dataProfile?.categoricalColumns?.length, y);

  if (cleaningLog?.length > 0) {
    y += 2;
    y = addKeyValue(doc, 'Data Cleaning', `${cleaningLog.length} operations applied`, y);
  }
  y += 4;

  // ====== MODEL RESULTS ======
  if (trainingResult || models?.length > 0) {
    y = checkPage(doc, y, 40);
    y = addHeader(doc, 'Model Performance', y);

    const problemType = trainingResult?.problemType || models?.[0]?.problemType || '—';
    y = addKeyValue(doc, 'Problem Type', problemType, y);
    y = addKeyValue(doc, 'Models Trained', models?.length || trainingResult?.leaderboard?.length || '—', y);

    // Best model
    const best = trainingResult?.bestModel || models?.[0];
    if (best) {
      y = addKeyValue(doc, 'Best Algorithm', best.algorithm || best.name, y);
      if (best.metrics) {
        Object.entries(best.metrics).forEach(([k, v]) => {
          y = addKeyValue(doc, `  ${k}`, typeof v === 'number' ? v.toFixed(4) : v, y);
        });
      }
    }
    y += 4;

    // Leaderboard table
    const leaderboard = trainingResult?.leaderboard || models?.map(m => ({
      algorithm: m.algorithm, ...m.metrics
    }));
    if (leaderboard?.length > 0) {
      y = checkPage(doc, y, 50);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Model Leaderboard', 18, y);
      doc.setFont('helvetica', 'normal');
      y += 4;

      // Get metric columns
      const metricKeys = [];
      leaderboard.forEach(m => {
        Object.keys(m).forEach(k => {
          if (k !== 'algorithm' && k !== 'name' && typeof m[k] === 'number' && !metricKeys.includes(k)) {
            metricKeys.push(k);
          }
        });
      });

      const headers = ['Algorithm', ...metricKeys.slice(0, 5)];
      const rows = leaderboard.map(m => [
        m.algorithm || m.name || '—',
        ...metricKeys.slice(0, 5).map(k => typeof m[k] === 'number' ? m[k].toFixed(4) : (m[k] ?? '—'))
      ]);

      doc.autoTable({
        startY: y,
        head: [headers],
        body: rows,
        margin: { left: 18, right: 18 },
        styles: { fontSize: 8, cellPadding: 2 },
        headStyles: { fillColor: VIOLET, textColor: WHITE, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [245, 243, 255] },
        theme: 'grid',
      });
      y = doc.lastAutoTable.finalY + 8;
    }
  }

  // ====== FEATURE IMPORTANCE ======
  if (shapGlobal?.length > 0) {
    y = checkPage(doc, y, 50);
    y = addHeader(doc, 'Feature Importance (SHAP)', y);

    const top10 = shapGlobal.slice(0, 10);
    const maxVal = Math.max(...top10.map(f => f.importance || f.value || 0));

    top10.forEach((feat, idx) => {
      y = checkPage(doc, y, 10);
      const val = feat.importance || feat.value || 0;
      const barWidth = maxVal > 0 ? (val / maxVal) * 80 : 0;

      doc.setFontSize(8);
      doc.setTextColor(...DARK);
      doc.text(`${idx + 1}. ${feat.feature || feat.name}`, 18, y);

      // Bar
      doc.setFillColor(139, 92, 246, 0.3);
      doc.roundedRect(75, y - 3, barWidth, 4, 1, 1, 'F');
      doc.setFillColor(...VIOLET);
      doc.roundedRect(75, y - 3, barWidth, 4, 1, 1, 'F');

      doc.setTextColor(...GRAY);
      doc.text(val.toFixed(4), 160, y);
      y += 7;
    });
    y += 4;
  }

  // ====== BUSINESS INTERPRETATION ======
  if (businessInterpretation) {
    y = checkPage(doc, y, 40);
    y = addHeader(doc, 'Business Insights', y);

    doc.setFontSize(9);
    doc.setTextColor(...DARK);
    const lines = doc.splitTextToSize(businessInterpretation, pw - 36);
    lines.forEach(line => {
      y = checkPage(doc, y, 6);
      doc.text(line, 18, y);
      y += 5;
    });
    y += 4;
  }

  // ====== PREDICTIONS SUMMARY ======
  if (predictionHistory?.length > 0) {
    y = checkPage(doc, y, 30);
    y = addHeader(doc, 'Prediction History', y);
    y = addKeyValue(doc, 'Total Predictions', predictionHistory.length, y);

    // Show last 10 predictions as a table
    const recent = predictionHistory.slice(-10);
    if (recent.length > 0 && recent[0]?.input) {
      const inputKeys = Object.keys(recent[0].input).slice(0, 4);
      const headers = [...inputKeys, 'Prediction'];
      const rows = recent.map(p => [
        ...inputKeys.map(k => String(p.input[k] ?? '').substring(0, 15)),
        String(typeof p.prediction === 'number' ? p.prediction.toFixed(4) : p.prediction)
      ]);

      doc.autoTable({
        startY: y,
        head: [headers],
        body: rows,
        margin: { left: 18, right: 18 },
        styles: { fontSize: 7, cellPadding: 1.5 },
        headStyles: { fillColor: FUCHSIA, textColor: WHITE, fontStyle: 'bold' },
        theme: 'grid',
      });
      y = doc.lastAutoTable.finalY + 8;
    }
    y += 4;
  }

  // ====== FOOTER ======
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(...GRAY);
    doc.text(`AutoML Master Report — Page ${i} of ${pageCount}`, pw / 2, doc.internal.pageSize.height - 8, { align: 'center' });
  }

  // Save
  const fileName = `automl_report_${(dataProfile?.fileName || 'analysis').replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;
  doc.save(fileName);
  return fileName;
}
