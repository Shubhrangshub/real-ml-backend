// ==================== CONSTANTS ====================
export const CLUSTER_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea', '#0891b2', '#e11d48', '#4f46e5', '#059669', '#d97706'];
export const fadeInUp = { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -20 } };
export const staggerContainer = { animate: { transition: { staggerChildren: 0.1 } } };
export const ALGO_NAMES = { linear_regression: 'Linear Regression', ridge_regression: 'Ridge Regression', logistic_regression: 'Logistic Regression', decision_tree: 'Decision Tree', random_forest: 'Random Forest', random_forest_regressor: 'Random Forest', random_forest_classifier: 'Random Forest', gradient_boosting: 'Gradient Boosting', knn: 'KNN', svm: 'SVM', naive_bayes: 'Naive Bayes', baseline: 'Baseline' };
export const ALGO_COLORS = { linear_regression: '#2563eb', ridge_regression: '#7c3aed', logistic_regression: '#2563eb', decision_tree: '#16a34a', random_forest_regressor: '#059669', random_forest_classifier: '#059669', gradient_boosting: '#d97706', knn: '#dc2626', svm: '#9333ea', naive_bayes: '#0891b2', baseline: '#6b7280' };

export const ALGO_DESCRIPTIONS = {
  auto: 'Trains all compatible algorithms and picks the one with the best score. Ideal when you are unsure which to use.',
  linear: 'Fits a straight line through your data. Best for simple numeric relationships between features and target.',
  ridge: 'Like Linear Regression but prevents overfitting by penalizing large coefficients. Use when you have many features.',
  logistic: 'Predicts categories (yes/no, A/B/C). Works well for binary classification with a clear decision boundary.',
  decision_tree: 'Splits data using yes/no questions. Easy to understand and visualize, but can overfit on small data.',
  random_forest: 'Combines many decision trees for more reliable predictions. Great all-rounder for both classification and regression.',
  gradient_boosting: 'Builds trees one at a time, each correcting the previous mistakes. Often the most accurate, but slower.',
  knn: 'Predicts based on the K most similar records. Simple and intuitive, works best with normalized features.',
  svm: 'Finds the best boundary between classes. Powerful for classification, especially with clear margins in data.',
  naive_bayes: 'Uses probability to classify. Very fast and works surprisingly well, especially with text or categorical data.',
};

export const GUIDE_STEPS = [
  { step: 1, title: 'Upload Dataset', desc: 'Upload a CSV file or pick a sample dataset. Ensure your data has meaningful columns and enough rows (at least 10+).' },
  { step: 2, title: 'Review & Clean', desc: 'Check the Dataset Scanner for issues like missing values or outliers. Use the auto-clean tools to fix them.' },
  { step: 3, title: 'Select Target', desc: 'Pick the column you want to predict. For classification choose a categorical column; for regression pick a numeric one.' },
  { step: 4, title: 'Train Model', desc: 'Choose an algorithm (or let Auto pick the best). Click "Start Training" and wait for results.' },
  { step: 5, title: 'Explore Results', desc: 'Review accuracy metrics, the algorithm leaderboard, and confusion matrix or scatter plots.' },
  { step: 6, title: 'Make Predictions', desc: 'Go to the Predictions tab to enter new data and get predictions from your trained model.' },
  { step: 7, title: 'Explain with XAI', desc: 'Visit Explainability to understand why the model makes its predictions using SHAP and LIME.' },
];

export const METRIC_EXPLANATIONS = {
  r2: { name: 'R² Score', description: 'Shows how well the model explains the data. Values closer to 1 mean better predictions.', higherBetter: true },
  mae: { name: 'Mean Absolute Error', description: 'Average prediction error. Lower values mean the model\'s predictions are closer to actual values.', higherBetter: false },
  rmse: { name: 'Root Mean Squared Error', description: 'Like MAE but penalizes larger errors more heavily. Lower is better.', higherBetter: false },
  accuracy: { name: 'Accuracy', description: 'Percentage of correct predictions. 100% means every prediction was right.', higherBetter: true },
  precision: { name: 'Precision', description: 'When the model predicts a class, how often is it correct? Higher means fewer false alarms.', higherBetter: true },
  recall: { name: 'Recall', description: 'Of all actual positives, how many did the model catch? Higher means fewer missed cases.', higherBetter: true },
  f1: { name: 'F1 Score', description: 'Balances precision and recall. Higher values mean fewer classification mistakes overall.', higherBetter: true },
  silhouette: { name: 'Silhouette Score', description: 'Measures how well data points fit within their clusters. Higher values mean clearer cluster separation.', higherBetter: true },
  daviesBouldin: { name: 'Davies-Bouldin Index', description: 'Measures how similar clusters are to each other. Lower values mean better-separated, more distinct clusters.', higherBetter: false },
  calinskiHarabasz: { name: 'Calinski-Harabasz Score', description: 'Ratio of between-cluster to within-cluster spread. Higher means denser, better-separated clusters.', higherBetter: true },
  inertia: { name: 'Inertia', description: 'Sum of distances from each point to its cluster center. Lower values mean tighter clusters, but too low may mean too many clusters.', higherBetter: false },
  cvScore: { name: 'Cross-Validation Score', description: 'Average model performance across multiple data splits. More reliable than a single test score.', higherBetter: true },
  shapValue: { name: 'SHAP Value', description: 'Shows how much a feature pushes the prediction up (positive) or down (negative) for a specific record.', higherBetter: null },
  baseValue: { name: 'Base Value', description: 'The average model prediction across the dataset. SHAP values show how each feature moves the prediction away from this baseline.', higherBetter: null },
};

export const UNSUPERVISED_TERMS = [
  { key: 'kmeans', name: 'K-Means', desc: 'Groups data into K clusters by minimizing the distance between points and their assigned cluster center. Fast and works well for spherical clusters.' },
  { key: 'hierarchical', name: 'Hierarchical Clustering', desc: 'Builds a tree of clusters by progressively merging the closest pairs. No need to specify K in advance.' },
  { key: 'dbscan', name: 'DBSCAN', desc: 'Finds clusters based on density. Points in dense regions form clusters; sparse points are marked as noise. Discovers clusters of arbitrary shape.' },
  { key: 'gmm', name: 'Gaussian Mixture Model', desc: 'Models data as a mixture of Gaussian distributions. Soft clustering: each point has a probability of belonging to each cluster.' },
  { key: 'cluster', name: 'Cluster', desc: 'A group of similar data points. Points within a cluster are more similar to each other than to points in other clusters.' },
  { key: 'centroid', name: 'Centroid', desc: 'The center point of a cluster, calculated as the average of all points in that cluster. Represents the "typical" member.' },
  { key: 'densityClustering', name: 'Density-Based Clustering', desc: 'Clustering approach that groups together points in high-density regions separated by low-density areas. DBSCAN is the most common example.' },
  { key: 'silhouette', name: 'Silhouette Score', desc: 'Measures how similar a point is to its own cluster vs. other clusters. Ranges from -1 to 1. Higher = better-defined clusters.' },
  { key: 'daviesBouldin', name: 'Davies-Bouldin Index', desc: 'Measures average similarity between clusters. Lower values = better separation. 0 = perfect clustering.' },
  { key: 'calinskiHarabasz', name: 'Calinski-Harabasz Score', desc: 'Ratio of between-cluster to within-cluster variance. Higher values = denser, better-separated clusters.' },
  { key: 'pca', name: 'PCA (Principal Component Analysis)', desc: 'Reduces data dimensions by finding directions of maximum variance. Useful for visualizing high-dimensional data in 2D.' },
  { key: 'tsne', name: 't-SNE', desc: 'Non-linear dimensionality reduction that preserves local structure. Excellent for visualizing clusters in 2D but not for distance interpretation.' },
  { key: 'anomalyDetection', name: 'Anomaly Detection', desc: 'Identifies data points that differ significantly from the majority. These outliers may indicate errors, fraud, or unusual behavior.' },
  { key: 'isolationForest', name: 'Isolation Forest', desc: 'Detects anomalies by randomly partitioning data. Anomalies are easier to isolate and have shorter path lengths in random trees.' },
  { key: 'lof', name: 'Local Outlier Factor', desc: 'Measures how isolated a point is relative to its neighbors. Points with much lower density than their neighbors are flagged as outliers.' },
];

export const SAMPLE_DATASETS = [
  { name: 'Loan Approval', description: 'Binary classification (20 rows)', data: `age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1\n22,32000,600,5000,0\n55,130000,810,80000,1\n29,48000,660,12000,0\n40,90000,750,35000,1\n33,70000,710,22000,1\n27,44000,640,11000,0\n50,110000,795,65000,1\n36,78000,730,28000,1\n24,40000,630,9000,0\n44,98000,770,45000,1` },
  { name: 'House Prices', description: 'Regression (15 rows)', data: `size,bedrooms,age,location_score,price\n1200,2,5,7,250000\n1800,3,10,8,380000\n2500,4,3,9,520000\n1000,1,15,6,180000\n2200,3,7,8,450000\n1500,2,8,7,300000\n3000,5,2,9,620000\n1100,1,20,5,170000\n1900,3,5,8,400000\n2800,4,1,10,580000\n1600,2,12,7,310000\n2100,3,6,8,430000\n1350,2,9,6,270000\n2600,4,4,9,540000\n1750,3,11,7,350000` },
  { name: 'Insurance Costs', description: 'Financial regression (20 rows)', data: `age,sex,bmi,children,smoker,region,charges\n19,female,27.9,0,yes,southwest,16884.92\n18,male,33.77,1,no,southeast,1725.55\n28,male,33.0,3,no,southeast,4449.46\n33,male,22.705,0,no,northwest,21984.47\n32,male,28.88,0,no,northwest,3866.86\n31,female,25.74,0,no,southeast,3756.62\n46,female,33.44,1,no,southeast,8240.59\n37,female,27.74,3,no,northwest,7281.51\n37,male,29.83,2,no,northeast,6406.41\n60,female,25.84,0,no,northwest,28923.14\n25,male,26.22,0,no,northeast,2721.32\n62,female,26.29,0,yes,southeast,27808.73\n23,male,34.4,0,no,southwest,1826.84\n56,female,39.82,0,no,southeast,11090.72\n27,male,42.13,0,yes,southeast,39611.76\n19,male,24.6,1,no,southwest,1837.24\n52,female,30.78,1,no,northeast,10797.34\n23,female,23.845,0,no,northeast,2395.17\n56,male,40.3,0,no,southwest,10602.39\n30,male,35.3,0,yes,southwest,36837.47` },
  { name: 'Customer Churn', description: 'Classification — 1,000 rows, 15 features', file: '/datasets/customer_churn.csv' },
  { name: 'Customer Segmentation', description: 'Unsupervised — 800 rows, 13 features', file: '/datasets/customer_segmentation.csv' },
];
