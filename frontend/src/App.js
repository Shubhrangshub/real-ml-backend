import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('train');
  const [csvText, setCsvText] = useState('');
  const [fileUrl, setFileUrl] = useState('');
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState([]);
  const [algorithm, setAlgorithm] = useState('auto');
  const [problemType, setProblemType] = useState('auto');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');

  const sampleDatasets = [
    { name: 'Loan Approval (Classification)', url: 'https://raw.githubusercontent.com/yourusername/data/main/loan_approval.csv' },
    { name: 'House Prices (Regression)', url: 'https://raw.githubusercontent.com/yourusername/data/main/house_prices.csv' }
  ];

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/models`);
      setModels(response.data.models);
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        setCsvText(text);
        parseColumns(text);
      };
      reader.readAsText(file);
    }
  };

  const parseColumns = (csvData) => {
    const lines = csvData.trim().split('\n');
    if (lines.length > 0) {
      const headers = lines[0].split(',').map(h => h.trim());
      setColumns(headers);
      setFeatureColumns(headers);
    }
  };

  const handleCsvTextChange = (text) => {
    setCsvText(text);
    if (text.trim()) {
      parseColumns(text);
    }
  };

  const handleTrain = async () => {
    setError('');
    setTrainingResult(null);
    
    if (!csvText && !fileUrl) {
      setError('Please provide CSV data or file URL');
      return;
    }
    
    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }

    setIsTraining(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/train`, {
        csv_text: csvText || undefined,
        file_url: fileUrl || undefined,
        target_column: targetColumn,
        feature_columns: featureColumns.filter(col => col !== targetColumn),
        algorithm: algorithm,
        problem_type: problemType
      });

      setTrainingResult(response.data);
      await loadModels();
    } catch (err) {
      setError(err.response?.data?.message || 'Training failed');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async () => {
    setError('');
    setPredictionResult(null);

    if (!selectedModelId) {
      setError('Please select a model');
      return;
    }

    if (!predictionInput) {
      setError('Please provide prediction data (JSON format)');
      return;
    }

    try {
      const data = JSON.parse(predictionInput);
      const response = await axios.post(`${BACKEND_URL}/api/predict`, {
        model_id: selectedModelId,
        data: Array.isArray(data) ? data : [data]
      });

      setPredictionResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    }
  };

  const handleDeleteModel = async (modelId) => {
    try {
      await axios.delete(`${BACKEND_URL}/api/models/${modelId}`);
      await loadModels();
      setError('');
    } catch (err) {
      setError('Failed to delete model');
    }
  };

  const loadSampleData = (sample) => {
    const sampleData = {
      'Loan Approval (Classification)': `age,income,credit_score,loan_amount,approved
25,45000,650,10000,0
35,75000,720,25000,1
45,95000,780,50000,1
28,52000,680,15000,0
52,120000,800,75000,1`,
      'House Prices (Regression)': `size,bedrooms,age,location_score,price
1200,2,5,7,250000
1800,3,10,8,380000
2500,4,3,9,520000
1000,1,15,6,180000
2200,3,7,8,450000`
    };
    
    const data = sampleData[sample.name];
    if (data) {
      handleCsvTextChange(data);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-4" data-testid="app-title">
            🤖 AutoML Master
          </h1>
          <p className="text-xl text-white opacity-90">
            Train machine learning models automatically with no code required
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center mb-8 space-x-4">
          <button
            data-testid="train-tab"
            onClick={() => setActiveTab('train')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'train'
                ? 'bg-white text-purple-700 shadow-lg'
                : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30'
            }`}
          >
            🎯 Train Models
          </button>
          <button
            data-testid="predict-tab"
            onClick={() => setActiveTab('predict')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'predict'
                ? 'bg-white text-purple-700 shadow-lg'
                : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30'
            }`}
          >
            🔮 Make Predictions
          </button>
          <button
            data-testid="models-tab"
            onClick={() => setActiveTab('models')}
            className={`px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'models'
                ? 'bg-white text-purple-700 shadow-lg'
                : 'bg-white bg-opacity-20 text-white hover:bg-opacity-30'
            }`}
          >
            📊 My Models
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500 text-white p-4 rounded-lg mb-6 shadow-lg" data-testid="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* Train Tab */}
        {activeTab === 'train' && (
          <div className="bg-white rounded-xl shadow-2xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">Train New Model</h2>

            {/* Sample Data */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Try Sample Datasets:
              </label>
              <div className="flex gap-3">
                {sampleDatasets.map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => loadSampleData(sample)}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
                    data-testid={`sample-dataset-${idx}`}
                  >
                    📁 {sample.name}
                  </button>
                ))}
              </div>
            </div>

            {/* CSV Input */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Upload CSV File:
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                data-testid="csv-file-input"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Or Paste CSV Data:
              </label>
              <textarea
                value={csvText}
                onChange={(e) => handleCsvTextChange(e.target.value)}
                placeholder="Paste your CSV data here..."
                rows={8}
                className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm"
                data-testid="csv-text-input"
              />
            </div>

            {/* Column Selection */}
            {columns.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Target Column (What to predict):
                  </label>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    data-testid="target-column-select"
                  >
                    <option value="">-- Select Target --</option>
                    {columns.map((col, idx) => (
                      <option key={idx} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Algorithm:
                  </label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                    data-testid="algorithm-select"
                  >
                    <option value="auto">Auto (Try All)</option>
                    <option value="logistic">Logistic Regression</option>
                    <option value="linear">Linear Regression</option>
                    <option value="decision_tree">Decision Tree</option>
                    <option value="random_forest">Random Forest</option>
                    <option value="gradient_boosting">Gradient Boosting</option>
                  </select>
                </div>
              </div>
            )}

            {/* Train Button */}
            <button
              onClick={handleTrain}
              disabled={isTraining}
              className={`w-full py-4 rounded-lg font-bold text-lg transition-all ${
                isTraining
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg'
              }`}
              data-testid="train-button"
            >
              {isTraining ? '🔄 Training Models...' : '🚀 Train Models'}
            </button>

            {/* Training Results */}
            {trainingResult && (
              <div className="mt-8 space-y-6">
                <div className="bg-green-50 border-2 border-green-500 rounded-lg p-6">
                  <h3 className="text-2xl font-bold text-green-800 mb-4" data-testid="training-success">
                    ✅ Training Complete!
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Problem Type</div>
                      <div className="text-xl font-bold text-purple-600">
                        {trainingResult.problemType}
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Best Algorithm</div>
                      <div className="text-xl font-bold text-blue-600">
                        {trainingResult.bestModel.algorithm}
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Training Time</div>
                      <div className="text-xl font-bold text-green-600">
                        {trainingResult.totalTime.toFixed(2)}s
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded-lg shadow">
                      <div className="text-sm text-gray-600">Samples</div>
                      <div className="text-xl font-bold text-orange-600">
                        {trainingResult.dataInfo.numSamples}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Best Model Metrics */}
                <div className="bg-white border-2 border-blue-500 rounded-lg p-6">
                  <h4 className="text-xl font-bold text-gray-800 mb-4">
                    🏆 Best Model Metrics
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(trainingResult.bestModel.metrics).map(([key, value]) => (
                      <div key={key} className="bg-gray-50 p-3 rounded-lg">
                        <div className="text-xs text-gray-600 uppercase">{key}</div>
                        <div className="text-lg font-bold text-gray-800">
                          {(value * 100).toFixed(2)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Importance */}
                {trainingResult.bestModel.featureImportance && 
                 trainingResult.bestModel.featureImportance.length > 0 && (
                  <div className="bg-white border-2 border-purple-500 rounded-lg p-6">
                    <h4 className="text-xl font-bold text-gray-800 mb-4">
                      📊 Feature Importance
                    </h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={trainingResult.bestModel.featureImportance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="importance" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Leaderboard */}
                <div className="bg-white border-2 border-gray-300 rounded-lg p-6">
                  <h4 className="text-xl font-bold text-gray-800 mb-4">
                    📋 All Models Leaderboard
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="p-3 text-left">Algorithm</th>
                          <th className="p-3 text-left">Status</th>
                          <th className="p-3 text-left">Duration</th>
                          <th className="p-3 text-left">Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trainingResult.leaderboard.map((model, idx) => (
                          <tr key={idx} className={`border-t ${model.status === 'ok' ? 'bg-white' : 'bg-red-50'}`}>
                            <td className="p-3 font-semibold">{model.algorithm}</td>
                            <td className="p-3">
                              {model.status === 'ok' ? (
                                <span className="text-green-600">✅ Success</span>
                              ) : (
                                <span className="text-red-600">❌ Failed</span>
                              )}
                            </td>
                            <td className="p-3">
                              {model.durationSec ? `${model.durationSec.toFixed(2)}s` : '-'}
                            </td>
                            <td className="p-3">
                              {model.metrics ? (
                                <span className="font-bold">
                                  {Object.values(model.metrics)[0]?.toFixed(4) || '-'}
                                </span>
                              ) : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Predict Tab */}
        {activeTab === 'predict' && (
          <div className="bg-white rounded-xl shadow-2xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">Make Predictions</h2>

            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Select Model:
              </label>
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                data-testid="model-select"
              >
                <option value="">-- Select a Model --</option>
                {models.map((model) => (
                  <option key={model.modelId} value={model.modelId}>
                    {model.algorithm} ({model.problemType}) - {model.modelId.substring(0, 8)}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Input Data (JSON format):
              </label>
              <textarea
                value={predictionInput}
                onChange={(e) => setPredictionInput(e.target.value)}
                placeholder='[{"feature1": value1, "feature2": value2, ...}]'
                rows={8}
                className="w-full p-3 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm"
                data-testid="prediction-input"
              />
              <p className="text-xs text-gray-500 mt-2">
                Example: [{"age": 30, "income": 50000, "credit_score": 700}]
              </p>
            </div>

            <button
              onClick={handlePredict}
              className="w-full py-4 bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 text-white rounded-lg font-bold text-lg transition-all shadow-lg"
              data-testid="predict-button"
            >
              🔮 Predict
            </button>

            {predictionResult && (
              <div className="mt-8 bg-green-50 border-2 border-green-500 rounded-lg p-6">
                <h3 className="text-2xl font-bold text-green-800 mb-4" data-testid="prediction-result">
                  ✅ Prediction Results
                </h3>
                <div className="bg-white p-4 rounded-lg">
                  <div className="font-mono text-sm">
                    <pre>{JSON.stringify(predictionResult, null, 2)}</pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="bg-white rounded-xl shadow-2xl p-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">My Trained Models</h2>

            {models.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <div className="text-6xl mb-4">📦</div>
                <p className="text-xl">No models trained yet</p>
                <p className="mt-2">Go to the Train tab to create your first model!</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {models.map((model) => (
                  <div
                    key={model.modelId}
                    className="bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-200 rounded-lg p-6 hover:shadow-xl transition-shadow"
                    data-testid={`model-card-${model.modelId}`}
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div className="text-2xl">🤖</div>
                      <button
                        onClick={() => handleDeleteModel(model.modelId)}
                        className="text-red-500 hover:text-red-700 font-bold"
                        data-testid={`delete-model-${model.modelId}`}
                      >
                        🗑️
                      </button>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 mb-2">
                      {model.algorithm}
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-gray-600">Type:</span>{' '}
                        <span className="font-semibold text-purple-600">{model.problemType}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">ID:</span>{' '}
                        <span className="font-mono text-xs">{model.modelId.substring(0, 16)}...</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Created:</span>{' '}
                        <span className="text-gray-800">{model.createdAt}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
