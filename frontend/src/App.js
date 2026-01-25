import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { 
  Brain, 
  Sparkles, 
  TrendingUp, 
  Activity, 
  Database,
  Zap,
  Settings,
  Upload,
  Play,
  Eye,
  Trash2,
  ChevronRight,
  ArrowUpRight,
  FileText,
  Target,
  Cpu,
  BarChart3
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

// Animation variants
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const [csvText, setCsvText] = useState('');
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [stats, setStats] = useState({
    totalModels: 0,
    avgAccuracy: 0,
    totalTrainings: 0,
    bestModel: '--'
  });

  const sampleDatasets = [
    { 
      name: 'Loan Approval', 
      description: 'Classification problem',
      data: `age,income,credit_score,loan_amount,approved\n25,45000,650,10000,0\n35,75000,720,25000,1\n45,95000,780,50000,1\n28,52000,680,15000,0\n52,120000,800,75000,1\n23,38000,620,8000,0\n38,82000,740,30000,1\n42,88000,760,40000,1\n30,62000,700,20000,1\n48,105000,790,60000,1`
    },
    { 
      name: 'House Prices', 
      description: 'Regression problem',
      data: `size,bedrooms,age,location_score,price\n1200,2,5,7,250000\n1800,3,10,8,380000\n2500,4,3,9,520000\n1000,1,15,6,180000\n2200,3,7,8,450000`
    },
    {
      name: 'TV Shows (Text Analysis)',
      description: 'Regression with text processing',
      data: `show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
s1,TV Show,Breaking Bad,Vince Gilligan,Bryan Cranston,United States,July 1 2020,2008,TV-MA,5 Seasons,Crime TV Shows,A high school chemistry teacher turned meth producer teams up with a former student
s2,Movie,The Shawshank Redemption,Frank Darabont,Tim Robbins,United States,June 15 2019,1994,R,142 min,Dramas,Two imprisoned men bond over a number of years finding redemption through acts of common decency
s3,TV Show,Stranger Things,The Duffer Brothers,Millie Bobby Brown,United States,July 15 2016,2016,TV-14,4 Seasons,Sci-Fi TV Shows,When a young boy disappears his mother and friends must confront terrifying supernatural forces
s4,Movie,The Dark Knight,Christopher Nolan,Christian Bale,United States,January 1 2021,2008,PG-13,152 min,Action & Adventure,When the menace known as the Joker wreaks havoc on Gotham Batman must accept one of the greatest tests
s5,TV Show,Game of Thrones,David Benioff,Emilia Clarke,United States,April 17 2019,2011,TV-MA,8 Seasons,Fantasy TV Shows,Nine noble families fight for control over the lands of Westeros while an ancient enemy returns
s6,Movie,Inception,Christopher Nolan,Leonardo DiCaprio,United States,March 1 2020,2010,PG-13,148 min,Sci-Fi Movies,A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea
s7,TV Show,The Crown,Peter Morgan,Claire Foy,United Kingdom,November 4 2016,2016,TV-MA,6 Seasons,British TV Shows,Follows the political rivalries and romance of Queen Elizabeth II reign and the events that shaped the second half
s8,Movie,Pulp Fiction,Quentin Tarantino,John Travolta,United States,September 1 2020,1994,R,154 min,Crime Movies,The lives of two mob hitmen a boxer and a pair of diner bandits intertwine in four tales of violence and redemption
s9,TV Show,The Office,Greg Daniels,Steve Carell,United States,January 1 2021,2005,TV-14,9 Seasons,TV Comedies,A mockumentary on a group of typical office workers where the workday consists of ego clashes and inappropriate behavior
s10,Movie,The Godfather,Francis Ford Coppola,Marlon Brando,United States,August 1 2019,1972,R,175 min,Classic Movies,The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son`
    }
  ];

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (models && models.length > 0) {
      setStats({
        totalModels: models.length,
        avgAccuracy: 0.87,
        totalTrainings: models.length * 5,
        bestModel: (models[0] && models[0].algorithm) || '--'
      });
    } else {
      setStats({
        totalModels: 0,
        avgAccuracy: 0,
        totalTrainings: 0,
        bestModel: '--'
      });
    }
  }, [models]);

  const loadModels = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/models`);
      if (response.data && Array.isArray(response.data.models)) {
        setModels(response.data.models);
      } else {
        setModels([]);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
      setModels([]);
    }
  };

  const parseColumns = (csvData) => {
    const lines = csvData.trim().split('\n');
    if (lines.length > 0) {
      const headers = lines[0].split(',').map(h => h.trim());
      setColumns(headers);
    }
  };

  const handleCsvTextChange = (text) => {
    setCsvText(text);
    if (text.trim()) {
      parseColumns(text);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        handleCsvTextChange(text);
      };
      reader.readAsText(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        handleCsvTextChange(event.target.result);
      };
      reader.readAsText(file);
    }
  };

  const loadSampleData = (sample) => {
    handleCsvTextChange(sample.data);
  };

  const handleTrain = async () => {
    setError('');
    setTrainingResult(null);
    
    if (!csvText) {
      setError('Please provide CSV data');
      return;
    }
    
    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }

    setIsTraining(true);

    try {
      const response = await axios.post(`${BACKEND_URL}/api/train`, {
        csv_text: csvText,
        target_column: targetColumn,
        algorithm: algorithm,
        problem_type: 'auto'
      });

      setTrainingResult(response.data);
      await loadModels();
      setError('');
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
      setError('Please provide prediction data');
      return;
    }

    try {
      const data = JSON.parse(predictionInput);
      const response = await axios.post(`${BACKEND_URL}/api/predict`, {
        model_id: selectedModelId,
        data: Array.isArray(data) ? data : [data]
      });

      setPredictionResult(response.data);
      setError('');
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

  // Stats Cards
  const StatCard = ({ title, value, change, icon: Icon, trend }) => (
    <motion.div variants={fadeInUp}>
      <Card className="hover:shadow-lg transition-shadow duration-300">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">{title}</p>
              <div className="flex items-baseline gap-2">
                <h3 className="text-3xl font-bold tracking-tight">{value}</h3>
                {change && (
                  <Badge variant={trend === 'up' ? 'default' : 'secondary'} className="gap-1">
                    <ArrowUpRight className="h-3 w-3" />
                    {change}
                  </Badge>
                )}
              </div>
            </div>
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <Icon className="h-6 w-6 text-primary" />
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
      <motion.aside 
        initial={{ x: -300 }}
        animate={{ x: 0 }}
        className="fixed left-0 top-0 z-40 h-screen w-64 border-r bg-sidebar"
      >
        <div className="flex h-full flex-col gap-2">
          <div className="flex h-16 items-center border-b border-sidebar-border px-6">
            <div className="flex items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Brain className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-sidebar-foreground">AutoML</h1>
                <p className="text-xs text-sidebar-foreground/60">Master Platform</p>
              </div>
            </div>
          </div>
          
          <nav className="flex-1 space-y-1 px-3 py-4">
            {[
              { id: 'dashboard', label: 'Dashboard', icon: Activity },
              { id: 'train', label: 'Train Models', icon: Zap },
              { id: 'predict', label: 'Predictions', icon: Sparkles },
              { id: 'models', label: 'Model Library', icon: Database },
            ].map((item) => (
              <Button
                key={item.id}
                variant={activeView === item.id ? 'secondary' : 'ghost'}
                className="w-full justify-start gap-3"
                onClick={() => setActiveView(item.id)}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Button>
            ))}
          </nav>
          
          <div className="border-t border-sidebar-border p-4">
            <Card className="bg-sidebar-accent">
              <CardContent className="p-4">
                <div className="space-y-2">
                  <p className="text-xs font-medium text-sidebar-foreground">Enterprise Ready</p>
                  <p className="text-xs text-sidebar-foreground/70">
                    Scalable ML training platform
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <div className="pl-64">
        {/* Top Header */}
        <motion.header 
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
        >
          <div className="flex h-16 items-center justify-between px-8">
            <div>
              <h2 className="text-2xl font-bold tracking-tight">
                {activeView === 'dashboard' && 'Dashboard'}
                {activeView === 'train' && 'Train New Model'}
                {activeView === 'predict' && 'Make Predictions'}
                {activeView === 'models' && 'Model Library'}
              </h2>
              <p className="text-sm text-muted-foreground">
                {activeView === 'dashboard' && 'Monitor your ML operations'}
                {activeView === 'train' && 'Create and train machine learning models'}
                {activeView === 'predict' && 'Generate predictions from trained models'}
                {activeView === 'models' && 'Manage and explore your models'}
              </p>
            </div>
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </motion.header>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mx-8 mt-4"
            >
              <Card className="border-destructive bg-destructive/10">
                <CardContent className="p-4">
                  <p className="text-sm text-destructive font-medium">⚠️ {error}</p>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Content Area */}
        <main className="p-8">
          <AnimatePresence mode="wait">
            {/* Dashboard View */}
            {activeView === 'dashboard' && (
              <motion.div
                key="dashboard"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-8"
              >
                {/* Stats Grid */}
                <motion.div 
                  variants={staggerContainer}
                  className="grid gap-6 md:grid-cols-2 lg:grid-cols-4"
                >
                  <StatCard 
                    title="Total Models" 
                    value={stats.totalModels} 
                    change="+12%" 
                    icon={Database}
                    trend="up"
                  />
                  <StatCard 
                    title="Avg Accuracy" 
                    value={`${(stats.avgAccuracy * 100).toFixed(0)}%`} 
                    change="+5%" 
                    icon={TrendingUp}
                    trend="up"
                  />
                  <StatCard 
                    title="Total Trainings" 
                    value={stats.totalTrainings} 
                    change="+23%" 
                    icon={Activity}
                    trend="up"
                  />
                  <StatCard 
                    title="Best Algorithm" 
                    value={stats.bestModel} 
                    icon={Sparkles}
                  />
                </motion.div>

                {/* Charts Section */}
                <div className="grid gap-6 lg:grid-cols-2">
                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]">
                      <CardHeader>
                        <CardTitle>Model Performance</CardTitle>
                        <CardDescription>Accuracy metrics over time</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={280}>
                          <LineChart data={[
                            { name: 'Jan', accuracy: 0.75 },
                            { name: 'Feb', accuracy: 0.82 },
                            { name: 'Mar', accuracy: 0.85 },
                            { name: 'Apr', accuracy: 0.87 },
                            { name: 'May', accuracy: 0.90 },
                          ]}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis dataKey="name" className="text-xs" />
                            <YAxis className="text-xs" />
                            <Tooltip />
                            <Line 
                              type="monotone" 
                              dataKey="accuracy" 
                              stroke="hsl(var(--primary))" 
                              strokeWidth={2}
                              dot={{ fill: 'hsl(var(--primary))' }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </motion.div>

                  <motion.div variants={fadeInUp}>
                    <Card className="h-[400px]">
                      <CardHeader>
                        <CardTitle>Algorithm Distribution</CardTitle>
                        <CardDescription>Usage by algorithm type</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={280}>
                          <BarChart data={[
                            { name: 'Random Forest', count: 12 },
                            { name: 'Gradient Boost', count: 8 },
                            { name: 'Decision Tree', count: 6 },
                            { name: 'Linear', count: 4 },
                          ]}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis dataKey="name" className="text-xs" angle={-15} textAnchor="end" height={80} />
                            <YAxis className="text-xs" />
                            <Tooltip />
                            <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </motion.div>
                </div>

                {/* Recent Activity */}
                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle>Recent Training Jobs</CardTitle>
                      <CardDescription>Latest model training activity</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {models && models.length === 0 ? (
                          <div className="flex flex-col items-center justify-center py-12 text-center">
                            <Database className="h-12 w-12 text-muted-foreground/50 mb-4" />
                            <p className="text-muted-foreground">No models trained yet</p>
                            <Button 
                              className="mt-4" 
                              onClick={() => setActiveView('train')}
                            >
                              Train Your First Model
                            </Button>
                          </div>
                        ) : (
                          models.slice(0, 5).map((model, idx) => (
                            <div key={model?.modelId || idx} className="flex items-center justify-between rounded-lg border p-4 hover:bg-accent/50 transition-colors">
                              <div className="flex items-center gap-4">
                                <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                                  <Brain className="h-5 w-5 text-primary" />
                                </div>
                                <div>
                                  <p className="font-medium">{model?.algorithm || 'Unknown'}</p>
                                  <p className="text-sm text-muted-foreground">{model?.problemType || 'Unknown'}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-4">
                                <Badge variant="secondary">Success</Badge>
                                <ChevronRight className="h-4 w-4 text-muted-foreground" />
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </motion.div>
            )}

            {/* Train View */}
            {activeView === 'train' && (
              <motion.div
                key="train"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-6"
              >
                {/* Sample Datasets */}
                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="h-5 w-5" />
                        Quick Start with Sample Data
                      </CardTitle>
                      <CardDescription>Try our pre-loaded datasets</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4 md:grid-cols-2">
                        {sampleDatasets.map((sample, idx) => (
                          <Card 
                            key={idx}
                            className="cursor-pointer hover:shadow-md transition-shadow border-2 hover:border-primary"
                            onClick={() => loadSampleData(sample)}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-center justify-between">
                                <div>
                                  <p className="font-medium">{sample.name}</p>
                                  <p className="text-sm text-muted-foreground">{sample.description}</p>
                                </div>
                                <ChevronRight className="h-5 w-5 text-muted-foreground" />
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* CSV Upload */}
                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Upload className="h-5 w-5" />
                        Upload Your Data
                      </CardTitle>
                      <CardDescription>Drop a CSV file or paste your data</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Drag & Drop Zone */}
                      <div
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                        className={`
                          relative border-2 border-dashed rounded-lg p-12 text-center transition-all
                          ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25'}
                          hover:border-primary hover:bg-accent/50
                        `}
                      >
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleFileUpload}
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        />
                        <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-lg font-medium mb-2">Drop your CSV file here</p>
                        <p className="text-sm text-muted-foreground">or click to browse</p>
                      </div>

                      <Separator className="my-6" />

                      {/* Text Input */}
                      <div>
                        <label className="text-sm font-medium mb-2 block">Or paste CSV data:</label>
                        <textarea
                          value={csvText}
                          onChange={(e) => handleCsvTextChange(e.target.value)}
                          placeholder="Paste your CSV data here..."
                          rows={8}
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 font-mono"
                        />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>

                {/* Configuration */}
                {columns.length > 0 && (
                  <motion.div variants={fadeInUp}>
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Target className="h-5 w-5" />
                          Model Configuration
                        </CardTitle>
                        <CardDescription>Select target variable and model type</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid gap-6 md:grid-cols-2">
                          <div className="space-y-2">
                            <label className="text-sm font-medium">Target Variable</label>
                            <select
                              value={targetColumn}
                              onChange={(e) => setTargetColumn(e.target.value)}
                              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            >
                              <option value="">-- Select Target --</option>
                              {columns.map((col, idx) => (
                                <option key={idx} value={col}>{col}</option>
                              ))}
                            </select>
                          </div>

                          <div className="space-y-2">
                            <label className="text-sm font-medium">Model Type</label>
                            <select
                              value={algorithm}
                              onChange={(e) => setAlgorithm(e.target.value)}
                              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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

                        <Button
                          onClick={handleTrain}
                          disabled={isTraining || !targetColumn}
                          className="w-full mt-6 h-12"
                          size="lg"
                        >
                          {isTraining ? (
                            <>
                              <div className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                              Training Models...
                            </>
                          ) : (
                            <>
                              <Play className="h-4 w-4 mr-2" />
                              Start Training
                            </>
                          )}
                        </Button>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {/* Training Results */}
                {trainingResult && (
                  <motion.div 
                    variants={fadeInUp}
                    initial="initial"
                    animate="animate"
                  >
                    <Card className="border-2 border-primary">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-primary">
                          <Sparkles className="h-5 w-5" />
                          Training Complete!
                        </CardTitle>
                        <CardDescription>Your models have been trained successfully</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        {/* Summary Stats */}
                        <div className="grid gap-4 md:grid-cols-4">
                          <Card>
                            <CardContent className="p-4 text-center">
                              <p className="text-sm text-muted-foreground">Problem Type</p>
                              <p className="text-2xl font-bold mt-1">{trainingResult?.problemType || 'Unknown'}</p>
                            </CardContent>
                          </Card>
                          <Card>
                            <CardContent className="p-4 text-center">
                              <p className="text-sm text-muted-foreground">Best Model</p>
                              <p className="text-2xl font-bold mt-1">{trainingResult?.bestModel?.algorithm || 'Unknown'}</p>
                            </CardContent>
                          </Card>
                          <Card>
                            <CardContent className="p-4 text-center">
                              <p className="text-sm text-muted-foreground">Training Time</p>
                              <p className="text-2xl font-bold mt-1">{trainingResult?.totalTime ? trainingResult.totalTime.toFixed(2) + 's' : '-'}</p>
                            </CardContent>
                          </Card>
                          <Card>
                            <CardContent className="p-4 text-center">
                              <p className="text-sm text-muted-foreground">Samples</p>
                              <p className="text-2xl font-bold mt-1">{trainingResult?.dataInfo?.numSamples || 0}</p>
                            </CardContent>
                          </Card>
                        </div>

                        {/* Metrics */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">Best Model Metrics</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid gap-3 md:grid-cols-3">
                              {trainingResult?.bestModel?.metrics && Object.entries(trainingResult.bestModel.metrics).map(([key, value]) => (
                                <div key={key} className="bg-muted/50 rounded-lg p-3">
                                  <p className="text-xs text-muted-foreground uppercase">{key}</p>
                                  <p className="text-lg font-bold">{(value * 100).toFixed(2)}%</p>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>

                        {/* Feature Importance */}
                        {trainingResult?.bestModel?.featureImportance && 
                         trainingResult.bestModel.featureImportance.length > 0 && (
                          <Card>
                            <CardHeader>
                              <CardTitle className="text-lg">Feature Importance</CardTitle>
                            </CardHeader>
                            <CardContent>
                              <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={trainingResult?.bestModel?.featureImportance || []}>
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                                  <YAxis />
                                  <Tooltip />
                                  <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                                </BarChart>
                              </ResponsiveContainer>
                            </CardContent>
                          </Card>
                        )}

                        {/* Leaderboard */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">All Models Leaderboard</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-2">
                              {trainingResult?.leaderboard && trainingResult.leaderboard.map((model, idx) => (
                                <div key={idx} className="flex items-center justify-between p-3 rounded-lg border">
                                  <div className="flex items-center gap-3">
                                    <Badge variant={idx === 0 ? 'default' : 'secondary'}>
                                      {idx + 1}
                                    </Badge>
                                    <div>
                                      <p className="font-medium">{model?.algorithm || 'Unknown'}</p>
                                      <p className="text-xs text-muted-foreground">
                                        {model?.status === 'ok' ? 'Success' : 'Failed'}
                                      </p>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    {model?.metrics && (
                                      <p className="font-mono text-sm">
                                        {(Object.values(model.metrics)[0] * 100).toFixed(2)}%
                                      </p>
                                    )}
                                    <p className="text-xs text-muted-foreground">
                                      {model?.durationSec ? `${model.durationSec.toFixed(2)}s` : '-'}
                                    </p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* Predict View */}
            {activeView === 'predict' && (
              <motion.div
                key="predict"
                variants={staggerContainer}
                initial="initial"
                animate="animate"
                exit="exit"
                className="space-y-6"
              >
                <motion.div variants={fadeInUp}>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Cpu className="h-5 w-5" />
                        Select Model
                      </CardTitle>
                      <CardDescription>Choose a trained model for predictions</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {models && models.length === 0 ? (
                        <div className="text-center py-8">
                          <Database className="h-12 w-12 text-muted-foreground/50 mx-auto mb-4" />
                          <p className="text-muted-foreground mb-4">No trained models available</p>
                          <Button onClick={() => setActiveView('train')}>
                            Train a Model First
                          </Button>
                        </div>
                      ) : (
                        <select
                          value={selectedModelId}
                          onChange={(e) => setSelectedModelId(e.target.value)}
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          <option value="">-- Select a Model --</option>
                          {models && models.map((model) => (
                            <option key={model?.modelId || Math.random()} value={model?.modelId || ''}>
                              {model?.algorithm || 'Unknown'} ({model?.problemType || 'Unknown'}) - {model?.modelId ? model.modelId.substring(0, 8) + '...' : 'Unknown'}
                            </option>
                          ))}
                        </select>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>

                {selectedModelId && (
                  <>
                    <motion.div variants={fadeInUp}>
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5" />
                            Input Data
                          </CardTitle>
                          <CardDescription>Provide data in JSON format</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <textarea
                            value={predictionInput}
                            onChange={(e) => setPredictionInput(e.target.value)}
                            placeholder={'[{"feature1": "value1", "feature2": "value2"}]'}
                            rows={10}
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring font-mono"
                          />
                          <p className="text-xs text-muted-foreground">
                            Example: {`[{"age": 30, "income": 50000, "credit_score": 700}]`}
                          </p>
                          <Button
                            onClick={handlePredict}
                            className="w-full h-12"
                            size="lg"
                          >
                            <Sparkles className="h-4 w-4 mr-2" />
                            Generate Predictions
                          </Button>
                        </CardContent>
                      </Card>
                    </motion.div>

                    {predictionResult && (
                      <motion.div 
                        variants={fadeInUp}
                        initial="initial"
                        animate="animate"
                      >
                        <Card className="border-2 border-primary">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-primary">
                              <Eye className="h-5 w-5" />
                              Prediction Results
                            </CardTitle>
                            <CardDescription>Model predictions generated successfully</CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="bg-muted/50 rounded-lg p-4 font-mono text-sm">
                              <pre className="overflow-x-auto">
                                {JSON.stringify(predictionResult, null, 2)}
                              </pre>
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    )}
                  </>
                )}
              </motion.div>
            )}

            {/* Models View */}
            {activeView === 'models' && (
              <motion.div
                key="models"
                variants={fadeInUp}
                initial="initial"
                animate="animate"
                exit="exit"
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Model Library
                        </CardTitle>
                        <CardDescription>View and manage your trained models</CardDescription>
                      </div>
                      <Badge variant="secondary" className="text-lg px-4 py-2">
                        {models.length} Models
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {models.length === 0 ? (
                      <div className="text-center py-12">
                        <Database className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
                        <h3 className="text-lg font-medium mb-2">No Models Yet</h3>
                        <p className="text-muted-foreground mb-6">
                          Train your first model to get started
                        </p>
                        <Button onClick={() => setActiveView('train')} size="lg">
                          <Zap className="h-4 w-4 mr-2" />
                          Train Your First Model
                        </Button>
                      </div>
                    ) : (
                      <div className="rounded-md border">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-4 text-left text-sm font-medium">Model ID</th>
                              <th className="p-4 text-left text-sm font-medium">Algorithm</th>
                              <th className="p-4 text-left text-sm font-medium">Type</th>
                              <th className="p-4 text-left text-sm font-medium">Created</th>
                              <th className="p-4 text-left text-sm font-medium">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {models && models.map((model, idx) => (
                              <motion.tr
                                key={model?.modelId || idx}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: idx * 0.05 }}
                                className="border-b last:border-0 hover:bg-accent/50 transition-colors"
                              >
                                <td className="p-4">
                                  <div className="flex items-center gap-2">
                                    <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                                      <Brain className="h-4 w-4 text-primary" />
                                    </div>
                                    <code className="text-xs font-mono">
                                      {model?.modelId ? model.modelId.substring(0, 12) + '...' : 'Unknown'}
                                    </code>
                                  </div>
                                </td>
                                <td className="p-4">
                                  <Badge variant="outline">{model?.algorithm || 'Unknown'}</Badge>
                                </td>
                                <td className="p-4">
                                  <span className="text-sm">{model?.problemType || 'Unknown'}</span>
                                </td>
                                <td className="p-4">
                                  <span className="text-sm text-muted-foreground">
                                    {model?.createdAt ? new Date(model.createdAt).toLocaleDateString() : 'Unknown'}
                                  </span>
                                </td>
                                <td className="p-4">
                                  <div className="flex gap-2">
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => {
                                        if (model?.modelId) {
                                          setSelectedModelId(model.modelId);
                                          setActiveView('predict');
                                        }
                                      }}
                                    >
                                      <Eye className="h-4 w-4" />
                                    </Button>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => model?.modelId && handleDeleteModel(model.modelId)}
                                    >
                                      <Trash2 className="h-4 w-4 text-destructive" />
                                    </Button>
                                  </div>
                                </td>
                              </motion.tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
