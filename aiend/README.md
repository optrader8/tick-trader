# AIend - AI Trading Engine

AIend is the core machine learning engine for the AI Trading Platform. This Python-based system analyzes futures and options tick data to predict price movements using advanced ML and deep learning techniques.

## Project Structure (Monorepo)

```
tick-trader/                    # Root monorepo
├── aiend/                     # ML Engine (this folder)
│   ├── src/                   # ML source code
│   ├── config/                # Configuration files
│   ├── data/                  # Data directories
│   ├── tests/                 # Test suite
│   ├── requirements.txt       # Python dependencies
│   └── setup.py              # Package setup
├── backend/                   # API backend service
├── frontend/                  # React web application
└── README.md                 # Main project README
```

## ML Engine Components

### 🧠 Machine Learning Models (`src/models/`)
- **LSTM Networks**: Time-series prediction for price movements
- **Transformers**: Attention-based models for sequence analysis
- **CNN-LSTM Hybrids**: Spatial-temporal pattern recognition
- **Ensemble Models**: LightGBM, XGBoost, Random Forest combinations

### 📊 Feature Engineering (`src/features/`)
- **Order Book Analysis**: Microstructure features, market imbalance indicators
- **Time Series Features**: Multiple window sizes, technical indicators
- **Real-time Processing**: Sliding window feature generation

### 🔄 Data Pipeline (`src/data/`)
- **Data Ingestion**: Raw tick data processing and validation
- **Storage**: Apache Parquet format for efficient data handling
- **Caching**: Redis integration for real-time feature access

### 🚀 Prediction Engine (`src/prediction/`)
- **Real-time Inference**: Live trading signal generation
- **Model Serving**: Optimized prediction serving
- **Performance Monitoring**: Latency and accuracy tracking

### 📈 Evaluation (`src/evaluation/`)
- **Backtesting**: Comprehensive trading simulation
- **Financial Metrics**: Sharpe ratio, maximum drawdown, PnL analysis
- **Risk Management**: Position sizing and portfolio optimization

## Getting Started

### ML Engine Development
```bash
cd aiend
pip install -r requirements.txt
pip install -e .

# Train a model
python -m src.training.trainer --model lstm --config config/config.yaml

# Run predictions
python -m src.prediction.realtime --config config/config.yaml

# Run backtesting
python -m src.evaluation.backtest --model-path data/models/latest
```

### Full Stack Development
```bash
# Backend API
cd ../backend
npm install
npm run dev

# Frontend UI
cd ../frontend
npm install
npm start
```

## Architecture Overview

The AIend ML Engine serves as the analytical core of the trading platform:

1. **Data Processing Layer**: Ingests and processes high-frequency tick data
2. **Feature Engineering Layer**: Extracts meaningful signals from raw data
3. **Model Layer**: Applies various ML models for prediction
4. **Serving Layer**: Provides real-time predictions to backend services
5. **Evaluation Layer**: Continuously validates and improves model performance

## Configuration

The system uses YAML-based configuration (`config/config.yaml`):

```yaml
# Model hyperparameters
training:
  lstm:
    sequence_length: 100
    lstm_units: [128, 64, 32]
    epochs: 50
    batch_size: 32

# Feature settings
features:
  orderbook_depth: 10
  time_windows: [5, 10, 20, 50]
  scaling_method: "standard"

# Real-time prediction
prediction:
  window_size: 100
  use_redis: false
```

## Advanced Features

### 🧬 Advanced ML Techniques
- **Graph Neural Networks**: Order book structure modeling
- **Topological Data Analysis**: High-dimensional pattern detection
- **Meta-Learning**: Market regime adaptation
- **Hawkes Processes**: Order flow analysis

### ⚡ Performance Optimizations
- **GPU Acceleration**: CUDA support for deep learning models
- **Distributed Training**: Multi-GPU and multi-node training
- **Memory Management**: Efficient handling of large datasets
- **Caching Strategy**: Redis-based feature caching

## Development Guidelines

### Adding New Models
1. Create model class in `src/models/` with standard interface
2. Add configuration parameters to `config/config.yaml`
3. Update `src/training/trainer.py` for model support
4. Add comprehensive tests in `tests/models/`

### Feature Development
1. Implement feature extraction in `src/features/`
2. Update feature pipeline configuration
3. Document methodology and mathematical formulation
4. Validate feature importance and correlation

### Testing Strategy
- Unit tests for individual components
- Integration tests for data pipelines
- Performance benchmarks for model training
- Backtesting validation for trading strategies

## Dependencies

- **Core ML**: TensorFlow, Scikit-learn, LightGBM, XGBoost
- **Data Processing**: Pandas, NumPy, Apache Arrow
- **Storage**: Redis for caching
- **Optional**: PyTorch for advanced models

This ML Engine provides the analytical foundation for the entire AI Trading Platform, generating actionable trading signals through sophisticated machine learning analysis of market data.