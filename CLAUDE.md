# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tick Trader is a Python-based AI trading system that analyzes futures and options tick data to predict price movements. The system processes hundreds of thousands of daily tick records using machine learning and deep learning techniques to identify trading patterns.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Install with optional advanced features
pip install -e ".[advanced]"
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest --cov=src tests/

# Run specific test module
python -m pytest tests/test_features.py

# Run with verbose output
python -m pytest -v tests/
```

### Model Training
```bash
# Train baseline model
python -m src.training.trainer --model lstm --config config/config.yaml

# Train ensemble model
python -m src.training.trainer --model ensemble --config config/config.yaml

# Train with custom parameters
python -m src.training.trainer --model transformer --epochs 100 --batch_size 64
```

### Data Processing
```bash
# Process raw tick data
python -m src.data.ingestion --input data/raw --output data/processed

# Generate features
python -m src.features.pipeline --config config/config.yaml

# Run backtesting
python -m src.evaluation.backtest --model-path data/models/latest --config config/config.yaml
```

### Real-time Prediction
```bash
# Start real-time prediction server
python -m src.prediction.realtime --config config/config.yaml --port 8080
```

## Architecture Overview

### Core Components

1. **Data Layer** (`src/data/`):
   - `models.py`: Data models for tick data and order book structures
   - `ingestion.py`: Data ingestion pipeline for raw tick data
   - `parser.py`: Parser for different data formats and exchanges

2. **Feature Engineering** (`src/features/`):
   - `order_book.py`: Order book analysis and microstructure features
   - `time_series.py`: Time-based feature extraction and windows
   - `pipeline.py`: Complete feature engineering pipeline

3. **ML Models** (`src/models/`):
   - `lstm.py`: LSTM networks for time-series prediction
   - `transformer.py`: Transformer models with attention mechanisms
   - `cnn_lstm.py`: CNN-LSTM hybrid architectures
   - `ensemble.py`: Ensemble methods (LightGBM, XGBoost, Random Forest)

4. **Training** (`src/training/`):
   - `trainer.py`: Unified training interface for all model types
   - Supports distributed training and hyperparameter tuning

5. **Prediction** (`src/prediction/`):
   - `realtime.py`: Real-time prediction engine with Redis caching
   - Sliding window predictions for live trading

6. **Evaluation** (`src/evaluation/`):
   - `backtest.py`: Comprehensive backtesting framework
   - `metrics.py`: Financial metrics (Sharpe ratio, drawdown, PnL)

7. **Storage** (`src/storage/`):
   - `parquet.py`: Efficient data storage with Apache Parquet
   - `redis_cache.py`: Real-time feature caching
   - `model_store.py`: Model versioning and persistence

### Data Flow Architecture

1. **Data Ingestion**: Raw tick data → Parquet files → Processed datasets
2. **Feature Pipeline**: Time windows → Technical indicators → ML features
3. **Model Training**: Feature datasets → Model training → Model storage
4. **Prediction**: Real-time data → Feature generation → Model inference → Trading signals

### Configuration System

The system uses YAML-based configuration (`config/config.yaml`) with sections for:
- `data`: File paths and data storage locations
- `ingestion`: Batch processing parameters
- `features`: Order book depth, time windows, scaling methods
- `training`: Model-specific hyperparameters (LSTM, Transformer, Ensemble)
- `backtest`: Trading simulation parameters
- `prediction`: Real-time prediction settings
- `logging`: Log level and output configuration

## Key Design Patterns

### Modular Model Architecture
- All models implement a common interface with `fit()`, `predict()`, and `save()` methods
- Model factory pattern for easy model switching
- Ensemble models combine multiple base learners

### Feature Engineering Pipeline
- Window-based feature generation with configurable time windows
- Microstructure features: order imbalance, flow toxicity, VPIN
- Multi-scale analysis using wavelet transforms

### Data Management
- Efficient storage with Apache Parquet format
- Redis caching for real-time feature access
- Sliding window processing for streaming data

### Backtesting Framework
- Financial metric calculations (Sharpe ratio, maximum drawdown)
- Transaction cost modeling
- Walk-forward analysis for time-series validation

## Working with Configuration

### Model Configuration
Models are configured through YAML sections. Example LSTM configuration:
```yaml
training:
  lstm:
    sequence_length: 100
    lstm_units: [128, 64, 32]
    dropout_rate: 0.2
    learning_rate: 0.001
    epochs: 50
    batch_size: 32
```

### Feature Configuration
Feature engineering parameters:
```yaml
features:
  orderbook_depth: 10
  time_windows: [5, 10, 20, 50]
  scaling_method: "standard"
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (Redis, file I/O)
- Focus on feature engineering logic and model interfaces

### Integration Tests
- End-to-end data pipeline testing
- Model training and prediction workflows
- Configuration loading and validation

### Performance Tests
- Benchmark model training times
- Memory usage profiling for large datasets
- Real-time prediction latency testing

## Common Development Patterns

### Adding New Models
1. Create model class in `src/models/` implementing the base interface
2. Add configuration section to `config/config.yaml`
3. Update `trainer.py` to support the new model type
4. Add unit tests in `tests/models/`

### Feature Engineering Extensions
1. Add feature extraction logic to appropriate module in `src/features/`
2. Update `pipeline.py` to include new features
3. Add configuration parameters as needed
4. Document feature calculation methodology

### Data Source Integration
1. Implement parser in `src/data/parser.py` for new data format
2. Update data models in `src/data/models.py`
3. Add ingestion logic to `src/data/ingestion.py`
4. Test with sample data files

## Dependencies and Versions

- **Python**: 3.8+ required
- **Core ML**: TensorFlow 2.10+, Scikit-learn 1.0+, LightGBM 3.3+, XGBoost 1.5+
- **Data**: Pandas 1.3+, NumPy 1.21+, Apache Arrow 5.0+
- **Storage**: Redis 4.0+ for caching
- **Advanced (optional)**: PyTorch, torch-geometric for GNNs

## Performance Considerations

- Use Parquet format for efficient data storage and retrieval
- Implement Redis caching for frequently accessed features
- Consider batch processing for large datasets
- Monitor memory usage during feature generation
- Use GPU acceleration for deep learning models when available

## Security and Data Privacy

- No API keys or sensitive credentials in code
- Configuration files should not contain production secrets
- Use environment variables for sensitive configuration
- Implement proper data validation for external data sources