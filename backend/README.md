# Tick Trader Backend (FastAPI)

Python FastAPI backend for Tick Trader web platform.

## Features

- **File Upload API**: Upload tick data files (CSV, Parquet, JSON)
- **Analysis Management**: Start, monitor, and cancel AI analysis jobs
- **WebSocket Support**: Real-time analysis status updates
- **AI Engine Integration**: Seamless integration with aiend Python engine
- **Job Queue**: Redis-based job queue for async processing
- **Database**: PostgreSQL for metadata storage

## Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

## Running

### Development
```bash
python -m app.main
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── endpoints/      # API endpoint modules
│   │   │   ├── files.py    # File upload/management
│   │   │   ├── analysis.py # Analysis management
│   │   │   └── system.py   # System monitoring
│   │   └── routes.py       # Route aggregation
│   ├── core/
│   │   ├── config.py       # Configuration
│   │   └── logging.py      # Logging setup
│   ├── models/
│   │   ├── database.py     # SQLAlchemy models
│   │   └── schemas.py      # Pydantic schemas
│   ├── services/
│   │   ├── database.py     # Database service
│   │   ├── redis.py        # Redis service
│   │   └── aiengine.py     # AI engine integration
│   └── main.py             # Application entry point
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
└── .env.example           # Environment variables template
```

## API Endpoints

### Files
- `POST /api/files/upload` - Upload tick data file
- `GET /api/files/list` - List uploaded files
- `DELETE /api/files/{file_id}` - Delete file

### Analysis
- `POST /api/analysis/start` - Start analysis job
- `GET /api/analysis/{job_id}/status` - Get job status
- `POST /api/analysis/{job_id}/cancel` - Cancel job
- `WS /api/analysis/ws/{job_id}` - WebSocket for real-time updates

### System
- `GET /api/system/status` - System resource usage
- `GET /api/system/health` - Health check

## Integration with aiend

The backend integrates with the existing aiend Python AI engine by:

1. Creating job configurations
2. Spawning Python subprocesses to run aiend scripts
3. Monitoring progress through Redis
4. Collecting results and making them available via API

## Database Schema

See `app/models/database.py` for full schema. Main tables:

- **files**: Uploaded file metadata
- **analyses**: Analysis job information
- **analysis_results**: Analysis results and metrics
