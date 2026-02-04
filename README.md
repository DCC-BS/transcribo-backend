# TODO
returns library
api error from backend common


# Transcribo Backend

Transcribo Backend is a powerful Python FastAPI service that provides advanced audio and video transcription capabilities with speaker diarization and AI-powered text summarization. This backend service enables high-quality transcription using OpenAI's Whisper API and intelligent summarization using large language models.

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/DCC-BS/transcribo-backend)](https://img.shields.io/github/license/DCC-BS/transcribo-backend)

---

<p align="center">
  <a href="https://dcc-bs.github.io/documentation/">DCC Documentation & Guidelines</a> | <a href="https://www.bs.ch/daten/databs/dcc">DCC Website</a>
</p>

---

## Features

- **Audio & Video Transcription**: High-quality transcription of audio and video files using OpenAI's Whisper API
- **Speaker Diarization**: Identify and separate different speakers in recordings
- **Language Detection**: Automatic language detection or specify the source language
- **AI Summarization**: Generate intelligent summaries of transcribed text using LLMs
- **Asynchronous Processing**: Task-based processing with status tracking for long-running transcriptions
- **Multi-format Support**: Handle various audio formats (MP3, WAV, etc.) and video files
- **Audio Conversion**: Automatic conversion to MP3 format for optimal processing
- **Privacy-Focused**: Pseudonymized user tracking for usage analytics

## Technology Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) with Python 3.12+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Transcription**: OpenAI Whisper API integration
- **AI Models**: LLM integration for text summarization
- **Audio Processing**: Audio format conversion with audioop
- **Logging**: Structured logging with structlog
- **Containerization**: Docker and Docker Compose

## Setup

### Prerequisites

- Python 3.12+
- uv package manager
- Docker and Docker Compose (for containerized deployment)
- Access to OpenAI Whisper API or compatible service
- LLM API access for summarization features

### Environment Configuration

Create a `.env` file in the project root with the required environment variables:

```env
# Whisper API Configuration
WHISPER_API=http://localhost:8001
WHISPER_API_KEY=your_whisper_api_key_here

# LLM API Configuration
LLM_API=http://localhost:8002
LLM_API_KEY=your_llm_api_key_here

# Security
HMAC_SECRET=your_secret_key_here

# Client Configuration (optional)
CLIENT_PORT=3000
CLIENT_URL=http://localhost:${CLIENT_PORT}
```

> **Note:** Configure the Whisper API and LLM API endpoints to match your deployment setup.

### Install Dependencies

Install dependencies using uv:

```bash
make install
```

This will:
- Create a virtual environment using uv
- Install all dependencies
- Install pre-commit hooks

## Development

### Start the Development Server

```bash
uv run fastapi dev ./src/transcribo_backend/app.py
```

Or use the provided task:

```bash
make dev
```

### Code Quality Tools

Run code quality checks:

```bash
# Run all quality checks
make check

# Format code with ruff
uv run ruff format .

# Run linting
uv run ruff check .

# Run type checking
uv run pyrefly check
```

## Production

Run the production server:

```bash
make run
```

## Docker Deployment

The application includes a Dockerfile and Docker Compose configuration for easy deployment:

### Using Docker Compose

```bash
# Start all services with Docker Compose
docker compose up -d

# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f
```

### Using Dockerfile Only

```bash
# Build the Docker image
docker build -t transcribo-backend .

# Run the container
docker run --rm --env-file .env -p 8000:8000 transcribo-backend
```

## Testing & Development Tools

Run tests with pytest:

```bash
# Run tests
make test

# Run tests with pytest directly
uv run pytest
```

## API Endpoints

### Transcription

- **POST `/transcribe`**: Submit an audio or video file for transcription
  - Parameters:
    - `audio_file`: The audio/video file to transcribe
    - `num_speakers` (optional): Number of speakers for diarization
    - `language` (optional): Source language code
  - Returns: Task status with task ID for tracking

- **GET `/task/{task_id}/status`**: Get the status of a transcription task
  - Returns: Current task status (pending, processing, completed, failed)

- **GET `/task/{task_id}/result`**: Get the transcription result
  - Returns: Transcription response with text and metadata

### Summarization

- **POST `/summarize`**: Generate an AI summary of transcribed text
  - Body: `SummaryRequest` with transcript text
  - Returns: Generated summary

### Health Checks

- **GET `/health/liveness`**: Liveness probe for Kubernetes deployments
  - Returns: Application status and uptime

## Project Architecture

```
src/transcribo_backend/
├── app.py                      # FastAPI application entry point
├── config.py                   # Configuration management
├── helpers/                    # Helper utilities
│   └── file_type.py           # File type detection
├── models/                     # Data models and schemas
│   ├── progress.py            # Progress tracking models
│   ├── response_format.py     # Response format definitions
│   ├── summary.py             # Summary models
│   ├── task_status.py         # Task status models
│   └── transcription_response.py  # Transcription response models
├── services/                   # Business logic services
│   ├── audio_converter.py     # Audio format conversion
│   ├── summary_service.py     # Text summarization service
│   └── whisper_service.py     # Whisper API integration
└── utils/                      # Utility functions
    ├── logger.py              # Logging configuration
    └── usage_tracking.py      # Privacy-focused usage analytics
```

## Acknowledgments

This application is based on [Transcribo](https://github.com/machinelearningZH/audio-transcription/) from the Statistical Office of the Canton of Zurich. We have rewritten the functionality of the original application to fit into a modular and modern web application that separates frontend, backend and AI models.

## License

[MIT](LICENSE) © Data Competence Center Basel-Stadt

---

<a href="https://www.bs.ch/schwerpunkte/daten/databs/schwerpunkte/datenwissenschaften-und-ki"><img src="https://github.com/DCC-BS/.github/blob/main/_imgs/databs_log.png?raw=true" alt="DCC Logo" width="200" /></a>

**Datenwissenschaften und KI**
Developed with ❤️ by DCC - Data Competence Center
