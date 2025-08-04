# EMPLUS Vault

A comprehensive data vault for digital archives that processes, analyzes, and visualizes media content. EMPLUS Vault extracts features from video, audio, and text content, stores them in a searchable database, and provides API access to this data.

## System Architecture

EMPLUS Vault consists of the following components:

1. **Core Backend**
   - FastAPI web server provides REST API access
   - SQLAlchemy with PostgreSQL (pgvector) for data storage
   - S3-compatible storage (MinIO) for media files

2. **Feature Extraction**
   - Audio analysis (speech detection, speaker identification)
   - Image processing (pose detection, visual features)
   - Text analysis (transcripts, semantic features)
   
3. **Processing Pipelines**
   - Media ingestion and normalization
   - Feature extraction and indexing
   - Data transformation workflows

4. **Web Frontend**
   - Interactive visualization using Cables.gl
   - Data exploration interface

## Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Poetry
- Access to media sources

### Option 1: CLI Tools Only
This requires the installation of poetry as a first step:

1. Install Poetry (https://python-poetry.org/docs/)
2. Clone this repository
3. Install dependencies:
   ```
   poetry install
   poe force-torch
   ```

### Option 2: Full Infrastructure

The system requires two environments:
- Docker environment for database and storage
- Python environment for application code

1. **Set up infrastructure with Docker:**
   ```
   cd docker
   cp .env.example .env  # Edit with your credentials
   docker compose up -d
   ```

2. **Configure application environment:**
   ```
   cd ..  # Back to main directory
   cp .env.example .env  # Edit with your credentials
   poetry install
   poe force-torch
   poe init-db  # Initialize the database schema
   ```

## Usage

### Run the API Server

```
poe serve-dev  # Development mode with auto-reload
```
or
```
poe serve-prod  # Production mode
```

Server will be available at http://localhost:8763

### Run Processing Pipelines

```
poe run-pipeline --source=rts  # Run the RTS pipeline
```

### Using the CLI

```
poetry run python -m emv.cli --help
```

## API Endpoints

The API is documented using OpenAPI and can be accessed at http://localhost:8763/docs when the server is running.

Key endpoints:
- `/api/v1/media` - Media management
- `/api/v1/features` - Feature access
- `/api/v1/library` - Content library
- `/api/v1/projections` - Data projections and visualization

## Development

### Testing

Running tests requires the test database to be running:

```
cd docker/tests
docker compose up -d
cd ../..
pytest tests  # Run all tests
pytest tests/test_features.py  # Run specific test file
pytest tests/test_features.py::test_function_name  # Run specific test
```

### Database Migrations

Database migrations are managed with Alembic:

```
# Change either the model definition or do a manual change in the migration file later (see db/versions/). This is the preferred way. 
nano emv/api/models.py

# Create a new migration file. This will have the new definition of the changes if models.py was changed. Otherwise add your changes here (also found in db/versions/).
alembic revision -m "description of change"

# Apply migrations to the db. 
alembic upgrade head
```

### Frontend Development

The frontend uses Cables.gl:

1. Install requirements:
   ```
   # Install NVM
   nvm install --lts
   npm install -g @cables/cables
   ```
   
2. Create API key at https://cables.gl/settings
3. Create `~/.cablesrc` file with `apikey=YOUR_API_KEY`

## Deployment

For Kubernetes deployment, see the `kubernetes/` directory.

## Troubleshooting

- **Poetry issues with Keyring**: Try `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
- **Network drive mounting**: 
  ```
  sudo mount -t drvfs '\\KNAS\mjf' /mnt/mjf
  sudo mount -t drvfs '\\emplussrv1.epfl.ch\EMPLUS-Network\RTS-Data' /mnt/rts
  ```

## Environment Variables

Required environment variables (see `.env.example` for template):

- **Database**: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- **S3 Storage**: `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_ENDPOINT`, `BUCKET_NAME`
- **Authentication**: `SECRET_KEY` (for JWT tokens), `SUPERUSER_CLI_KEY`
- **Optional**: `HF_TOKEN` (for HuggingFace models), `BASE_PATH` (for data storage)

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{emv2025,
  title = {EMV - EPFL Media Vault},
  author = {Rattinger, André and Benzi, Kirell and Alliata, Giacomo},
  year = {2025},
  institution = {École Polytechnique Fédérale de Lausanne (EPFL)},
  url = {https://github.com/EPFL-EMPlus/emv}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
