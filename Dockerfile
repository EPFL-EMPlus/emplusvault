# Define the base image for building the environment
FROM python:3.9-slim as builder

RUN pip install poetry

# Set the working directory
WORKDIR /app

COPY pyproject.toml poetry.lock* ./
# Configure Poetry:
# - Disable virtual environments creation
# - Install dependencies only (no dev-dependencies)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Define the final image
FROM python:3.9-slim as runner

WORKDIR /app

# Copy the Python environment
COPY --from=builder /usr/local /usr/local

# Copy the project source code
COPY . .

# Run the tests
CMD ["poetry", "run", "pytest", "-v"]
