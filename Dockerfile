# Define the base image for building the environment
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Define the final image
FROM python:3.9-slim as runner

WORKDIR /app

# Copy the Python environment
COPY --from=builder /usr/local /usr/local

# Copy the project source code
COPY . .

# Run the tests
CMD ["pytest", "-v"]
