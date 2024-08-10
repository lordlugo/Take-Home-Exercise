FROM python:3.8-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    pkg-config \
    bash \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && rm -rf /var/lib/apt/lists/*

# Reload the environment variables
ENV PATH="/root/.cargo/bin:${PATH}"

# Ensure that Python is symlinked correctly
RUN ln -s /usr/bin/python3 /usr/bin/python


# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy

COPY Task_1.py Task_2.py Task_4.py ./
COPY README.md ./


# Running Task 1 as the default command
CMD ["python", "Task_1.py"]