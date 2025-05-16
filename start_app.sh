#!/bin/bash

# Set required LangGraph environment variables
export LANGGRAPH_RUNTIME_EDITION=inmem
export BG_JOB_ISOLATED_LOOPS=true

# Activate virtual environment
source myenv/bin/activate

# Install the package in development mode if needed
# pip install -e .

echo "Starting Executive AI Assistant with proper configuration..."
echo "Using isolated loops for blocking operations."
echo "Using in-memory runtime edition."
echo "Press Ctrl+C to stop the application."

# Start LangGraph with necessary flags
python -m langgraph_runtime_api --allow-blocking 