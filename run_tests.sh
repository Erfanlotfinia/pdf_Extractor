#!/bin/bash
set -e

echo "Setting PYTHONPATH..."
export PYTHONPATH=.

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests..."
pytest tests/
