#!/bin/bash

# Auto White Balance System - Virtual Environment Setup Script
# This script creates a virtual environment in ../virtual_env and installs dependencies

echo "=========================================="
echo "Auto White Balance System - Setup"
echo "=========================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
VENV_PATH="$PARENT_DIR/virtual_env"

# Check Python version
echo "Checking Python version..."
python3 --version

# Check if virtual environment already exists
if [ -d "$VENV_PATH" ]; then
    echo ""
    echo "Virtual environment already exists at: $VENV_PATH"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    else
        echo "Using existing virtual environment."
        echo ""
        echo "To activate:"
        echo "  source $VENV_PATH/bin/activate"
        echo "  cd $PROJECT_ROOT"
        exit 0
    fi
fi

# Create virtual environment in parent directory
echo ""
echo "Creating virtual environment at: $VENV_PATH"
python3 -m venv "$VENV_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from $PROJECT_ROOT/requirements.txt..."
cd "$PROJECT_ROOT"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Virtual environment location: $VENV_PATH"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $PROJECT_ROOT"
echo ""
echo "Or use the run script:"
echo "  ./examples/run.sh test"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
