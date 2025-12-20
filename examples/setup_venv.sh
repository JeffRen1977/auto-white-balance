#!/bin/bash

# Auto White Balance System - Virtual Environment Setup Script
# This script creates a virtual environment and installs all dependencies

echo "=========================================="
echo "Auto White Balance System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demo:"
echo "  python3 demo.py"
echo ""
echo "To test with real data:"
echo "  python3 test_real_data.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""

