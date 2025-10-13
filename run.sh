#!/bin/bash

# Auto White Balance System - Run Script
# This script activates the virtual environment and runs the specified Python script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Auto White Balance System - Runner"
echo -e "==========================================${NC}"
echo ""

# Find the virtual environment
VENV_PATH=""
VENV_FOUND=false

# Check common venv locations
VENV_LOCATIONS=(
    "./venv"
    "./virtual_enviornment"
    "../venv"
    "../virtual_enviornment"
    "../../venv"
    "../../virtual_enviornment"
    "../../../venv"
    "../../../virtual_enviornment"
)

for venv_loc in "${VENV_LOCATIONS[@]}"; do
    if [ -d "$venv_loc" ]; then
        VENV_PATH="$venv_loc"
        VENV_FOUND=true
        echo -e "${GREEN}Found virtual environment: $venv_loc${NC}"
        break
    fi
done

# Activate virtual environment if found
if [ "$VENV_FOUND" = true ]; then
    echo "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
    
    # Check if activation was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Virtual environment activated${NC}"
    else
        echo -e "${RED}Warning: Failed to activate virtual environment${NC}"
        echo "Continuing with system Python..."
    fi
else
    echo -e "${BLUE}No virtual environment found in common locations${NC}"
    echo "Using system Python installation..."
    echo "To use a virtual environment, create one with:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
fi

echo ""

# Check which script to run
if [ "$1" == "demo" ]; then
    echo "Running demo..."
    python3 demo.py
elif [ "$1" == "test" ] || [ "$1" == "real" ]; then
    echo "Running real data test..."
    python3 test_real_data.py
elif [ "$1" == "system" ]; then
    echo "Running basic system test..."
    python3 awb_system.py
elif [ -n "$1" ]; then
    # Run custom script
    echo "Running custom script: $1"
    python3 "$1"
else
    # Default: show menu
    echo "Usage: ./run.sh [option]"
    echo ""
    echo "Options:"
    echo "  demo     - Run synthetic demo"
    echo "  test     - Test with real RAW data"
    echo "  real     - Test with real RAW data (same as test)"
    echo "  system   - Run basic system test"
    echo "  [file]   - Run custom Python file"
    echo ""
    echo "Example:"
    echo "  ./run.sh demo"
    echo "  ./run.sh test"
    echo "  ./run.sh my_script.py"
    echo ""
    
    # Interactive menu
    echo "Select an option:"
    echo "  1) Run demo"
    echo "  2) Test with real RAW data"
    echo "  3) Run basic system test"
    echo "  4) Exit"
    echo ""
    read -p "Enter your choice [1-4]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Running demo..."
            python3 demo.py
            ;;
        2)
            echo ""
            echo "Running real data test..."
            python3 test_real_data.py
            ;;
        3)
            echo ""
            echo "Running basic system test..."
            python3 awb_system.py
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Execution completed successfully"
    echo -e "==========================================${NC}"
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Execution failed with errors"
    echo -e "==========================================${NC}"
    exit 1
fi

