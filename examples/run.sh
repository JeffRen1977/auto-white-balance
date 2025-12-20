#!/bin/bash

# Auto White Balance System - Run Script
# This script activates the virtual environment and runs the specified Python script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}=========================================="
echo "Auto White Balance System - Runner"
echo -e "==========================================${NC}"
echo ""

# Find the virtual environment
# Expected location: ../virtual_env (one directory above AutoWhiteBalance)
VENV_PATH=""
VENV_FOUND=false

# Check venv locations (prioritize ../virtual_env)
VENV_LOCATIONS=(
    "$(dirname "$PROJECT_ROOT")/virtual_env"  # Primary location
    "$PROJECT_ROOT/venv"
    "$PROJECT_ROOT/virtual_env"
    "$(dirname "$PROJECT_ROOT")/venv"
    "$(dirname "$PROJECT_ROOT")/virtual_enviornment"
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
    echo -e "${BLUE}No virtual environment found${NC}"
    echo "Expected location: $(dirname "$PROJECT_ROOT")/virtual_env"
    echo ""
    echo "To create the virtual environment:"
    echo "  cd $(dirname "$PROJECT_ROOT")"
    echo "  python3 -m venv virtual_env"
    echo "  source virtual_env/bin/activate"
    echo "  cd AutoWhiteBalance"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Using system Python installation for now..."
fi

echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check which script to run
if [ "$1" == "basic" ] || [ "$1" == "example" ]; then
    echo "Running basic usage example..."
    python3 examples/basic_usage.py
elif [ "$1" == "test" ] || [ "$1" == "real" ]; then
    echo "Running real data test..."
    python3 tests/test_real_data.py
elif [ -n "$1" ]; then
    # Run custom script
    echo "Running custom script: $1"
    python3 "$1"
else
    # Default: show menu
    echo "Usage: ./run.sh [option]"
    echo ""
    echo "Options:"
    echo "  basic    - Run basic usage example"
    echo "  test     - Test with real RAW data"
    echo "  real     - Test with real RAW data (same as test)"
    echo "  [file]   - Run custom Python file"
    echo ""
    echo "Example:"
    echo "  ./run.sh basic"
    echo "  ./run.sh test"
    echo "  ./run.sh my_script.py"
    echo ""
    
    # Interactive menu
    echo "Select an option:"
    echo "  1) Run basic usage example"
    echo "  2) Test with real RAW data"
    echo "  3) Exit"
    echo ""
    read -p "Enter your choice [1-3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "Running basic usage example..."
            python3 examples/basic_usage.py
            ;;
        2)
            echo ""
            echo "Running real data test..."
            python3 tests/test_real_data.py
            ;;
        3)
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
