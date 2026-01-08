#!/bin/bash
# One-click installation script
# Usage: bash install.sh

set -e  # Exit on error

echo "=========================================="
echo "Exoplanet Atmosphere Retrieval Framework"
echo "Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check pip
echo ""
echo "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found, please install pip first"
    exit 1
fi

# Install dependencies
echo ""
echo "Starting installation..."
echo "----------------------------------------"

# Method 1: Install using setup.py (recommended, editable mode)
if [ "$1" == "--editable" ] || [ "$1" == "-e" ]; then
    echo "Installing in editable mode..."
    pip3 install -e .
else
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    echo ""
    echo "Note: Dependencies installed, but package not installed in editable mode."
    echo "To enable immediate code changes, run: pip3 install -e ."
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure MultiNest C library is installed (required for pymultinest)"
echo "2. Make sure petitRADTRANS is installed"
echo "3. Run example: python simple_retrieval.py"
echo ""
echo "For detailed instructions, see INSTALL.md"

