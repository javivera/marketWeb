#!/bin/bash

# Run Monte Carlo Portfolio Simulation
# This script runs the Monte Carlo simulation using settings from .env file

echo "🎲 Starting Monte Carlo Portfolio Simulation"
echo "============================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the Monte Carlo simulation
echo "📊 Running Monte Carlo analysis..."
python3 monte_carlo_portfolio.py

# Check if the simulation was successful
if [ $? -eq 0 ]; then
    echo "✅ Monte Carlo simulation completed successfully!"
    echo "📄 Results saved to HTML report"
    
    # Check if we should also commit to git
    if [ -f ".env" ]; then
        AUTO_GIT_PUSH=$(grep "AUTO_GIT_PUSH" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ "$AUTO_GIT_PUSH" = "true" ]; then
            echo "📦 Auto-committing to git..."
            git add .
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            git commit -m "Auto-update: Monte Carlo simulation - $TIMESTAMP"
            git push origin main
            echo "🚀 Results pushed to git repository"
        fi
    fi
else
    echo "❌ Monte Carlo simulation failed"
    exit 1
fi

echo "🎉 Process completed!"
