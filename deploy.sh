#!/bin/bash

# Auto-run script for advanced correlation analysis with git deployment
# Runs the analysis with default stocks and 3-year lookback, then pushes to git

echo "🚀 Starting Advanced Correlation Analysis Auto-Deploy"
echo "=========================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Run the advanced correlation analysis with 3-year lookback
echo ""
echo "📊 Step 1: Running Advanced Correlation Analysis (3-year lookback)"
python3 -c "from advanced_correlation_analysis import main; main(use_defaults=True)"

# Check if the analysis was successful
if [ $? -ne 0 ]; then
    echo "❌ Analysis failed. Stopping deployment."
    exit 1
fi

# Check if index.html was created
if [ ! -f "index.html" ]; then
    echo "❌ index.html was not created. Stopping deployment."
    exit 1
fi

echo "✅ Analysis completed and index.html generated"

# Step 2: Git operations
echo ""
echo "📦 Step 2: Git Operations"

# Initialize git if not already a repository
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    if [ $? -ne 0 ]; then
        echo "❌ Git initialization failed"
        exit 1
    fi
fi

# Add files to git
echo "🔄 Adding files to git..."
git add .
if [ $? -ne 0 ]; then
    echo "❌ Failed to add files to git"
    exit 1
fi

# Create commit with timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMMIT_MESSAGE="Auto-update: Stock analysis report - $TIMESTAMP"

echo "🔄 Creating git commit..."
git commit -m "$COMMIT_MESSAGE"
COMMIT_RESULT=$?

if [ $COMMIT_RESULT -eq 0 ]; then
    echo "✅ Git commit created successfully"
elif [ $COMMIT_RESULT -eq 1 ]; then
    # Check if there are no changes to commit
    if [ -z "$(git status --porcelain)" ]; then
        echo "ℹ️  No changes to commit - repository is up to date"
    else
        echo "❌ Git commit failed"
        exit 1
    fi
else
    echo "❌ Git commit failed"
    exit 1
fi

# Check if remote origin exists
if ! git remote -v | grep -q "origin"; then
    echo "⚠️  No remote 'origin' found. Please set up your remote repository:"
    echo "   git remote add origin <your-repo-url>"
    echo "   Then run this script again."
    exit 1
fi

# Get current branch name
CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
    CURRENT_BRANCH="main"
fi

# Push to remote
echo "🔄 Pushing to remote repository..."
git push origin "$CURRENT_BRANCH"
if [ $? -ne 0 ]; then
    echo "🔄 Trying to set upstream and push..."
    git push -u origin "$CURRENT_BRANCH"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to push to remote repository"
        exit 1
    fi
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "=========================================================="
echo "📊 Analysis report generated: index.html"
echo "🌐 Repository updated and pushed to git"
echo "🔗 Your analysis is now live in your git repository"
