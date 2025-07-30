#!/usr/bin/env python3
"""
Auto-run script for advanced correlation analysis with git deployment
Runs the analysis with default stocks and 3-year lookback, then pushes to git
"""

import subprocess
import sys
from pathlib import Path
import os
from datetime import datetime

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 Starting Advanced Correlation Analysis Auto-Deploy")
    print("=" * 60)
    
    # Get current directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Step 1: Run the advanced correlation analysis
    print("\n📊 Step 1: Running Advanced Correlation Analysis")
    
    # Create a temporary script to run the analysis with default inputs
    temp_script = '''
import sys
from unittest.mock import patch
from advanced_correlation_analysis import main

# Mock input to provide default values (3 years = 1095 days)
inputs = iter(["1095"])
with patch("builtins.input", lambda prompt: next(inputs)):
    main()
'''
    
    # Write temporary script
    with open("temp_run_analysis.py", "w") as f:
        f.write(temp_script)
    
    analysis_command = "python3 temp_run_analysis.py"
    
    if not run_command(analysis_command, "Advanced Correlation Analysis"):
        print("❌ Analysis failed. Stopping deployment.")
        # Clean up temporary file
        if Path("temp_run_analysis.py").exists():
            Path("temp_run_analysis.py").unlink()
        return False
    
    # Clean up temporary file
    if Path("temp_run_analysis.py").exists():
        Path("temp_run_analysis.py").unlink()
    
    # Step 2: Check if index.html was created
    if not Path("index.html").exists():
        print("❌ index.html was not created. Stopping deployment.")
        return False
    
    print("✅ Analysis completed and index.html generated")
    
    # Step 3: Git operations
    print("\n📦 Step 2: Git Operations")
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("🔧 Initializing git repository...")
        if not run_command("git init", "Git initialization"):
            return False
    
    # Add files to git
    if not run_command("git add .", "Adding files to git"):
        return False
    
    # Create commit with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Auto-update: Stock analysis report - {timestamp}"
    
    if not run_command(f'git commit -m "{commit_message}"', "Creating git commit"):
        # Check if there are no changes to commit
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️  No changes to commit - repository is up to date")
        else:
            return False
    
    # Check if remote origin exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if "origin" not in result.stdout:
        print("⚠️  No remote 'origin' found. Please set up your remote repository:")
        print("   git remote add origin <your-repo-url>")
        print("   Then run this script again.")
        return False
    
    # Push to remote
    if not run_command("git push origin main", "Pushing to remote repository"):
        # Try to set upstream if needed
        if not run_command("git push -u origin main", "Setting upstream and pushing"):
            return False
    
    print("\n🎉 Deployment completed successfully!")
    print("=" * 60)
    print("📊 Analysis report generated: index.html")
    print("🌐 Repository updated and pushed to git")
    print("🔗 Your analysis is now live in your git repository")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
