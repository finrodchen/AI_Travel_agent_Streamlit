#!/usr/bin/env python3
"""
Quick start script for AI Travel Agent
Checks dependencies and starts the Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def check_file_exists(filename):
    """Check if required files exist"""
    if not Path(filename).exists():
        print(f"❌ Missing file: {filename}")
        return False
    print(f"✅ Found: {filename}")
    return True

def check_env_file():
    """Check if .env file exists and has required keys"""
    if not Path(".env").exists():
        print("⚠️  .env file not found!")
        print("📝 Creating sample .env file...")
        
        env_content = """# Add your API keys here
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
"""
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Sample .env file created!")
        print("🔧 Please add your actual API keys to the .env file")
        return False
    
    # Check if .env has actual keys (not placeholder text)
    with open(".env", "r") as f:
        content = f.read()
        if "your_" in content:
            print("⚠️  Please update your .env file with actual API keys")
            return False
    
    print("✅ .env file looks good!")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def start_streamlit():
    """Start the Streamlit app"""
    print("🚀 Starting AI Travel Agent...")
    print("🌐 Your app will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using AI Travel Agent!")
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit. Make sure it's installed.")

def main():
    print("🌍 AI Travel Agent - Quick Start")
    print("=" * 40)
    
    # Check required files
    required_files = ["app.py", "requirements.txt"]
    files_ok = all(check_file_exists(f) for f in required_files)
    
    if not files_ok:
        print("❌ Missing required files. Please check your project structure.")
        return
    
    # Check .env file
    env_ok = check_env_file()
    
    # Install requirements
    if not install_requirements():
        return
    
    if not env_ok:
        print("\n⚠️  API keys need to be configured in .env file")
        print("🔧 Edit the .env file with your actual API keys, then run this script again")
        return
    
    # Start the app
    print("\n🎉 Everything looks good!")
    start_streamlit()

if __name__ == "__main__":
    main()
