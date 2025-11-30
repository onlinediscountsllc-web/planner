#!/bin/bash

# Life Planner Startup Script for Linux/Mac

echo "===================================="
echo "Life Planner Application Startup"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "[+] Virtual environment created"
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[!] .env file not found!"
    echo "[!] Copying .env.template to .env"
    cp .env.template .env
    echo ""
    echo "[!] IMPORTANT: Edit .env file with your configuration before continuing!"
    echo "[!] Press any key to exit and configure .env..."
    read -n 1
    exit 1
fi

# Install/update dependencies
echo "[*] Checking dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[!] Failed to install dependencies"
    exit 1
fi
echo "[+] Dependencies installed"

# Create logs directory
mkdir -p logs

# Check if database is initialized
python3 -c "from app import app, db; import os; app.app_context().push(); exit(0 if os.path.exists('life_planner.db') or bool(db.engine.url.database) else 1)" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[!] Database not initialized"
    echo "[*] Run 'python3 init_db.py' to set up the database"
    exit 1
fi

echo ""
echo "===================================="
echo "Starting Life Planner Application"
echo "===================================="
echo ""
echo "[*] Access at: http://localhost:5000"
echo "[*] Admin: onlinediscountsllc@gmail.com / admin8587037321"
echo "[*] Press Ctrl+C to stop"
echo ""

# Start the application
python3 app.py
