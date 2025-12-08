#!/bin/bash

# 1. Get the directory where this script is located
cd "$(dirname "$0")"

# 2. Define a cleanup function to kill Python when you close the window
cleanup() {
    echo "🛑 Shutting down servers..."
    # Kill all child processes (Python & Node)
    pkill -P $$
    exit
}
# Run cleanup if the user hits Ctrl+C or closes the window
trap cleanup SIGINT SIGTERM EXIT

# 3. Start the Backend in the background
echo "-------------------------------------------------"
echo "🧠 STARTING AI AGENT (Server + Vector DB)..."
echo "-------------------------------------------------"
source venv/bin/activate
python server.py &

# 4. Wait for Python to load 
echo "⏳ Waiting 10s for Neural Networks to load..."
sleep 10

# 5. Start the Frontend (React)
echo "-------------------------------------------------"
echo "⚛️ STARTING Q..."
echo "-------------------------------------------------"
cd Q
npm run dev

# Keep script running to maintain the trap
wait
