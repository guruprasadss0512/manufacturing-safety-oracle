#!/bin/bash
# Manufacturing Safety Oracle — quick start script
cd ~/manufacturing-safety-oracle
source venv/bin/activate
echo "Starting Manufacturing Safety Oracle..."
echo "Open http://localhost:8501 in your browser"
streamlit run app/main.py --server.headless true
