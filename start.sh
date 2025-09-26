#!/bin/bash
# PDFHikaku Application Launcher

export PYTHONPATH="/home/vscode/.local/lib/python3.11/site-packages:$PYTHONPATH"

echo "🚀 Starting PDFHikaku Application..."
echo "📦 Setting up Python path..."

streamlit run pdfhikaku.py --server.port 8502 --server.address 0.0.0.0