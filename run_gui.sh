#!/bin/bash
# Run AIGVDet GUI locally (without Docker)

echo "Starting AIGVDet GUI..."
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing required packages..."
    pip install streamlit torch torchvision opencv-python numpy pillow natsort tqdm
fi

echo ""
echo "Starting GUI on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run gui_app.py
