@echo off
echo Starting AIGVDet GUI...
echo.
echo Checking if streamlit is installed...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing required packages...
    pip install streamlit torch torchvision opencv-python numpy pillow natsort tqdm
)

echo.
echo Starting GUI on http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run gui_app.py
pause
