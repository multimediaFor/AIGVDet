@echo off
REM Run AIGVDet GUI in Docker (Windows)

echo Starting AIGVDet GUI...
echo Installing Streamlit and starting server...
echo Access the GUI at: http://localhost:8501
echo.

docker run --gpus all -p 8501:8501 -it ^
  -v %cd%/output_data:/app/output_data ^
  -v %cd%/checkpoints:/app/checkpoints ^
  -v %cd%/raft-model:/app/raft-model ^
  sacdalance/thesis-aigvdet:latest-gpu ^
  streamlit run gui_app.py --server.address=0.0.0.0
