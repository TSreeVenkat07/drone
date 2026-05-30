@echo off
title UAV Search & Rescue - AirSim Wrapper
cd /d "%~dp0"
echo ====================================================
echo Launching Multi-Agent VDN AirSim Wrapper...
echo ====================================================
python airsim_wrapper.py
echo ====================================================
echo Execution completed.
pause
