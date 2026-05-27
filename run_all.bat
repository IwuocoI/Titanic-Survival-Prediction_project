@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   Titanic Survival Prediction - Auto Run
echo ========================================
echo.

cd code
set PYTHONIOENCODING=utf-8

echo [1/8] data_processing.py ...
python data_processing.py
if %errorlevel% neq 0 (
    echo FAILED: data_processing.py
    pause
    exit /b 1
)
echo OK
echo.

echo [2/8] model_logistic.py (baseline LR) ...
python model_logistic.py
if %errorlevel% neq 0 (
    echo FAILED: model_logistic.py
    pause
    exit /b 1
)
echo OK
echo.

echo [3/8] model_rf.py (baseline RF) ...
python model_rf.py
if %errorlevel% neq 0 (
    echo FAILED: model_rf.py
    pause
    exit /b 1
)
echo OK
echo.

echo [4/8] model_xgboost.py (baseline XGB) ...
python model_xgboost.py
if %errorlevel% neq 0 (
    echo FAILED: model_xgboost.py
    pause
    exit /b 1
)
echo OK
echo.

echo [5/8] exp_feature.py ...
python exp_feature.py
if %errorlevel% neq 0 (
    echo FAILED: exp_feature.py
    pause
    exit /b 1
)
echo OK
echo.

echo [6/8] exp.ablation.py ...
python exp.ablation.py
if %errorlevel% neq 0 (
    echo FAILED: exp.ablation.py
    pause
    exit /b 1
)
echo OK
echo.

echo [7/8] exp_hybrid.py ...
python exp_hybrid.py
if %errorlevel% neq 0 (
    echo FAILED: exp_hybrid.py
    pause
    exit /b 1
)
echo OK
echo.

echo [8/8] visualization.py ...
set MPLBACKEND=Agg
python visualization.py
if %errorlevel% neq 0 (
    echo FAILED: visualization.py
    pause
    exit /b 1
)
echo OK
echo.

echo ========================================
echo ALL DONE!
echo Results: ..\results\vis\
echo Kaggle predictions: ..\results\kaggle\
echo ========================================
pause
