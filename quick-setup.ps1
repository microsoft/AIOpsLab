#!/usr/bin/env powershell
# Quick Setup Script for AIOpsLab on Windows

Write-Host "🚀 AIOpsLab Quick Setup" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

# Check if Python is installed
Write-Host "`n1. Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.1\d+") {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python 3.11+ required. Current: $pythonVersion" -ForegroundColor Red
        Write-Host "Please install Python 3.11+ from https://python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+ from https://python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check if Poetry is installed
Write-Host "`n2. Checking Poetry installation..." -ForegroundColor Yellow
try {
    $poetryVersion = poetry --version 2>&1
    Write-Host "✅ Poetry found: $poetryVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Poetry not found. Installing Poetry..." -ForegroundColor Red
    Write-Host "Installing Poetry via pip..." -ForegroundColor Yellow
    python -m pip install poetry
    
    # Verify installation
    try {
        $poetryVersion = poetry --version 2>&1
        Write-Host "✅ Poetry installed: $poetryVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Poetry installation failed. Please install manually." -ForegroundColor Red
        exit 1
    }
}

# Configure Poetry environment
Write-Host "`n3. Configuring Poetry environment..." -ForegroundColor Yellow
poetry env use python

# Install dependencies
Write-Host "`n4. Installing project dependencies..." -ForegroundColor Yellow
poetry install

# Create config file
Write-Host "`n5. Creating configuration file..." -ForegroundColor Yellow
if (-Not (Test-Path "aiopslab\config.yml")) {
    Copy-Item "aiopslab\config.yml.example" "aiopslab\config.yml"
    Write-Host "✅ Configuration file created" -ForegroundColor Green
} else {
    Write-Host "✅ Configuration file already exists" -ForegroundColor Green
}

# Setup complete
Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "To run AIOpsLab:" -ForegroundColor Cyan
Write-Host "  poetry run python cli.py" -ForegroundColor White
Write-Host ""
Write-Host "Or activate the environment first:" -ForegroundColor Cyan
Write-Host "  poetry shell" -ForegroundColor White
Write-Host "  python cli.py" -ForegroundColor White
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Cyan
Write-Host "  poetry run python -m pytest tests/ -v" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🎯" -ForegroundColor Green
