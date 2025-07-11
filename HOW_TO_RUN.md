# 🚀 AIOpsLab - How to Run the Project

## Prerequisites Setup

### 1. Install Python 3.11+ (REQUIRED)

**Method 1: From Python.org (Recommended)**
1. Visit [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.11+ for Windows
3. Run installer and **IMPORTANT: Check "Add Python to PATH"**
4. Verify installation: Open new terminal and run `python --version`

**Method 2: Using Chocolatey**
```powershell
# Install Chocolatey first if you don't have it
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Python
choco install python --version=3.11.0
```

**Method 3: Using Windows Package Manager**
```powershell
winget install Python.Python.3.11
```

### 2. Install Poetry (Dependency Manager)

After Python is installed, install Poetry:

```powershell
# Method 1: Official installer (Recommended)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Method 2: Using pip
pip install poetry

# Method 3: Using pipx (if you have it)
pipx install poetry
```

**Add Poetry to PATH:**
- Add `%APPDATA%\Python\Scripts` to your PATH environment variable
- Or restart your terminal/VS Code

## Project Setup & Running

### Step 1: Verify Prerequisites
```powershell
# Check Python version (should be 3.11+)
python --version

# Check Poetry
poetry --version
```

### Step 2: Configure Poetry Environment
```powershell
# Set Poetry to use Python 3.11
poetry env use python3.11

# Or if you have a specific Python path
poetry env use C:\Python311\python.exe
```

### Step 3: Install Dependencies
```powershell
# Install all project dependencies
poetry install
```

### Step 4: Create Configuration
```powershell
# Copy example config
copy aiopslab\config.yml.example aiopslab\config.yml
```

### Step 5: Set Environment Variables (Optional)
```powershell
# For OpenAI API (if you want to use AI features)
$env:OPENAI_API_KEY = "your-openai-api-key-here"

# Make it permanent (optional)
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-key-here", "User")
```

## Running the Project

### Method 1: Using Poetry (Recommended)
```powershell
# Activate the virtual environment
poetry shell

# Run the CLI
poetry run python cli.py
```

### Method 2: Using VS Code Tasks
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select "Run AIOpsLab CLI"

### Method 3: Using VS Code Debug
1. Press `F5` to start debugging
2. Choose "Python: AIOpsLab CLI"

## Available Commands

Once the CLI is running, you can use these commands:

```
# Start a problem
start <problem_id>

# List available problems
# (The problems are defined in aiopslab/orchestrator/problems/registry.py)

# Exit the application
exit
```

## Running Tests
```powershell
# Run all tests
poetry run python -m pytest tests/ -v

# Or use VS Code task
# Ctrl+Shift+P → Tasks: Run Task → Run Tests
```

## Code Formatting and Type Checking
```powershell
# Format code
poetry run black .

# Type checking
poetry run pyright
```

## Troubleshooting

### Python Not Found
- Ensure Python 3.11+ is installed and in PATH
- Try `python --version` and `py --version`
- Restart your terminal/VS Code after installation

### Poetry Not Found
- Ensure Poetry is installed and in PATH
- Try `poetry --version`
- Add `%APPDATA%\Python\Scripts` to PATH

### Dependencies Issues
```powershell
# Clear cache and reinstall
poetry cache clear --all pypi
poetry install --no-cache
```

### Virtual Environment Issues
```powershell
# Remove and recreate environment
poetry env remove python
poetry env use python3.11
poetry install
```

## What the Project Does

AIOpsLab is a framework for:
- **Testing AIOps agents** in simulated environments
- **Fault injection** in microservices
- **Workload generation** for testing
- **Telemetry collection** and analysis
- **Kubernetes orchestration** for cloud environments

The CLI provides an interactive interface to:
- Start problem scenarios
- Interact with simulated environments
- Test AI agents for operations tasks
- Evaluate agent performance

## Next Steps

1. **Start with local setup** (no Azure needed)
2. **Use kind for Kubernetes** (local cluster)
3. **Set qualitative_eval: false** in config (no LLM calls)
4. **Explore the problems** in `aiopslab/orchestrator/problems/`
5. **Check the tutorial** in `TutorialSetup.md` for Kubernetes setup

Remember: You can run AIOpsLab completely locally without any cloud resources!
