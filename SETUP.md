# AIOpsLab Setup Guide

## Prerequisites Installation

### For Windows:

1. **Install Python 3.11+**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use Windows Store: `winget install Python.Python.3.11`
   - Or use Chocolatey: `choco install python --version=3.11.0`

2. **Install Poetry**:
   ```powershell
   # Method 1: Official installer
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   
   # Method 2: Using pip
   pip install poetry
   
   # Method 3: Using pipx (recommended)
   pip install pipx
   pipx install poetry
   ```

3. **Add Poetry to PATH**:
   - Add `%APPDATA%\Python\Scripts` to your PATH environment variable
   - Or restart your terminal/VS Code

## Project Setup

1. **Configure Poetry to use Python 3.11**:
   ```bash
   poetry env use python3.11
   ```

2. **Install project dependencies**:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Usage

### Running the CLI
```bash
poetry run python cli.py
```

### Running Tests
```bash
poetry run python -m pytest tests/ -v
```

### Code Formatting
```bash
poetry run black .
```

### Type Checking
```bash
poetry run pyright
```

## VS Code Integration

The workspace is configured with:
- **Tasks**: Use `Ctrl+Shift+P` → "Tasks: Run Task" to run predefined tasks
- **Debug**: Use `F5` to start debugging the CLI or current file
- **Extensions**: Python, Black Formatter, Pylint, and Kubernetes tools are installed

## Next Steps

1. Install Python 3.11+ and Poetry following the instructions above
2. Run `poetry install` to set up the project
3. Review the `README.md` for detailed project information
4. Check `TutorialSetup.md` for additional setup instructions
5. Explore the `aiopslab/` directory for core functionality
