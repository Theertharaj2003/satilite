$ErrorActionPreference = "Stop"
Write-Host "Creating Python 3.11 virtual environment..."
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Setup complete. Activate with: .\\.venv\\Scripts\\Activate.ps1"
