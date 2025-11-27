# PowerShell Setup Script for Urban Sound Classification project

Write-Host "🔊 Urban Sound Classification - Setup Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create directory structure
Write-Host "Creating directory structure..." -ForegroundColor Yellow
python -c "from src.utils import create_directory_structure; create_directory_structure()"

# Create placeholder files
New-Item -ItemType File -Path "data\train\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "data\test\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "data\uploaded\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "models\.gitkeep" -Force | Out-Null
New-Item -ItemType File -Path "models\backups\.gitkeep" -Force | Out-Null

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Place your UrbanSound8K dataset in data\train\ and data\test\" -ForegroundColor White
Write-Host "2. Run the Jupyter notebook: jupyter notebook notebook\urbansound_pipeline.ipynb" -ForegroundColor White
Write-Host "3. Train the model by running all cells" -ForegroundColor White
Write-Host "4. Start the APIs:" -ForegroundColor White
Write-Host "   - Prediction: python -m uvicorn src.prediction:app --port 8000" -ForegroundColor White
Write-Host "   - Retraining: python -m uvicorn src.retrain:retrain_app --port 8001" -ForegroundColor White
Write-Host "5. Start the web UI: streamlit run src\app.py" -ForegroundColor White
Write-Host ""
Write-Host "For Docker deployment:" -ForegroundColor Cyan
Write-Host "   docker-compose up --build" -ForegroundColor White
