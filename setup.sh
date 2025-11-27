#!/bin/bash

# Setup script for Urban Sound Classification project

echo "🔊 Urban Sound Classification - Setup Script"
echo "============================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
python -c "from src.utils import create_directory_structure; create_directory_structure()"

# Create placeholder files
touch data/train/.gitkeep
touch data/test/.gitkeep
touch data/uploaded/.gitkeep
touch models/.gitkeep
touch models/backups/.gitkeep

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place your UrbanSound8K dataset in data/train/ and data/test/"
echo "2. Run the Jupyter notebook: jupyter notebook notebook/urbansound_pipeline.ipynb"
echo "3. Train the model by running all cells"
echo "4. Start the APIs:"
echo "   - Prediction: python -m uvicorn src.prediction:app --port 8000"
echo "   - Retraining: python -m uvicorn src.retrain:retrain_app --port 8001"
echo "5. Start the web UI: streamlit run src/app.py"
echo ""
echo "For Docker deployment:"
echo "   docker-compose up --build"
