# rossmann-pharma-sales-prediction

A machine learning solution to forecast sales for Rossmann Pharmaceuticals' stores across various cities six weeks in advance. Factors like promotions, competition, holidays, seasonality, and locality are considered for accurate predictions.
The project structure is organized to support reproducible and scalable data processing, modeling, and visualization.

## Project Structure

```plaintext
├── .dvc/
│   └── config                        # Configuration files for data version control
├── .vscode/
│   └── settings.json                 # Configuration for VSCode environment
├── .github/
│   └── workflows/
│       ├── unittests.yml             # GitHub Actions workflow for running unit tests
├── .gitignore                        # Files and directories to be ignored by Git
├── requirements.txt                  # List of dependencies for the project
├── README.md                         # Project overview and instructions
├── scripts/
│   ├── __init__.py
│   ├── data_processing.py            # Script for data cleaning and processing
│   ├── data_visualization.py         # Scritpt for different plots
│   ├── load_data.py                  # Scritpt extracting and loading dataset
│   ├── hypothesis_testing.ipynb      # Script for hypothesis testing analysis
├── notebooks/
│   ├── __init__.py
│   ├── eda_notebook.ipynb            # Jupyter notebook for eda analysis
│   ├── hypothesis_testing.ipynb      # Jupyter notebook for hypothesis testing analysis
│   ├── data_preprocessing.ipynb      # Jupyter notebook for data preprocessing
│   ├── model_training.ipynb          # Jupyter notebook for statistical model training
│   ├── README.md                     # Description of notebooks
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py          # Unit tests for data processing module
│   
└── src/
    ├── __init__.py
    └── README.md                     # Description of scripts
```
# Installation

>>> git clone https://github.com/epythonlab/rossman-pharma-sales-prediction.git

>> cd rossman-pharma-sales-prediction

### Create virtual environment

>>> python3 -m venv venv # on MacOs or Linux

>>> source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install Dependencies

>>> pip install -r requirements.txt

## To run tests
navigate 
>>> cd tests/

>>pytest # all tests will be tested

