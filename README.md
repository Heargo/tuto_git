This small machine-learning proof-of-concept is set up to run inside a Python virtual environment (venv).

Quick start (Windows PowerShell):

1. Create and activate the venv

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run (as admin) to allow the activate script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
``` 

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run the proof-of-concept

```powershell
python poc.py
```

Notes:
- The script expects a `data.csv` file in the same folder.
- If you prefer conda, create a conda env and install the packages from `requirements.txt` or translate into an `environment.yml`.
