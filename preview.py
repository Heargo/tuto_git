import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# --------------------------------------------------
# 1. Load CSV (try multiple encodings to avoid UnicodeDecodeError)
# --------------------------------------------------

def load_csv_with_fallback(path):
	encodings = ["utf-8", "cp1252", "latin-1"]
	last_exc = None
	for enc in encodings:
		try:
			df = pd.read_csv(path, encoding=enc)
			print(f"✅ Data loaded using encoding: {enc}")
			return df
		except UnicodeDecodeError as e:
			last_exc = e
			print(f"Failed to read with encoding {enc}: {e}")
		except Exception as e:
			# If pandas raises a different error (e.g. parser error), re-raise
			raise

	# Last resort: open text with replacement of invalid bytes and pass file handle to pandas
	try:
		with open(path, "r", encoding="utf-8", errors="replace") as fh:
			df = pd.read_csv(fh)
			print("✅ Data loaded using 'utf-8' with errors='replace'")
			return df
	except Exception:
		# If still failing, raise the original Unicode error for visibility
		raise last_exc or RuntimeError("Failed to read CSV with fallback encodings")


df = load_csv_with_fallback("./data/recipeData.csv")
print("✅ Data loaded")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Preview of data:")
print(df.head())

# log  5 example of values for TARGET_COLUMNS separated by comma
TARGET_COLUMNS = ["Color", "MashThickness"]
for col in TARGET_COLUMNS:
    print(f"Examples for target column '{col}': {df[col].dropna().unique()[:5].tolist()}")
