import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import minimize
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class InverseModel:
    def __init__(self, csv_path):
        try:
            # encoding='latin-1' handles special characters common in recipe datasets
            self.df = pd.read_csv(csv_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Error: File '{csv_path}' not found.")
            exit()
            
        self.encoders = {}
        self.feature_cols = []
        self.target_cols = []
        self.model = None
        self.X_bounds = []

        print(f"Data loaded. Shape: {self.df.shape}")

    def preprocess_data(self):
        """
        Robust preprocessing that handles mixed types and missing values safely.
        """
        # --- STEP 1: DEFINE TARGETS & FEATURES ---
        print(f"\nAvailable columns: {list(self.df.columns)}")
        target_input = input("Enter the target column(s) separated by comma (e.g., OG,ABV): ")
        self.target_cols = [t.strip() for t in target_input.split(',')]

        # Remove BeerID, Name, URL cols
        for col in ['BeerID', 'Name', 'URL']:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        
        # Validation
        for t in self.target_cols:
            if t not in self.df.columns:
                raise ValueError(f"Column '{t}' not found.")

        # Features are everything else
        self.feature_cols = [c for c in self.df.columns if c not in self.target_cols]

        # --- STEP 2: HANDLE MISSING VALUES (The Fix) ---
        print("Cleaning data...")
        # Create a copy to avoid SettingWithCopy warnings
        self.data_encoded = self.df.copy()

        for col in self.data_encoded.columns:
            # Check if column is numeric
            is_numeric = pd.api.types.is_numeric_dtype(self.data_encoded[col])
            
            if self.data_encoded[col].isnull().any():
                if is_numeric:
                    # Fill numbers with Mean
                    self.data_encoded[col] = self.data_encoded[col].fillna(self.data_encoded[col].mean())
                else:
                    # Fill text with Mode (most common value)
                    if not self.data_encoded[col].mode().empty:
                        fill_val = self.data_encoded[col].mode()[0]
                    else:
                        fill_val = "Unknown" # Fallback if empty
                    self.data_encoded[col] = self.data_encoded[col].fillna(fill_val)

        # --- STEP 3: ENCODE CATEGORICAL DATA ---
        for col in self.feature_cols + self.target_cols:
            # If not numeric, encode it
            if not pd.api.types.is_numeric_dtype(self.data_encoded[col]):
                le = LabelEncoder()
                # Convert to string first to handle any mixed types safeley
                self.data_encoded[col] = le.fit_transform(self.data_encoded[col].astype(str))
                self.encoders[col] = le

        # --- STEP 4: DEFINE BOUNDS ---
        self.X_bounds = []
        for col in self.feature_cols:
            c_min = self.data_encoded[col].min()
            c_max = self.data_encoded[col].max()
            self.X_bounds.append((c_min, c_max))

    def train_model(self):
        X = self.data_encoded[self.feature_cols]
        y = self.data_encoded[self.target_cols]

        print(f"\nTraining model on {len(self.feature_cols)} features...")
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.model = MultiOutputRegressor(rf)
        self.model.fit(X, y)
        print(f"Model Accuracy (R^2): {self.model.score(X, y):.4f}")

    def find_inputs_for_target(self):
        """
        The Reverse Operation: 
        1. Translates user text/numbers into model-ready values.
        2. Runs a mathematical search to find the best inputs.
        3. Translates the results back into readable text/units.
        """
        print("\n--- INVERSE PREDICTION MODE ---")
        
        # --- 1. COLLECT AND TRANSLATE TARGET VALUES ---
        desired_y = []
        for col in self.target_cols:
            while True:
                user_val = input(f"Enter desired value for target '{col}': ")
                
                # Case A: The target is a text category (like 'Style')
                if col in self.encoders:
                    try:
                        le = self.encoders[col]
                        # Convert text to the number the model was trained on
                        val = float(le.transform([user_val.strip()])[0])
                        desired_y.append(val)
                        break
                    except ValueError:
                        print(f"Error: '{user_val}' not found in data.")
                        print(f"Try one of these: {list(le.classes_[:10])}...")
                
                # Case B: The target is a normal number (like 'Color' or 'ABV')
                else:
                    try:
                        val = float(user_val)
                        desired_y.append(val)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        
        desired_y = np.array(desired_y)

        # --- 2. DEFINE THE SEARCH MATH (OBJECTIVE) ---
        def objective_function_rename_dev_1(x_input):
            # We want the difference between (Prediction) and (Target) to be zero
            prediction = self.model.predict([x_input])[0]
            # Mean Squared Error: ensures all targets are weighted
            return np.mean((prediction - desired_y)**2)

        # Start search from the average of all features (numeric_only for safety)
        x0 = self.data_encoded[self.feature_cols].mean(numeric_only=True).values

        # --- 3. RUN THE OPTIMIZER ---
        print("\nSearching for the perfect recipe variables...")
        result = minimize(
            objective_function, 
            x0, 
            method='SLSQP', 
            bounds=self.X_bounds,
            tol=1e-3
        )

        # --- 4. DECODE AND DISPLAY RESULTS ---
        print("\n" + "="*40)
        print("           OPTIMAL INPUTS FOUND         ")
        print("="*40)
        
        optimized_features = result.x
        final_recommendation = {}
        
        for i, col in enumerate(self.feature_cols):
            val = optimized_features[i]
            
            if col in self.encoders:
                # Turn the optimized number back into text (e.g., 4.0 -> 'IPA')
                val_int = int(round(val))
                val_int = max(0, min(val_int, len(self.encoders[col].classes_) - 1))
                decoded_val = self.encoders[col].inverse_transform([val_int])[0]
                final_recommendation[col] = decoded_val
            else:
                # Keep as number, round for readability
                final_recommendation[col] = round(float(val), 2)
        
        # Print input features line by line
        for k, v in final_recommendation.items():
            print(f"{k:20}: {v}")
            
        # --- 5. FINAL VERIFICATION ---
        # Get the final prediction from the model using the found inputs
        raw_pred = self.model.predict([optimized_features])[0]
        # Fix for "only 0-dimensional arrays" error: ensure it is always an indexable array
        pred_array = np.atleast_1d(raw_pred)
        
        print("-" * 40)
        
        display_results = {}
        for i, col in enumerate(self.target_cols):
            current_p_val = pred_array[i]
            
            if col in self.encoders:
                idx = int(round(current_p_val))
                idx = max(0, min(idx, len(self.encoders[col].classes_) - 1))
                display_results[col] = self.encoders[col].inverse_transform([idx])[0]
            else:
                display_results[col] = round(float(current_p_val), 2)
                
        print(f"Expected Outcome: {display_results}")
        print("="*40)
        
if __name__ == "__main__":
    # Use your path
    file_path = "./data/recipeData.csv" 
    
    app = InverseModel(file_path)
    app.preprocess_data()
    app.train_model()
    
    while True:
        app.find_inputs_for_target()
        if input("\nTry another target? (y/n): ").lower() != 'y':
            break