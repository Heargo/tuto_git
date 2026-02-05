import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from typing import List, Dict, Any  # Ensure Any is imported for Pydantic
import warnings

warnings.filterwarnings('ignore')

# 1. Define and Rebuild the Schema
class RecipeGoal(BaseModel):
    targets: Dict[str, Any]

# This line "finishes" the Pydantic model definition so Any is recognized
RecipeGoal.model_rebuild()

class ProInverseSystem:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding='latin-1')
        self.feature_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.target_encoders = {}
        self.model = None
        self.feature_cols = []
        self.target_cols = []
        # Scientific blacklist to remove non-brewing variables
        self.blacklist = ['BeerID', 'Name', 'URL', 'UserId', 'image', 'StyleID']

    def preprocess(self, target_names: List[str]):
        self.target_cols = target_names
        self.feature_cols = [c for c in self.df.columns if c not in self.target_cols and c not in self.blacklist]
        
        working_df = self.df.copy()

        # 1. CLEANING: Separating types correctly
        # Use pandas-native numeric check to avoid Numpy dtype errors
        num_cols = [c for c in working_df.columns if pd.api.types.is_numeric_dtype(working_df[c])]
        obj_cols = [c for c in working_df.columns if not pd.api.types.is_numeric_dtype(working_df[c])]
        
        # Fill missing values
        working_df[num_cols] = working_df[num_cols].fillna(working_df[num_cols].median())
        for col in obj_cols:
            working_df[col] = working_df[col].fillna(working_df[col].mode()[0] if not working_df[col].mode().empty else "Unknown")

        # 2. FEATURE ENCODING
        self.cat_mask = []
        for col in self.feature_cols:
            is_cat = not pd.api.types.is_numeric_dtype(working_df[col])
            self.cat_mask.append(is_cat)
            
        actual_cat_features = [col for col, is_cat in zip(self.feature_cols, self.cat_mask) if is_cat]
        
        if actual_cat_features:
            # Force to string to prevent encoder crashes with new StringDtype
            working_df[actual_cat_features] = self.feature_encoder.fit_transform(working_df[actual_cat_features].astype(str))

        # 3. TARGET ENCODING: The final fix for 'Style' / 'Cream Ale'
        for t in self.target_cols:
            # If it's not numeric, it MUST be encoded
            if not pd.api.types.is_numeric_dtype(working_df[t]):
                print(f"-> Encoding categorical target: {t}")
                le = LabelEncoder()
                working_df[t] = le.fit_transform(working_df[t].astype(str))
                self.target_encoders[t] = le
        
        # Final safety conversion
        self.data_encoded = working_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        print("Preprocessing Complete. All types aligned.")

    def solve(self, goals: RecipeGoal, trials=100):
        numeric_goals = []
        
        # Standardize keys to match the dataframe exactly
        goal_dict = {k: v for k, v in goals.targets.items()}
        
        for t in self.target_cols:
            raw_val = goal_dict[t]
            
            # 1. Check if the value is a string (like 'Cream Ale')
            if isinstance(raw_val, str):
                if t in self.target_encoders:
                    try:
                        # Translate 'Cream Ale' -> Numeric ID (e.g., 12.0)
                        encoded_val = self.target_encoders[t].transform([raw_val.strip()])[0]
                        numeric_goals.append(float(encoded_val))
                    except ValueError:
                        # Help the user if they misspelled the style
                        valid = list(self.target_encoders[t].classes_[:5])
                        raise ValueError(f"Style '{raw_val}' not found. Did you mean: {valid}?")
                else:
                    # If it's a string but we have no encoder, we can't do math on it
                    raise ValueError(f"Target '{t}' contains text '{raw_val}' but wasn't encoded. Check your CSV types.")
            
            # 2. Otherwise, treat it as a number
            else:
                numeric_goals.append(float(raw_val))
        
        numeric_goals = np.array(numeric_goals)

        # --- OPTIMIZATION LOOP ---
        def objective(trial):
            candidate_x = []
            for i, col in enumerate(self.feature_cols):
                low, high = self.data_encoded[col].min(), self.data_encoded[col].max()
                if self.cat_mask[i]:
                    candidate_x.append(trial.suggest_int(col, int(low), int(high)))
                else:
                    candidate_x.append(trial.suggest_float(col, float(low), float(high)))
            
            # The model predicts numeric IDs for styles
            pred = self.model.predict(np.array([candidate_x]))[0]
            # Calculate the distance between prediction and your numeric goal
            return np.mean((pred - numeric_goals)**2)

        print(f"Goal established: {numeric_goals} (Numeric Translation)")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)
        return self._format_output(study.best_params)


    def train(self):
        X = self.data_encoded[self.feature_cols]
        y = self.data_encoded[self.target_cols]
        
        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08)
        self.model.fit(X, y)
        print("Model Trained.")
        
        # Trigger Feature Importance Plot
        self._plot_importance()

    def _plot_importance(self):
        plt.figure(figsize=(10, 6))
        # Get feature importance from XGBoost
        feat_importances = pd.Series(self.model.feature_importances_, index=self.feature_cols)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Top 10 Factors Driving Recipe Outcomes")
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()
        plt.show()

    def _format_output(self, params):
        res = {}
        for i, col in enumerate(self.feature_cols):
            val = params[col]
            if self.cat_mask[i]:
                cat_idx = sum(self.cat_mask[:i])
                res[col] = self.feature_encoder.categories_[cat_idx][int(round(val))]
            else:
                res[col] = round(val, 3)
        return res

# --- Execution ---
if __name__ == "__main__":
    system = ProInverseSystem("./data/recipeData.csv")
    system.preprocess(["Style", "Color"])
    system.train()
    
    my_goals = RecipeGoal(targets={"Style": "Cream Ale", "Color": 4.5})
    recipe = system.solve(my_goals)
    
    print("\n--- OPTIMIZED BREW SPECIFICATIONS ---")
    for k, v in recipe.items():
        print(f"{k:20}: {v}")