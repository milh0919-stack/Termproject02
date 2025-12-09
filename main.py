import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch.nn as nn

def load_and_preprocess_data(filepath):
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    # 2. Feature Selection 
    # We use 'Outdoor Temperature', 'Household Size' (Numerical)
    # and 'Appliance Type', 'Season' (Categorical)
    feature_cols = ['Outdoor Temperature (°C)', 'Household Size', 'Appliance Type', 'Season']
    X = df[feature_cols]
    y = df['Energy Consumption (kWh)'].values.astype(np.float32)
    
    # 3. Define Preprocessor
    # Numerical features -> Standardization (StandardScaler)
    # Categorical features -> One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Outdoor Temperature (°C)', 'Household Size']),
            ('cat', OneHotEncoder(sparse_output=False), ['Appliance Type', 'Season'])
        ])
    
    # 4. Transform Data
    X_processed = preprocessor.fit_transform(X).astype(np.float32)
    
    print(f"Data Loaded. Input Shape: {X_processed.shape}")
    return X_processed, y, preprocessor

class EnergyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnergyPredictor, self).__init__()
        
        # Layer 1: Expansion (Input -> 64 nodes)
        # Captures complex features from input data
        self.layer1 = nn.Linear(input_dim, 64) 
        self.relu = nn.ReLU() # Activation function for non-linearity
        
        # Layer 2: Compression (64 -> 32 nodes)
        # Summarizes the learned features
        self.layer2 = nn.Linear(64, 32)
        
        # Output Layer: Regression (32 -> 1 node)
        # Predicts the final energy consumption value (kWh)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        # Forward pass: Input -> Hidden 1 -> Hidden 2 -> Output
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x) # No activation at output (Linear regression)
        return x
