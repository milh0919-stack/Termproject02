import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Data Preprocessing Module
# Description: Handles data loading, feature selection, and transformation.
def load_and_preprocess_data(filepath):
    """
    Loads dataset and transforms features into machine-readable format.
    """
    # Step 1. Load Dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Step 2. Feature Selection
    # Selects only relevant features for context-aware analysis.
    # Excludes 'Time' and 'Date' to focus on situational patterns rather than time-series.
    feature_cols = ['Outdoor Temperature (°C)', 'Household Size', 'Appliance Type', 'Season']
    X = df[feature_cols]
    
    # Target Variable: Energy Consumption (Regression Task)
    y = df['Energy Consumption (kWh)'].values.astype(np.float32)
    
    # Step 3. Data Transformation Pipeline
    # - Numerical Features (Temp, Size): Standardized (Mean=0, Std=1) for training stability.
    # - Categorical Features (Appliance, Season): One-Hot Encoded to vector format.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Outdoor Temperature (°C)', 'Household Size']),
            ('cat', OneHotEncoder(sparse_output=False), ['Appliance Type', 'Season'])
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X).astype(np.float32)
    
    print(f"[System] Data Loaded Successfully. Input Dimension: {X_processed.shape}")
    return X_processed, y, preprocessor

# 2. Deep Learning Model Architecture
# Description: Multi-Layer Perceptron (MLP) for regression tasks.
# Structure: Input(16) -> Hidden(64) -> ReLU -> Hidden(32) -> ReLU -> Output(1)
class EnergyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnergyPredictor, self).__init__()
        
        # [Layer 1] Expansion Layer
        # Projects input features to a higher-dimensional space (16 -> 64) 
        # to capture complex feature interactions.
        self.layer1 = nn.Linear(input_dim, 64) 
        
        # [Activation Function] ReLU (Rectified Linear Unit)
        # Introduces non-linearity to learn complex patterns (e.g., sudden spikes).
        self.relu = nn.ReLU() 
        
        # [Layer 2] Compression Layer
        # Compresses learned features (64 -> 32) to extract core information.
        self.layer2 = nn.Linear(64, 32)
        
        # [Output Layer] Linear Regression
        # Predicts a single continuous value (kWh). No activation is used here.
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        """
        Forward Propagation: Passes input data through the network layers.
        """
        x = self.relu(self.layer1(x))   # Input -> Hidden 1 -> Activation
        x = self.relu(self.layer2(x))   # Hidden 1 -> Hidden 2 -> Activation
        x = self.output_layer(x)        # Hidden 2 -> Output
        return x

# 3. Main Execution Block
# Description: Orchestrates the training process and runs the interactive demo.
def main():
    # ---------------------------------------------------------
    # [Phase 1] Data Loading & Preparation
    # ---------------------------------------------------------
    print(">>> [System] Initializing AI Energy Auditor...")
    
    # Define file path (Relative path for portability)
    file_path = 'smart_home_energy_consumption_large.csv'
    
    try:
        X, y, preprocessor = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"[Error] {e}")
        return

    # Split dataset into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert Numpy arrays to PyTorch Tensors for model input
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1) # Reshape to (N, 1)
    
    # Determine input dimension dynamically (Expected: 16)
    input_dim = X_train.shape[1]

    # ---------------------------------------------------------
    # [Phase 2] Model Initialization
    # ---------------------------------------------------------
    model = EnergyPredictor(input_dim)
    
    # Loss Function: Mean Squared Error (MSE) for regression accuracy
    criterion = nn.MSELoss() 
    
    # Optimizer: Adam (Adaptive Moment Estimation) for efficient convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ---------------------------------------------------------
    # [Phase 3] Training Loop (Learning Process)
    # ---------------------------------------------------------
    print("\n>>> [System] Training Model (Epochs: 50)...")
    epochs = 50 
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        
        # 1. Zero Gradients: Clear previous gradients
        optimizer.zero_grad()
        
        # 2. Forward Pass: Compute predictions
        outputs = model(X_train_tensor)
        
        # 3. Compute Loss: Calculate error between prediction and ground truth
        loss = criterion(outputs, y_train_tensor)
        
        # 4. Backward Pass: Backpropagation of errors
        loss.backward()
        
        # 5. Update Weights: Optimize model parameters
        optimizer.step()
        
    print(">>> [System] Training Complete.")

    # ---------------------------------------------------------
    # [Phase 4] Interactive Edge AI Demo
    # Scenario: Appliance self-diagnoses its energy efficiency.
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("   🔌  Smart Appliance Self-Diagnosis Mode (Edge AI)  🔌")
    print("="*60)
    print("Please input the current context to evaluate energy efficiency.\n")

    try:
        # Simulate Sensor Inputs from User
        temp = float(input("1. Outdoor Temperature (C) [e.g., -5.0]: "))
        size = int(input("2. Household Size (People) [e.g., 4]: "))
        
        print("\n[Supported Appliances]: Fridge, Oven, Dishwasher, Heater, Microwave, Air Conditioning, TV")
        appliance = input("3. Appliance Type [Case Sensitive!]: ")
        
        print("\n[Seasons]: Spring, Summer, Fall, Winter")
        season = input("4. Current Season: ")
        
        # Actual usage measured by smart plug
        actual_usage = float(input("\n>>> Actual Energy Usage (kWh): "))

        # Construct Input DataFrame
        my_situation = pd.DataFrame({
            'Outdoor Temperature (°C)': [temp],
            'Household Size': [size],
            'Appliance Type': [appliance],
            'Season': [season]
        })

        # Preprocess Input (Transform to Tensor)
        my_input_vector = preprocessor.transform(my_situation).astype(np.float32)
        my_input_tensor = torch.tensor(my_input_vector)
        
        # Perform Inference (Prediction)
        model.eval() # Set to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            ai_prediction = model(my_input_tensor).item()

        # Display Diagnosis Results
        print("\n" + "-"*50)
        print(f" >>> [AI Diagnosis Report]")
        print("-"*50)
        print(f" - Actual Usage      : {actual_usage:.2f} kWh")
        print(f" - Standard Usage    : {ai_prediction:.2f} kWh (AI Predicted)")
        
        # Anomaly Detection Logic
        diff = actual_usage - ai_prediction
        threshold = 0.5 # Anomaly threshold (kWh)
        
        print("-" * 50)
        if diff > threshold:
            print(" Result: [WARNING] Over-consumption Detected!")
            print(f" You are using {diff:.2f} kWh MORE than the standard context.")
        elif diff < -threshold:
            print(" Result: [EXCELLENT] High Energy Efficiency!")
            print(f" You are saving {-diff:.2f} kWh compared to the standard.")
        else:
            print(" Result: [NORMAL] Usage is within the expected range.")
        print("-" * 50)

    except Exception as e:
        print(f"\n[Error] Invalid Input: {e}")
        print("Please check your inputs (e.g., spelling of Appliance Type) and try again.")

if __name__ == '__main__':
    main()