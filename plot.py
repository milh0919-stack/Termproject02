import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# =============================================================================
# 1. Data Preprocessing Module
# Description: Handles data loading, feature selection, and transformation.
# =============================================================================
def load_and_preprocess_data(filepath):
    """
    Loads dataset and transforms features into machine-readable format.
    """
    # [Step 1] Load Dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # [Step 2] Feature Selection
    feature_cols = ['Outdoor Temperature (°C)', 'Household Size', 'Appliance Type', 'Season']
    X = df[feature_cols]
    
    # Target Variable: Energy Consumption (Regression Task)
    y = df['Energy Consumption (kWh)'].values.astype(np.float32)
    
    # [Step 3] Data Transformation Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Outdoor Temperature (°C)', 'Household Size']),
            ('cat', OneHotEncoder(sparse_output=False), ['Appliance Type', 'Season'])
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X).astype(np.float32)
    
    print(f"[System] Data Loaded Successfully. Input Dimension: {X_processed.shape}")
    return X_processed, y, preprocessor

# =============================================================================
# 2. Deep Learning Model Architecture
# Description: Multi-Layer Perceptron (MLP) for regression tasks.
# Structure: Input(16) -> Hidden(64) -> ReLU -> Hidden(32) -> ReLU -> Output(1)
# =============================================================================
class EnergyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnergyPredictor, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 64) 
        self.relu = nn.ReLU() 
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# =============================================================================
# 3. Helper Function for Menu Selection
# =============================================================================
def get_user_choice(options, prompt_name):
    """
    Displays a numbered list of options and returns the user's choice.
    """
    print(f"\n[Select {prompt_name}]:")
    for idx, option in enumerate(options, 1):
        print(f" {idx}. {option}")
        
    while True:
        try:
            choice = int(input(f">>> Enter number (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"[Error] Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("[Error] Invalid input. Please enter a number.")
# loss plot generator
def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title('Model Training Loss Curve (400 Epochs)', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # show&save the plot
    plt.savefig('training_loss_curve.png', dpi=300)
    print(">>> [System] Loss graph saved as 'training_loss_curve.png'")
    plt.show() 

# =============================================================================
# 4. Main Execution Block
# =============================================================================
def main():
    # ---------------------------------------------------------
    # [Phase 1] Data Loading & Preparation
    # ---------------------------------------------------------
    print(">>> [System] Initializing AI Energy Auditor...")
    
    file_path = 'smart_home_energy_consumption_large.csv'
    
    try:
        X, y, preprocessor = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"[Error] {e}")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    
    input_dim = X_train.shape[1]

    # ---------------------------------------------------------
    # [Phase 2] Model Initialization & Training
    # ---------------------------------------------------------
    model = EnergyPredictor(input_dim)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    print("\n>>> [System] Training Model (Epochs: 50)...")
    epochs = 400 
    loss_history = [] 
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Loss value
        loss_history.append(loss.item())

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    print(">>> [System] Training Complete.")
    
    # plot generator
    plot_loss_curve(loss_history)

    # ---------------------------------------------------------
    # [Phase 3] Interactive Edge AI Demo (Menu Selection)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Smart Appliance Self-Diagnosis Mode (Edge AI)")
    print("="*60)
    print("Please input the current context to evaluate energy efficiency.\n")

    try:
        # 1. Numerical Inputs
        temp = float(input("1. Outdoor Temperature (C) [e.g., -5.0]: "))
        size = int(input("2. Household Size (People) [e.g., 4]: "))
        
        # 2. Categorical Inputs (Menu Selection)
        # Define available options strictly based on dataset
        appliance_options = [
            'Fridge', 'Oven', 'Dishwasher', 'Heater', 'Microwave', 
            'Air Conditioning', 'Computer', 'TV', 'Washing Machine', 'Lights'
        ]
        appliance = get_user_choice(appliance_options, "Appliance Type")
        print(f" -> Selected: {appliance}")
        
        season_options = ['Spring', 'Summer', 'Fall', 'Winter']
        season = get_user_choice(season_options, "Season")
        print(f" -> Selected: {season}")
        
        # 3. Actual Usage Input
        actual_usage = float(input("\n>>> Actual Energy Usage (kWh): "))

        # Construct Input DataFrame
        my_situation = pd.DataFrame({
            'Outdoor Temperature (°C)': [temp],
            'Household Size': [size],
            'Appliance Type': [appliance],
            'Season': [season]
        })

        # Preprocess Input & Inference
        my_input_vector = preprocessor.transform(my_situation).astype(np.float32)
        my_input_tensor = torch.tensor(my_input_vector)
        
        model.eval() 
        with torch.no_grad():
            ai_prediction = model(my_input_tensor).item()

        # Display Diagnosis Results
        print("\n" + "-"*50)
        print(f" >>> [AI Diagnosis Report]")
        print("-"*50)
        print(f" - Actual Usage      : {actual_usage:.2f} kWh")
        print(f" - Standard Usage    : {ai_prediction:.2f} kWh (AI Predicted)")
        
        # Anomaly Detection Logic
        diff = actual_usage - ai_prediction
        threshold = 0.5 
        
        print("-" * 50)
        if diff > threshold:
            print(" Result: [WARNING] Over-consumption Detected!")
            print(f" You are using {diff:.2f} kWh MORE than the standard.")
        elif diff < -threshold:
            print(" Result: [EXCELLENT] High Energy Efficiency!")
            print(f" You are saving {-diff:.2f} kWh compared to the standard.")
        else:
            print(" Result: [NORMAL] Usage is within the expected range.")
        print("-" * 50)

    except Exception as e:
        print(f"\n[Error] Invalid Input: {e}")

if __name__ == '__main__':
    main()