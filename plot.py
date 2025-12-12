import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Added libraries for evaluation and plotting
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# =============================================================================
# 1. Data Preprocessing Module
# Description: Handles data loading, feature selection, and scaling/encoding.
# =============================================================================
def load_and_preprocess_data(filepath):
    """
    Loads the dataset and transforms features into a machine-readable format
    suitable for training the neural network.
    """
    # [Step 1] Load Dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The dataset file '{filepath}' was not found.")
    
    # [Step 2] Feature Selection
    # We select situational context features and exclude temporal data (Time, Date).
    feature_cols = ['Outdoor Temperature (°C)', 'Household Size', 'Appliance Type', 'Season']
    X = df[feature_cols]
    
    # Target Variable: Energy Consumption (Continuous value for regression task)
    y = df['Energy Consumption (kWh)'].values.astype(np.float32)
    
    # [Step 3] Data Transformation Pipeline
    # - StandardScaler for numerical features to normalize scales.
    # - OneHotEncoder for categorical features to convert them into binary vectors.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Outdoor Temperature (°C)', 'Household Size']),
            ('cat', OneHotEncoder(sparse_output=False), ['Appliance Type', 'Season'])
        ])
    
    # Fit and transform the data, then convert to float32 for PyTorch compatibility.
    X_processed = preprocessor.fit_transform(X).astype(np.float32)
    
    print(f"[System] Data loaded and preprocessed successfully. Input features shape: {X_processed.shape}")
    return X_processed, y, preprocessor

# =============================================================================
# 2. Deep Learning Model Architecture
# Description: Multi-Layer Perceptron (MLP) designed for regression tasks.
# Structure: Input(16 nodes) -> Hidden(64, ReLU) -> Hidden(32, ReLU) -> Output(1 node)
# =============================================================================
class EnergyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(EnergyPredictor, self).__init__()
        
        # First hidden layer: Expands feature space to capture complex patterns
        self.layer1 = nn.Linear(input_dim, 64) 
        # Activation function: Introduces non-linearity
        self.relu = nn.ReLU() 
        # Second hidden layer: Compresses features to extract essential information
        self.layer2 = nn.Linear(64, 32)
        # Output layer: Predicts a single continuous value (kWh)
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, x):
        """Defines the forward pass of the data through the network."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x) # No activation in final layer for regression
        return x

# =============================================================================
# 3. Helper Functions (Menu Selection & Visualization)
# =============================================================================
def get_user_choice(options, prompt_name):
    """
    Displays a numbered list of options and returns the user's valid choice.
    Handles invalid inputs robustly.
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
                print(f"[Error] Invalid selection. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("[Error] Invalid input format. Please enter an integer.")

def plot_actual_vs_predicted(y_true, y_pred, r2_score_val):
    """
    Generates and saves a scatter plot comparing actual vs. predicted values.
    Visualizes the R-squared score.
    """
    plt.figure(figsize=(8, 8))
    # Scatter plot of data points
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue', label='Predictions')

    # Add a diagonal red line representing perfect prediction (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Fit (Identity Line)')

    # Labels and Title including R2 score
    plt.title(f'Model Evaluation: Actual vs. Predicted Energy Consumption\nR-Squared ($R^2$) Score: {r2_score_val:.4f}', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Usage (kWh) [Ground Truth]', fontsize=12)
    plt.ylabel('Predicted Usage (kWh) [AI Model]', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot to a high-resolution image file
    save_path = 'r2_actual_vs_predicted_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">>> [System] R² visualization graph saved as '{save_path}'.")
    plt.close() # Close plot to free memory

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
        print(f"[Critical Error] {e}")
        return

    # Train/Test Split: Reserve 20% of data for unbiased evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert numpy arrays to PyTorch tensors for model training
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    
    input_dim = X_train.shape[1]

    # ---------------------------------------------------------
    # [Phase 2] Model Initialization & Training
    # ---------------------------------------------------------
    model = EnergyPredictor(input_dim)
    # Loss Function: Mean Squared Error for regression
    criterion = nn.MSELoss() 
    # Optimizer: Adam for efficient stochastic gradient descent
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Increased epochs to ensure convergence based on preliminary tests
    epochs = 400 
    print(f"\n>>> [System] Starting model training for {epochs} epochs...")
    
    loss_history = []

    for epoch in range(epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Clear previous gradients
        outputs = model(X_train_tensor) # Forward pass
        loss = criterion(outputs, y_train_tensor) # Calculate loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update weights
        
        loss_history.append(loss.item())

        # Print training progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Training Loss (MSE): {loss.item():.4f}")
            
    print(">>> [System] Training process completed.")

    # ---------------------------------------------------------
    # [Phase 2.5] Performance Evaluation & Visualization (New!)
    # ---------------------------------------------------------
    print("\n>>> [System] Evaluating model performance on unseen Test Data...")
    
    # Prepare test data tensors
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)

    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for inference
        y_pred_tensor = model(X_test_tensor)
        y_pred_np = y_pred_tensor.numpy() # Convert predictions back to numpy
        y_true_np = y_test_tensor.numpy()

    # Calculate quantitative metrics
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_np, y_pred_np)
    r2 = r2_score(y_true_np, y_pred_np)

    print("="*60)
    print(" >>> [Quantitative Performance Results (Test Set)]")
    print("="*60)
    print(f" • MSE (Mean Squared Error)       : {mse:.4f}")
    print(f" • RMSE (Root Mean Squared Error) : {rmse:.4f} kWh")
    print(f" • MAE (Mean Absolute Error)      : {mae:.4f} kWh")
    print(f" • R2 Score (Coefficient of Det.) : {r2:.4f}")
    print("="*60)

    # Generate and save the R2 visualization plot
    plot_actual_vs_predicted(y_true_np, y_pred_np, r2)


    # ---------------------------------------------------------
    # [Phase 3] Interactive Edge AI Demo (Menu Selection)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Smart Appliance Self-Diagnosis Mode (Edge AI Demo)")
    print("="*60)
    print("Please input the current environmental context to evaluate efficiency.\n")

    try:
        # 1. Numerical Inputs
        temp = float(input("1. Outdoor Temperature (°C) [e.g., -5.0]: "))
        size = int(input("2. Household Size (People) [e.g., 4]: "))
        
        # 2. Categorical Inputs (Menu Selection based on dataset defined options)
        appliance_options = [
            'Fridge', 'Oven', 'Dishwasher', 'Heater', 'Microwave', 
            'Air Conditioning', 'Computer', 'TV', 'Washing Machine', 'Lights'
        ]
        appliance = get_user_choice(appliance_options, "Appliance Type")
        print(f" -> Selected: {appliance}")
        
        season_options = ['Spring', 'Summer', 'Fall', 'Winter']
        season = get_user_choice(season_options, "Season")
        print(f" -> Selected: {season}")
        
        # 3. Actual Usage Input for Comparison
        actual_usage = float(input("\n>>> Enter Actual Energy Usage (kWh) from sensor: "))

        # Construct Input DataFrame for the current situation
        my_situation = pd.DataFrame({
            'Outdoor Temperature (°C)': [temp],
            'Household Size': [size],
            'Appliance Type': [appliance],
            'Season': [season]
        })

        # Preprocess Input & Perform Inference
        # Note: We use 'transform', not 'fit_transform', to use saved scaling parameters.
        my_input_vector = preprocessor.transform(my_situation).astype(np.float32)
        my_input_tensor = torch.tensor(my_input_vector)
        
        model.eval() 
        with torch.no_grad():
            # Get the single prediction value
            ai_prediction = model(my_input_tensor).item()

        # Display Diagnosis Results
        print("\n" + "-"*60)
        print(f" >>> [AI Diagnosis Report]")
        print("-" * 60)
        print(f" - Actual Usage Recorded : {actual_usage:.2f} kWh")
        print(f" - AI Predicted Standard : {ai_prediction:.2f} kWh")
        print("-" * 60)
        
        # Anomaly Detection Logic based on deviation from prediction
        diff = actual_usage - ai_prediction
        # Threshold refined based on test set RMSE (approx. 0.6 kWh)
        threshold = 1.0 
        
        if diff > threshold:
            print(" [RESULT]:WARNING - Significant Over-consumption Detected!")
            print(f" Analysis: Usage is {diff:.2f} kWh HIGHER than the standard estimate.")
        elif diff < -threshold:
            print(" [RESULT]:EXCELLENT - High Energy Efficiency Detected!")
            print(f" Analysis: Usage is {-diff:.2f} kWh LOWER than the standard estimate.")
        else:
            print(" [RESULT]:NORMAL - Usage is within the expected standard range.")
        print("-" * 60)

    except Exception as e:
        print(f"\n[Error] An unexpected error occurred during demo: {e}")

if __name__ == '__main__':
    main()