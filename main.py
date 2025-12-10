import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# ==========================================
# 3. Main Execution Flow
# ==========================================
def main():
    # ---------------------------------------------------------
    # [Step 3-1] Load Data (데이터 불러오기)
    # ---------------------------------------------------------
    print(">>> [Stage 1] Loading Data...")
    
    # file_path = r'C:\Termproject02\smart_home_energy_consumption_large.csv' 
    file_path = 'smart_home_energy_consumption_large.csv' 
    
    try:
        X, y, preprocessor = load_and_preprocess_data(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return

    # ---------------------------------------------------------
    # [Step 3-2] Train / Test Split (학습/평가 데이터 분리)
    # ---------------------------------------------------------
    # 80% for Training, 20% for Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)
    
    input_dim = X_train.shape[1] # 16 features
    print(f"Data Loaded Successfully. Input Dimension: {input_dim}") 

    # ---------------------------------------------------------
    # [Step 3-3] Initialize Model, Loss, Optimizer (모델 초기화)
    # ---------------------------------------------------------
    model = EnergyPredictor(input_dim)
    
    # Loss Function: Mean Squared Error (MSE) - 예측값과 실제값 차이의 제곱
    criterion = nn.MSELoss() 
    
    # Optimizer: Adam (Adaptive Moment Estimation) - 가장 대중적인 최적화 도구
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ---------------------------------------------------------
    # [Step 3-4] Training Loop (학습 진행)
    # ---------------------------------------------------------
    print("\n>>> [Stage 2] Training Model...")
    epochs = 50 # 반복 횟수 
    
    for epoch in range(epochs):
        model.train() # 학습 모드 전환
        
        # 1. Forward pass 
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # 2. Backward pass and optimization 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 10번마다 로그 출력
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # [Step 3-5] Evaluation (성능 평가)
    # ---------------------------------------------------------
    model.eval() # 평가 모드 전환 (학습 중단)
    with torch.no_grad():
        test_loss = criterion(model(X_test_tensor), y_test_tensor)
    print(f"\nFinal Test Loss (MSE): {test_loss.item():.4f}")

    # =========================================================
    # 4. Demo Application: "My Energy Auditor"
    # =========================================================
    print("\n" + "="*50)
    print(">>> [Stage 3] Demo: AI Energy Auditor")
    print("="*50)
    
    # 1. Define My Situation 
    my_situation = pd.DataFrame({
        'Outdoor Temperature (°C)': [-5.0],  # 아주 추운 날
        'Household Size': [4],               # 4인 가족
        'Appliance Type': ['Heater'],        # 히터 사용
        'Season': ['Winter']                 # 겨울
    })
    
    # 2. Transform Input (전처리기를 이용해 숫자로 변환)
    my_input_vector = preprocessor.transform(my_situation).astype(np.float32)
    my_input_tensor = torch.tensor(my_input_vector)
    
    # 3. AI Prediction 
    model.eval()
    with torch.no_grad():
        ai_prediction = model(my_input_tensor).item()
    
    # 4. Compare with Actual Usage (내 실제 사용량 비교)
    my_actual_usage = 4.5  # 가정: 내가 실제로 쓴 전력량 (스마트 플러그 측정값)
    
    print(f"[Situation]: Winter(-5°C), 4 People, Using 'Heater'")
    print(f"--------------------------------------------------")
    print(f"My Actual Usage        : {my_actual_usage:.2f} kWh")
    print(f"AI Standard Prediction : {ai_prediction:.2f} kWh")
    print(f"--------------------------------------------------")
    
    diff = my_actual_usage - ai_prediction
    if diff > 0:
        print(f"Result: WARNING! You are using {diff:.2f} kWh MORE than average.")
        print(f"Efficiency Grade: C (Poor)")
    else:
        print(f"Result: GREAT! You are saving {-diff:.2f} kWh compared to average.")
        print(f"Efficiency Grade: A (Excellent)")

if __name__ == '__main__':
    main()