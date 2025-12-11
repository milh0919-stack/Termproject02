# Self-Aware Appliances: AI-Driven Energy Diagnosis System

> **Term Project 02 - Develop your AI Model**
> * **Author:** Hyunchan Lee (Student ID: 20220033)
> * **Affiliation:** Dept. of Energy Engineering, KENTECH
> * **Date:** December 11, 2025

## 1. Project Overview
Traditional energy efficiency ratings (e.g., Grade 1-5) are static and fail to account for dynamic environmental factors. A device rated as "efficient" can still lead to energy waste if used improperly under specific conditions.

This project introduces a **"Self-Aware Appliance"** system. By embedding a lightweight **Edge AI model**, home appliances can autonomously evaluate whether their current energy consumption is appropriate based on real-time contexts such as outdoor temperature, household size, and season. This approach enables **dynamic anomaly detection** without relying on central servers.

### Key Features
* **Context-Aware Diagnosis:** Calculates the standard energy consumption customized for specific environmental conditions (e.g., Winter, -5°C, 4-person household).
* **On-Device AI Architecture:** Implements a lightweight Multi-Layer Perceptron (MLP) model optimized for embedded systems.
* **Interactive Diagnosis Interface:** Provides a Command Line Interface (CLI) where users can input sensor data and receive immediate feedback on energy efficiency.

---

## 2. System Architecture

### Data Pipeline
We utilized the **Smart Home Energy Consumption Dataset** (100,000 samples) to train the model.
* **Feature Selection:** Temporal features (Time, Date) were removed to focus on situational patterns.
* **Preprocessing:**
    * **Numerical Features:** Standard Scaling applied to Outdoor *Temperature* and *Household Size*.
    * **Categorical Features:** One-Hot Encoding applied to *Appliance Type* and *Season*.

### Model Structure (MLP)
The model is designed using **PyTorch** to perform regression analysis.
* **Input Layer (16 Nodes):** Receives the encoded context vector.
* **Hidden Layers:**
    * **Layer 1:** 64 Nodes with **ReLU** activation (Feature Expansion).
    * **Layer 2:** 32 Nodes with **ReLU** activation (Feature Compression).
* **Output Layer (1 Node):** Predicts the expected energy consumption (kWh).
* **Optimization:**
    * **Loss Function:** Mean Squared Error (MSE).
    * **Optimizer:** Adam (Learning Rate: 0.001).

---

## 3. Environment & Dependencies

This project was developed and tested in the following environment.

* **OS:** Windows 11
* **Python Version:** Python 3.9.25
* **Key Libraries:**
    * `torch`: For building and training the MLP model.
    * `pandas`: For data manipulation and CSV loading.
    * `numpy`: For numerical operations and tensor conversion.
    * `scikit-learn`: For data preprocessing.

## 4. Installation and Usage

### Installation
Clone this repository and install the dependencies.

```bash
git clone https://github.com/milh0919-stack/Termproject02.git
pip install -r requirements.txt
```
### Usage
**Execute the `main.py` script.**
```python main.py```
**Enter and select conditions**

## 5.appendix.
* `plot.py` and `training_loss_curve.png` are the code and figure to draw the loss plot to find the appropriate epoch.