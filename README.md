# Multilayer Perceptron (MLP) for California Housing Price Prediction

**Course**: COMP 2211 Exploring Artificial Intelligence  
**Lab**: 6

## Overview
This project involves building a Multilayer Perceptron (MLP) model to predict house prices in California using the California Housing dataset. The lab focuses on data preprocessing, model construction, training, and evaluation to achieve optimal regression performance.

## Dataset
The California Housing dataset contains 20,640 samples with 8 features:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age
- **AveRooms**: Average rooms per household
- **AveBedrms**: Average bedrooms per household
- **Population**: Block group population
- **AveOccup**: Average household members
- **Latitude**: Block group latitude
- **Longitude**: Block group longitude

**Target**: `MedHouseVal` (Median House Value in hundreds of thousands of dollars).  
[Dataset Details](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html)

## Tasks
1. **Data Preprocessing**  
   - Split data into training (80%) and testing (20%) sets with `random_state=42`.
   - Normalize features and target using Z-score normalization based on training statistics.

2. **Model Building**  
   - Construct an MLP with:
     - Input layer (8 neurons)
     - Hidden layers (64 and 32 neurons with ReLU activation)
     - Output layer (1 neuron for regression)

3. **Model Training**  
   - Compile with Adam optimizer and Mean Squared Error (MSE) loss.
   - Train for 150 epochs with a batch size of 64 and 20% validation split.

4. **Evaluation**  
   - Metrics: MSE, RMSE, R² Score.
   - Visualization: Actual vs. Predicted values plot.

## Installation
```
pip install tensorflow keras scikit-learn pandas matplotlib seaborn
```
## Usage
Run the Script
1. Execute the provided Python script (lab6_tasks.py) to:

-Preprocess data

-Train the MLP model

-Evaluate performance

-Generate plots

2. Model Weights
Trained weights are saved as mlp_model.weights.h5.

##Results
-MSE: 0.203

-RMSE: 0.451

-R² Score: 0.793

#### Actual vs Predicted Plot

![image](https://github.com/user-attachments/assets/c8f299fc-ec3c-466c-af75-657db2dc71c8)

Note: Plot generated during execution shows predictions closely aligned with the actual values.

## Model Architecture

Model: "sequential"
| Layer (type)       | Output Shape      | Param #   |
|--------------------|-------------------|-----------|
| dense (Dense)      | `(None, 64)`      | 576       |
| dense_1 (Dense)    | `(None, 32)`      | 2,080     |
| dense_2 (Dense)    | `(None, 1)`       | 33        |

**Total params**: 2,689  
**Trainable params**: 2,689  
**Non-trainable params**: 0  

## Grading Scheme
- Data Preprocessing: 3 points

- Model Implementation: 2 points

- Model Compilation: 1 point

- R² Score ≥ 0.70: 1 point

- R² Score ≥ 0.75: 2 points

- R² Score ≥ 0.78: 3 points (Achieved: 0.793)
