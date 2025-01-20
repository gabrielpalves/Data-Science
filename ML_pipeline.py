import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import itertools
import random


# Define the Neural Network architecture
class ChurnPredictionNN(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3, act1, act2, act3):
        super(ChurnPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, 2)  # Binary classification: 2 output classes
        self.act1 = act1
        self.act2 = act2
        self.act3 = act3
        self.initialize_weights()  # Initialize weights

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)  # No activation here; handled by CrossEntropyLoss
        return x

    def initialize_weights(self):
        # He initialization for layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Functions for the features
def drop_irrelevant(df):
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df = df.dropna()  # drop missing values
    return df


def encode_categorical(df):
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    df = pd.get_dummies(df, columns=['Geography'], drop_first=False)
    return df


# Read .csv files
df_original = pd.read_csv('Abandono_clientes.csv')
df_test = pd.read_csv('Abandono_teste.csv', delimiter=';')

# Preprocess data
df = drop_irrelevant(df_original)
df = encode_categorical(df)

df_test = drop_irrelevant(df_test)
df_test = encode_categorical(df_test)

X_test = df_test.values

# Split dataset
X = df.drop('Exited', axis=1).values  # Features
y = df['Exited'].values               # Target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


## Neural Network model
# To torch tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Define hyperparameter search space
param_space = {
    'batch_size': [64, 128],
    'hidden_size1': [32, 64, 128],
    'hidden_size2': [16, 32, 64],
    'hidden_size3': [8, 16, 32],
    'act1': [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU()],
    'act2': [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU()],
    'act3': [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU()],
}

# Generate random combinations of hyperparameters
n_searches = 20  # Number of random searches
param_combinations = [
    {k: random.choice(v) for k, v in param_space.items()}
    for _ in range(n_searches)
]

learning_rate = 0.001
epochs = 100

# Initialize the model
input_size = X_train.shape[1]  # Number of features

# Training loop
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

best_model_all = None
best_val_acc_all = 0
best_train_loss_all = float('inf')
# Random search loop
for i, params in enumerate(param_combinations):
    print(f"Search {i+1}/{n_searches}, Params: {params}")

    # Initialize the model with the current parameters
    model = ChurnPredictionNN(input_size,
                              params['hidden_size1'],
                              params['hidden_size2'],
                              params['hidden_size3'],
                              params['act1'],
                              params['act2'],
                              params['act3'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare data loader with the current batch size
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True
    )
    
    best_model = None
    best_val_acc = 0
    best_train_loss = float('inf')

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_val_tensor, val_predictions)
        
        # Save the best model based on validation accuracy and training loss
        if val_accuracy > best_val_acc or (val_accuracy == best_val_acc and avg_train_loss < best_train_loss):
            best_val_acc = val_accuracy
            best_train_loss = avg_train_loss
            best_model = copy.deepcopy(model)

    # Validation
    best_model.eval()
    with torch.no_grad():
        val_outputs = best_model(X_val_tensor)
        val_predictions = torch.argmax(val_outputs, dim=1)
        val_accuracy = accuracy_score(y_val_tensor, val_predictions)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Track the best model
    if val_accuracy > best_val_acc_all or (val_accuracy == best_val_acc_all and best_train_loss < best_train_loss_all):
        best_val_acc_all = val_accuracy
        best_model_all = copy.deepcopy(best_model)
        best_params = params

print(f"Best Validation Accuracy: {best_val_acc_all:.4f}")
print(f"Best Parameters: {best_params}")

# Use the best model to predict on the test set
best_model_all.eval()
with torch.no_grad():
    test_outputs = best_model_all(X_test_tensor)
    test_predictions = torch.argmax(test_outputs, dim=1)

# Save test predictions
row_numbers = np.arange(1, len(test_predictions) + 1)
result_df = pd.DataFrame({
    'rowNumber': row_numbers,
    'predictedValues': test_predictions.numpy()
})
result_df.to_csv('optimized_predictions.csv', index=False)
print("Predictions saved to 'optimized_predictions.csv'.")
