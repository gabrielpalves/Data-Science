import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import accuracy_score


def nn_model(X_train, y_train, X_val, y_val, X_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Define the Neural Network architecture
    class ChurnPredictionNN(nn.Module):
        def __init__(self, input_size):
            super(ChurnPredictionNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)  # Binary classification: 2 output classes

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)  # No activation here; handled by CrossEntropyLoss
            return x

    # Initialize the model
    input_size = X_train.shape[1]  # Number of features
    model = ChurnPredictionNN(input_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    batch_size = 64
    train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_model = None
    best_val_acc = 0
    best_train_loss = float('inf')

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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Restore the best model
    model = best_model
    print(f"Best Validation Accuracy: {best_val_acc:.4f}, Best Training Loss: {best_train_loss:.4f}")

    # Predict on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1)

    # Convert predictions to a NumPy array
    y_test_pred = test_predictions.numpy()

    churn = test_predictions.sum()
    total = X_test_tensor.shape[0]
    print(f'Total predicted churn: {churn} out of {total} ({churn/total*100:.1f}%)')

    return {"validation_accuracy": val_accuracy, "test_predictions": y_test_pred}
