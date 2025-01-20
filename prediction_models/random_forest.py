from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Random Forest Model
def rf_model(X_train, y_train, X_val, y_val, X_test):
    """
    Train Random Forest model and evaluate it.
    
    Args:
        X_train (ndarray): Scaled training features.
        y_train (ndarray): Training labels.
        X_val (ndarray): Scaled validation features.
        y_val (ndarray): Validation labels.
        X_test (ndarray): Scaled test features.
        
    Returns:
        dict: Dictionary with validation accuracy and test predictions.
    """
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(n_estimators=300, random_state=28)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Predict on test set
    y_test_pred = model.predict(X_test)
    
    return {"validation_accuracy": val_accuracy, "test_predictions": y_test_pred}, model
