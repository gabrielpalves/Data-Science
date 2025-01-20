from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def svm_model(X_train, y_train, X_val, y_val, X_test):
    """
    Train Gradient Boosting model and evaluate it.
    
    Args:
        X_train (ndarray): Scaled training features.
        y_train (ndarray): Training labels.
        X_val (ndarray): Scaled validation features.
        y_val (ndarray): Validation labels.
        X_test (ndarray): Scaled test features.
        
    Returns:
        dict: Dictionary with validation accuracy and test predictions.
    """
    # Initialize the SVM model with RBF kernel (default)
    svm_model = SVC(kernel='rbf', probability=True, random_state=28)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred = svm_model.predict(X_val)

    # Validation metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    # Predict on the test set
    y_test_pred = svm_model.predict(X_test)

    return {"validation_accuracy": val_accuracy, "test_predictions": y_test_pred}
