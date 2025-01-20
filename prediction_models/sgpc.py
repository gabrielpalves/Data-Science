import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from sklearn.metrics import accuracy_score


def sgpc_model(X_train, y_train, X_val, y_val, X_test):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Define the Sparse GP Model
    class SparseGPClassificationModel(ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )  # RBF kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Select inducing points
    num_inducing_points = 600
    inducing_points = X_train_tensor[:num_inducing_points]

    # Create model and likelihood
    model = SparseGPClassificationModel(inducing_points)
    likelihood = BernoulliLikelihood()

    # Define the ELBO loss
    mll = VariationalELBO(likelihood, model, num_data=len(X_train_tensor))

    # Training the model
    training_iterations = 600
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = -mll(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    likelihood.eval()

    # Make predictions on the test set
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_val_tensor))
        probabilities = observed_pred.probs
        predictions = probabilities.round()  # Threshold at 0.5

    # Calculate accuracy
    val_accuracy = accuracy_score(y_val_tensor, predictions.numpy())

    # Analyze uncertainty
    with torch.no_grad():
        pred = likelihood(model(X_test_tensor))
        y_test_pred = pred.probs
        predictions_std = pred.variance ** 2
        predictions_upper = y_test_pred + predictions_std*torch.special.ndtri(torch.tensor(1-0.05/2))
        predictions_lower = y_test_pred - predictions_std*torch.special.ndtri(torch.tensor(1-0.05/2))
        
        y_test_pred = y_test_pred.round()
        
        predictions_upper[predictions_upper >= 0.5] = 1
        predictions_upper[predictions_upper < 0.5] = 0
        
        predictions_lower[predictions_lower >= 0.5] = 1
        predictions_lower[predictions_lower < 0.5] = 0
        print(f"Predictive uncertainty:\nMax churn: {predictions_upper.sum()} \
\nPredicted churn: {y_test_pred.sum()}\nMin churn: {predictions_lower.sum()}")
    
    return {
        "validation_accuracy": val_accuracy,
        "test_predictions": y_test_pred,
        "uncertainty": [predictions_lower.sum(), predictions_upper.sum()],
        }
