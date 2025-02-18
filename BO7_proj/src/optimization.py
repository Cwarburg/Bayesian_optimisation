import numpy as np
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Import the data loader and the model creation function.
from data import load_data
from model import get_simple_model

# Define the search space for Bayesian optimization.
# We optimize learning_rate_init and alpha on a logarithmic scale,
# and batch_size is an integer between 16 and 256.
space = [
    Real(1e-4, 1e-1, prior="log-uniform", name="learning_rate_init"),
    Real(1e-5, 1e-1, prior="log-uniform", name="alpha"),
    Integer(16, 256, name="batch_size")
]

@use_named_args(space)
def objective(**params):
    """
    Objective function that creates a model with the given hyperparameters,
    performs 3-fold cross-validation on the MNIST data, and returns the negative
    accuracy (since gp_minimize minimizes the objective).
    """
    # Print the parameters being evaluated.
    print(f"Evaluating with parameters: {params}")

    # Load the MNIST data.
    X, y = load_data()
    
    # (Optional) For faster evaluation on a laptop, you can use a subset:
    # X, y = X[:5000], y[:5000]
    
    # Create the model using the current set of hyperparameters.
    model = get_simple_model(
        # Using a fixed hidden layer size for simplicity.
        hidden_layer_size=50,
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        batch_size=params["batch_size"],
        random_state=42
    )
    
    # Evaluate the model using 3-fold cross-validation.
    # We use negative accuracy because gp_minimize minimizes the objective.
    cv_accuracy = cross_val_score(model, X, y, cv=3, n_jobs=-1, scoring="accuracy").mean()
    
    print(f"Accuracy: {cv_accuracy:.4f}\n")
    return -cv_accuracy

def run_optimization(n_calls=10):
    """
    Runs Bayesian optimization to tune the model's hyperparameters.
    
    Parameters:
        n_calls (int): The number of hyperparameter sets to evaluate.
    """
    print("Starting Bayesian Optimization...")
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    
    best_params = {
        "learning_rate_init": result.x[0],
        "alpha": result.x[1],
        "batch_size": result.x[2]
    }
    best_accuracy = -result.fun  # Negate because we minimized the negative accuracy
    
    print("Optimization complete!")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    run_optimization()
