from sklearn.neural_network import MLPClassifier

def get_simple_model(hidden_layer_size=50, alpha=1e-4, learning_rate_init=1e-3, batch_size='auto', random_state=42):
    """
    Creates and returns a simple neural network model using scikit-learn's MLPClassifier.
    
    Parameters:
        hidden_layer_size (int): Number of neurons in the single hidden layer.
        alpha (float): L2 regularization parameter.
        learning_rate_init (float): Initial learning rate.
        batch_size (int or "auto"): Size of minibatches for stochastic optimizers.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        model (MLPClassifier): Configured neural network model.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='relu',
        solver='adam',
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        batch_size=batch_size,
        max_iter=100,  # Keeping iterations low for faster runs on a laptop.
        random_state=random_state,
        verbose=False
    )
    return model

if __name__ == "__main__":
    # Example usage: create the model and print its configuration.
    model = get_simple_model(batch_size=32)
    print("Simple neural network model created:")
    print(model)
