import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from optimization import run_optimization

def main():
    # Run the optimization (n_calls controls how many evaluations are done)
    result = run_optimization(n_calls=10)
    
    # Visualization 1: Convergence Plot
    # This plot shows the best found objective value as the optimization progresses.
    plt.figure(figsize=(8, 6))
    plot_convergence(result)
    plt.title("Convergence Plot")
    plt.xlabel("Number of Calls")
    plt.ylabel("Negative Accuracy")
    plt.show()

    # Visualization 2: Evaluations Plot
    # This plot shows the objective values obtained at each hyperparameter setting.
    plt.figure(figsize=(8, 6))
    plot_evaluations(result)
    plt.title("Evaluation Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Accuracy")
    plt.show()

    # Visualization 3: Objective Plot (Optional)
    # This plot gives a sense of the objective landscape with respect to pairs of hyperparameters.
    # Note: With three hyperparameters, this creates multiple subplots.
    plt.figure(figsize=(10, 8))
    plot_objective(result)
    plt.suptitle("Objective Function Landscape")
    plt.show()

if __name__ == "__main__":
    main()
