import numpy as np
import matplotlib.pyplot as plt
import SCRIPT_CONFITG
import os.path as osp
from typing import Callable, List
from lib.Chebytool import ChebyshevApproximator

# Define the target function
def target_function(x):
    return 5 * np.sin(x) + 4 * np.cos(3 * x)

# Function to split the interval
def split_interval(start: float, end: float, num_subintervals: int) -> List[tuple]:
    """
    Split the interval [start, end] into num_subintervals subintervals.
    """
    points = np.linspace(start, end, num_subintervals + 1)
    intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    return intervals

def piecewise_chebyshev_approximation(
    tar_func: Callable[[float], float],
    start: float,
    end: float,
    num_subintervals: int,
    degree: int,
    sampling_option: str = 'Chebyshev',  # 'uniform' or 'Chebyshev'
    perturbation_magnitude: float = 0.0,  # Maximum magnitude of perturbation
    perturbation_probability: float = 0.0  # Probability of perturbing each sample point
):
    # Split the interval
    intervals = split_interval(start, end, num_subintervals)
    approximations = []
    segment_data_list = []

    for idx, (a, b) in enumerate(intervals):
        # Select sampling nodes
        if sampling_option == 'Chebyshev':
            # Chebyshev nodes
            nodes = np.cos((2 * np.arange(1, degree + 2) - 1) / (2 * (degree + 1)) * np.pi)
            nodes = 0.5 * (nodes + 1) * (b - a) + a  # Map to [a, b]
        elif sampling_option == 'uniform':
            # Uniform sampling nodes
            nodes = np.linspace(a, b, degree + 1)
        else:
            raise ValueError("Invalid sampling_option. Choose 'uniform' or 'Chebyshev'.")

        # Perturb the nodes
        nodes_perturbed = nodes.copy()
        is_perturbed = np.random.rand(len(nodes)) < perturbation_probability
        perturbations = (np.random.rand(len(nodes)) * 2 - 1) * perturbation_magnitude
        nodes_perturbed[is_perturbed] += perturbations[is_perturbed]
        # Ensure the perturbed nodes are within [a, b]
        nodes_perturbed = np.clip(nodes_perturbed, a, b)

        # Compute perturbed function values
        values = tar_func(nodes_perturbed)
        # Optionally, you could also perturb the function values themselves
        # For this example, let's perturb the function values at perturbed nodes
        values_perturbed = values.copy()
        values_perturbed[is_perturbed] += (np.random.rand(np.sum(is_perturbed)) * 2 - 1) * perturbation_magnitude

        # Create the approximator instance and fit using perturbed nodes and values
        approximator = ChebyshevApproximator(tar_func, a, b, degree)
        approx_func = approximator.fit_nodes(nodes_perturbed, values_perturbed)
        approximations.append((approx_func, (a, b)))

        # Compute error
        x_test = np.linspace(a, b, 100)
        y_true = tar_func(x_test)
        y_approx = approx_func(x_test)
        error = np.max(np.abs(y_true - y_approx))

        # Save segment data
        segment_data = {
            'a': a,
            'b': b,
            'nodes': nodes_perturbed,
            'original_nodes': nodes,
            'is_perturbed': is_perturbed,
            'approx_func': approx_func,
            'error': error,
            'index': idx,
            'values': values_perturbed
        }
        segment_data_list.append(segment_data)

    def piecewise_func(x):
        # Define the piecewise function
        x = np.array(x)
        y = np.zeros_like(x)
        for i in range(len(x)):
            xi = x[i]
            # Find which subinterval xi belongs to
            for approx_func, (a, b) in approximations:
                if a <= xi <= b:
                    y[i] = approx_func(xi)
                    break
        return y

    return piecewise_func, segment_data_list

if __name__ == '__main__':
    # Set parameters
    start = 1
    end = 5
    degrees = [1, 2, 3, 4]  # Polynomial degrees
    num_subintervals_list = [2, 4, 8]  # Number of subintervals

    # Control perturbation parameters
    perturbation_magnitude = 0.8  # Maximum perturbation magnitude
    perturbation_probability = 0.3  # Probability of perturbing each sample point

    # Create test points for plotting
    x_vals = np.linspace(start, end, 1000)
    y_true = target_function(x_vals)

    # Plot results
    for degree in degrees:
        for num_subintervals in num_subintervals_list:
            plt.figure(figsize=(12, 8))
            # Get piecewise approximation function and segment data
            approx_func, segment_data_list = piecewise_chebyshev_approximation(
                tar_func=target_function,
                start=start,
                end=end,
                num_subintervals=num_subintervals,
                degree=degree,
                sampling_option='Chebyshev',  # or 'uniform'
                perturbation_magnitude=perturbation_magnitude,
                perturbation_probability=perturbation_probability
            )
            y_approx = approx_func(x_vals)

            # Plot the approximation function
            plt.plot(x_vals, y_approx, label=f'Approximation (Max Error: {max([sd["error"] for sd in segment_data_list]):.4f})')

            # Plot each segment's sampling points and annotate error
            for segment_data in segment_data_list:
                nodes = segment_data['nodes']
                values = segment_data['values']
                is_perturbed = segment_data['is_perturbed']
                idx = segment_data['index']
                errors = segment_data['error']
                # Assign a different color for each segment
                color = plt.cm.rainbow(idx / num_subintervals)
                # Plot unperturbed nodes
                plt.scatter(nodes[~is_perturbed], values[~is_perturbed], color=color, marker='o', s=50)
                # Plot perturbed nodes
                plt.scatter(nodes[is_perturbed], values[is_perturbed], color='red', marker='o', s=50)
                # Add legend entries
                plt.scatter([], [], color=color, marker='o', label=f'Segment {idx+1} Error: {errors:.4f}')

            # Add legend entry for perturbed nodes
            plt.scatter([], [], color='red', marker='o', label='Perturbed Nodes')

            # Plot the true function
            plt.plot(x_vals, y_true, 'k--', label='True Function')
            plt.title(f'Piecewise Chebyshev Approximation (Degree {degree}, Segments {num_subintervals})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            path_save = osp.join(SCRIPT_CONFITG.PATH_DATA, f'piecewise_chebyshev_degree_{degree}_segments_{num_subintervals}.png')
            plt.savefig(path_save)
            plt.show()