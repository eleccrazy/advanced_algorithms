from input_file_generator import generate_input_file
import matplotlib.pyplot as plt
from shares import find_max_shares, read_input_from_file, validate_input
import time
from typing import List, Tuple

def generate_analysis_files(file_prefix: str, size: int, share_range: Tuple[int, int], num_instances: int) -> List[str]:
    """
    Generate multiple input files of a specific size for the Shares Problem.

    Parameters:
        file_prefix (str): Prefix for the generated file names.
        size (int): Number of nodes in the graph.
        share_range (Tuple[int, int]): Tuple representing the range of share values (min, max).
        num_instances (int): Number of instances to generate.

    Returns:
        List[str]: List of file paths for the generated input files.
    """
    file_paths = []
    for instance in range(num_instances):
        file_path = f"{file_prefix}_{size}_instance_{instance + 1}.txt"
        generate_input_file(file_path, size, share_range)
        file_paths.append(file_path)
    return file_paths

def measure_time(file_path: str) -> float:
    """
    Measure the execution time of the Shares Problem algorithm for a given input file.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        float: Execution time in seconds.
    """
    start_time = time.perf_counter()
    try:
        validate_input(file_path)
        shares, edges = read_input_from_file(file_path)
        find_max_shares(shares, edges)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    end_time = time.perf_counter()
    return end_time - start_time

def measure_average_runtime(file_prefix: str, size: int, share_range: Tuple[int, int], num_instances: int) -> float:
    """
    Measure the average execution time over multiple instances of the Shares Problem.

    Parameters:
        file_prefix (str): Prefix for the generated file names.
        size (int): Number of nodes in the graph.
        share_range (Tuple[int, int]): Tuple representing the range of share values (min, max).
        num_instances (int): Number of instances to generate and measure.

    Returns:
        float: Average execution time in seconds.
    """
    file_paths = generate_analysis_files(file_prefix, size, share_range, num_instances)
    total_time = 0.0

    for file_path in file_paths:
        total_time += measure_time(file_path)

    return total_time / num_instances

def plot_analysis(sizes: List[int], empirical_runtimes: List[float], theoretical_runtimes: List[float]) -> None:
    """
    Plot the empirical runtime against theoretical runtime for the Shares Problem.

    Parameters:
        sizes (List[int]): List of input sizes (number of nodes).
        empirical_runtimes (List[float]): Measured runtime values for each input size.
        theoretical_runtimes (List[float]): Theoretical runtime values for each input size.

    Returns:
        None
    """
    plt.plot(sizes, empirical_runtimes, label="Empirical Runtime", marker="o")
    plt.plot(sizes, theoretical_runtimes, label="Theoretical Runtime (O|V|)", linestyle="--")
    plt.xlabel("Input Size (Number of Nodes)")
    plt.ylabel("Runtime (Seconds)")
    plt.title("Empirical vs. Theoretical Runtime for Shares Problem")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_scaling_factor(sizes: List[int], empirical_runtimes: List[float]) -> float:
    """
    Calculate a dynamic scaling factor based on empirical runtime results.

    Parameters:
        sizes (List[int]): List of input sizes (number of nodes).
        empirical_runtimes (List[float]): Measured runtime values for each input size.

    Returns:
        float: Adjusted scaling factor.
    """
    return sum(empirical_runtime / size for size, empirical_runtime in zip(sizes, empirical_runtimes)) / len(sizes)

def main() -> None:
    """
    Main function to conduct runtime analysis for the Shares Problem.

    Parameters:
        None

    Returns:
        None
    """
    # Define input sizes and share range
    sizes = [10, 100, 200, 500, 1000]
    share_range = (10, 150)
    num_instances = 10  # Number of instances to average over

    # Measure runtime for each input size
    empirical_runtimes = []
    for size in sizes:
        avg_runtime = measure_average_runtime("input", size, share_range, num_instances)
        empirical_runtimes.append(avg_runtime)
        print(f"Average runtime for {size} nodes: {avg_runtime:.4f} seconds")

    # Dynamically calculate scaling factor based on observed empirical results
    scaling_factor = calculate_scaling_factor(sizes, empirical_runtimes)
    print(f"Adjusted Scaling Factor: {scaling_factor:.6f}")
    theoretical_runtimes = [size * scaling_factor for size in sizes]

    # Plot the analysis
    plot_analysis(sizes, empirical_runtimes, theoretical_runtimes)

if __name__ == "__main__":
    main()
