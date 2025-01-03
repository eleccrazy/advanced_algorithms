import random

def generate_input_file(file_path: str, n: int, share_range: tuple):
    """
    Generate an input file for the Shares Problem.

    Parameters:
        file_path (str): Path to save the input file.
        n (int): Number of shareholders.
        share_range (tuple): Range of shares as (min_share, max_share).
    
    Returns:
        None
    """
    if n < 1:
        raise ValueError("Number of shareholders must be at least 1.")
    
    if share_range[0] <= 0 or share_range[1] <= 0:
        raise ValueError("Shares must be positive integers.")

    # Step 1: Generate shares
    shares = [random.randint(*share_range) for _ in range(n)]

    # Step 2: Generate "spying on" relationships
    edges = []
    for i in range(1, n):  # Start from the second node
        target = random.randint(1, i)  # Spy on any previous node
        edges.append((i + 1, target))  # 1-based indexing

    # Step 3: Write to the file
    with open(file_path, "w") as file:
        file.write(f"{n}\n")  # Write the number of shareholders
        # Write the root node (first shareholder) without spying
        file.write(f"{shares[0]}\n")
        # Write the other shareholders with their spying relationships
        for i, share in enumerate(shares[1:], start=1):
            spied_on = [v for u, v in edges if u == i + 1][0]  # Find the target node
            file.write(f"{share} {spied_on}\n")


if __name__ == "__main__":
    print("=== Shares Problem Input File Generator ===")
    file_path = input("Enter the genearated input file path (e.g., 'input.txt'): ").strip()
    n = int(input("Enter the number of shareholders: "))
    min_share = int(input("Enter the minimum share value: "))
    max_share = int(input("Enter the maximum share value: "))

    try:
        generate_input_file(file_path, n, (min_share, max_share))
    except ValueError as e:
        print(f"Error: {e}")