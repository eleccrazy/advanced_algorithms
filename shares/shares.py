"""File: shares.py
This module contains the implementation of the shares problem.
Author: Gizachew Bayness Kassa
"""
from collections import defaultdict
import sys
import os


def build_graph(edges: list) -> dict:
    """
    Build an adjacency list representation of the directed graph.

    Parameters:
        edges (list): List of tuples representing directed edges.
        
    Returns:
        dict: Adjacency list representation of the directed graph.
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # Assuming an undirected graph
    return graph


def find_wccs(graph: dict) -> list:
    """
    Find the weakly connected components of the directed graph.

    Parameters:
        graph (dict): Adjacency list representation of the directed graph.

    Returns:
        list: List of weakly connected components.
    """
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            component = []
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current])
            components.append(component)
    return components


def detect_cycle(component: list, graph: dict) -> bool:
    """
    Detect a cycle in the given component of the graph.

    Parameters:
        component (list): List of nodes in the connected component.
        graph (dict): Adjacency list representation of the directed graph.

    Returns:
        bool: True if a cycle is detected, False otherwise.
    """
    visited = set()
    for node in component:
        if node not in visited:
            stack = [(node, None)]
            while stack:
                current, parent = stack.pop()
                if current in visited:
                    return True
                visited.add(current)
                for neighbor in graph[current]:
                    if neighbor != parent:
                        stack.append((neighbor, current))
    return False


def process_tree_component(tree: list, graph: dict, shares: list) -> tuple:
    """
    Process a tree component using dynamic programming to find the maximum shares
    and the selected shareholders.

    Parameters:
        tree (list): List of nodes in the tree component.
        graph (dict): Adjacency list representation of the directed graph.
        shares (list): List of integers representing the shares.

    Returns:
        tuple: Maximum shares and the list of selected shareholders.
    """
    # Check if the tree is empty
    if not tree:
        return 0, []
    
    # DP arrays: dp[node][0] = max shares if node is not included
    #            dp[node][1] = max shares if node is included
    dp = {node: [0, 0] for node in tree}
    included = {node: False for node in tree}  # Track included nodes for backtracking

    def iterative_dfs(root: int) -> None:
        """
        Perform iterative DFS to calculate DP values for the tree.

        Parameters:
            root (int): The starting node of the tree.

        Modifies:
            dp (dict): Updates the DP values for each node in the tree.

        Returns:
            None
        """
        stack = [root]
        parent = {root: None}  # Track parent nodes to avoid revisiting
        post_order = []  # To ensure post-order traversal for DP updates
        visited = set()  # Avoid revisiting nodes during traversal

        # Forward pass: Traverse the tree
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                post_order.append(node)
                for child in graph.get(node, []):
                    if child in tree and child not in visited:
                        stack.append(child)
                        parent[child] = node

        # Backward pass: Update DP values in post-order
        for node in reversed(post_order):
            dp[node][1] = shares[node]  # Include the current node
            for child in graph.get(node, []):
                if child in tree and child != parent[node]:
                    dp[node][0] += max(dp[child][0], dp[child][1])  # Exclude node
                    dp[node][1] += dp[child][0]  # Include node (exclude children)

    def iterative_backtrack(root: int, include_root: bool) -> None:
        """
        Iterative implementation of backtracking to reconstruct the selected shareholders.

        Parameters:
            root (int): The starting node of the tree.
            include_root (bool): Whether to include the root node in the selection.

        Modifies:
            included (dict): Updates the included dictionary to track selected nodes.
        """
        stack = [(root, include_root)]  # Initialize stack with the root node and its inclusion decision
        visited = set()  # Avoid revisiting nodes during backtracking

        while stack:
            node, include = stack.pop()  # Process the current node
            if node in visited:
                continue
            visited.add(node)

            if include:
                included[node] = True  # Mark the node as included
                for child in graph.get(node, []):
                    if child in tree and not included[child]:
                        stack.append((child, False))  # Exclude all children if the node is included
            else:
                for child in graph.get(node, []):
                    if child in tree and not included[child]:
                        # Include children if their "include" DP value is better
                        stack.append((child, dp[child][1] > dp[child][0]))

    # Start DFS to compute DP values
    root = tree[0]
    iterative_dfs(root)

    # Determine the maximum shares
    max_shares = max(dp[root][0], dp[root][1])

    # Backtrack to find the selected shareholders
    if dp[root][1] > dp[root][0]:
        iterative_backtrack(root, True)
    else:
        iterative_backtrack(root, False)

    # Gather the selected shareholders
    selected_shareholders = [node for node in tree if included[node]]

    return max_shares, selected_shareholders


def process_cycle_component(component: list, graph: dict, shares: list) -> tuple:
    """
    Process a cyclic component by simulating breaking the cycle into two configurations.

    Parameters:
        component (list): List of nodes in the cycle component.
        graph (dict): Adjacency list representation of the directed graph.
        shares (list): List of integers representing the shares.
    Returns:
        tuple: Maximum shares and the list of selected shareholders.
    """
    # Step 1: Detect cycle nodes
    cycle_nodes = detect_cycle_nodes(component, graph)

    # Step 2: Choose a representative node
    representative_node = cycle_nodes[0]

    # Configuration 1: Include the representative node
    forest_include = remove_node_and_neighbors(representative_node, component, graph)
    max_include, shareholders_include = process_forest(forest_include, graph, shares)

    # Add the representative node's share
    max_include += shares[representative_node]
    shareholders_include.append(representative_node)

    # Configuration 2: Exclude the representative node
    forest_exclude = remove_node(representative_node, component)
    max_exclude, shareholders_exclude = process_forest(forest_exclude, graph, shares)

    # Compare results
    if max_include > max_exclude:
        return max_include, shareholders_include
    else:
        return max_exclude, shareholders_exclude


def detect_cycle_nodes(component: list, graph: dict) -> list:
    """
    Detect all nodes in a cycle within the component.

    Parameters:
        component (list): List of nodes in the component.
        graph (dict): Adjacency list representation of the graph.

    Returns:
        list: List of nodes that are part of the cycle.
    """
    visited = set()
    stack = set()
    cycle_nodes = set()

    def dfs(node, parent):
        visited.add(node)
        stack.add(node)
        for neighbor in graph[node]:
            if neighbor in component:
                if neighbor not in visited:
                    dfs(neighbor, node)
                elif neighbor != parent and neighbor in stack:
                    cycle_nodes.update(stack)
        stack.remove(node)

    for node in component:
        if node not in visited:
            dfs(node, None)

    return list(cycle_nodes)


def remove_node_and_neighbors(node: int, component: list, graph: dict) -> list:
    """
    Remove a node and its neighbors from the component.

    Parameters:
        node (int): The node to remove.
        component (list): List of nodes in the component.
        graph (dict): Adjacency list representation of the graph.

    Returns:
        list: A list of smaller components (forests or trees).
    """
    removed_nodes = set([node] + graph[node])  # Remove node and its neighbors
    return [n for n in component if n not in removed_nodes]


def remove_node(node: int, component: list) -> list:
    """
    Remove a single node from the component.

    Parameters:
        node (int): The node to remove.
        component (list): List of nodes in the component.
        graph (dict): Adjacency list representation of the graph.

    Returns:
        list: A list of smaller components (forests or trees).
    """
    return [n for n in component if n != node]


def process_forest(forest: list, graph: dict, shares: list) -> tuple:
    """
    Process a forest (list of disconnected tree components).

    Parameters:
        forest (list): List of nodes in the forest.
        graph (dict): Adjacency list representation of the graph.
        shares (list): List of integers representing the shares.

    Returns:
        tuple: Maximum shares and the list of selected shareholders.
    """
    total_max_shares = 0
    total_shareholders = []
    visited = set()

    def dfs_tree(root):
        tree = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                tree.append(node)
                for child in graph[node]:
                    if child not in visited and child in forest:
                        stack.append(child)
        return tree

    for node in forest:
        if node not in visited:
            tree = dfs_tree(node)
            max_shares, shareholders = process_tree_component(tree, graph, shares)
            total_max_shares += max_shares
            total_shareholders.extend(shareholders)

    return total_max_shares, total_shareholders


def find_max_shares(shares: list, edges: list) -> tuple:
    """
    Find the maximum shares that can be selected by independent shareholders.

    Parameters: 
        shares (list): List of integers representing the shares.
        edges (list): List of tuples representing directed edges between shareholders.

    Returns:
        tuple: Maximum shares that can be selected and the list of selected shareholders.
    """
    # Check if there is only one shareholder
    if len(shares) == 1:
        return shares[0], [1]
    graph = build_graph(edges)
    components = find_wccs(graph)
    # Check if the graph is valid (no cycles if there is only one component)
    if len(components) == 1 and detect_cycle(components[0], graph):
        # Raise an error if the graph has a cycle
        raise ValueError("Invalid spying relationships: There must be one shareholder without spying.")
    total_max_shares = 0
    selected_shareholders = []

    for component in components:
        if detect_cycle(component, graph):
            max_shares, shareholders = process_cycle_component(component, graph, shares)
        else:
            max_shares, shareholders = process_tree_component(component, graph, shares)

        total_max_shares += max_shares
        selected_shareholders.extend(shareholders)

        selected_shareholders.sort() # Sort the selected shareholders
        # Convert the selected shareholders to 1-based indexing
        selected_shareholders = [shareholder + 1 for shareholder in selected_shareholders]

    return total_max_shares, selected_shareholders


def read_input_from_file(file_path: str) -> tuple:
    """
    Reads input from a file and converts it into shares and edges for further processing.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        tuple: A list of shares and a list of edges.
    """
    with open(file_path, 'r') as file:
            lines = file.readlines()
    # First line contains the number of shareholders (not directly used here)
    n = int(lines[0].strip())

    shares = []  # List to store the shares
    edges = []   # List to store the edges (spying relationships)

    # Process each line of shareholder information
    for i, line in enumerate(lines[1:], start=0):  # Start with index 0 for shares
        data = line.strip().split()
        shares.append(int(data[0]))  # First number is the number of shares

        # Second number, if present, indicates the shareholder being spied on
        if len(data) > 1:
            spied_on = int(data[1]) - 1  # Convert to 0-based index
            edges.append((i, spied_on))  # Add the edge (i -> spied_on)

    return shares, edges


def write_output_to_file(file_path: str, output: tuple)-> None:
    """
    Writes the output (maximum shares and selected shareholders) to a file.

    Parameters:
        file_path (str): Path to the output file.
        output (tuple): A tuple containing the maximum shares and the list of selected shareholders.
                        Format: (max_shares, selected_shareholders)

    Returns:
        None
    """
    max_shares, selected_shareholders = output
    with open(file_path, 'w') as file:
        file.write(f"Maximum Shares: {max_shares}\n")
        file.write(f"Selected Shareholders: {selected_shareholders}\n")


def validate_input(file_path: str) -> bool:
    """
    Validates the input file for the Shares Problem.
    Enforces:
      1. Only the first shareholder can be the root (without spying).
      2. All other shareholders must have exactly one valid "spying on" relationship.
      3. Spying targets must be within the valid range [1, n].
    
    Parameters:
        file_path (str): Path to the input file.

    Returns:
        bool: True if valid; raises ValueError otherwise.
    """
    if not os.path.exists(file_path):
        raise ValueError("Invalid input: File does not exist.")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    n = int(lines[0].strip())  # Number of shareholders
    if n < 1:
        raise ValueError("Invalid input: Number of shareholders must be at least 1.")

    # Validate the first shareholder (root)
    first_line = lines[1].strip().split()
    if len(first_line) != 1:
        raise ValueError("Invalid input: Only the first shareholder (node 1) can be the root without spying.")

    # Validate all other shareholders
    for i, line in enumerate(lines[2:], start=2):  # Start checking from the second shareholder
        parts = line.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid input: Shareholder {i} must spy on exactly one other shareholder.")
        
        target = int(parts[1])
        if target < 1 or target > n:
            raise ValueError(f"Invalid input: Shareholder {i} spies on non-existent shareholder {target}.")
        
    return True


def read_filenames_from_console() -> tuple:
    """
    Read input and output filenames from the console.

    Returns:
        tuple: Input filename and output filename.
    """
    input_file = input("Enter the input filename: ")
    output_file = input("Enter the output filename: ")

    # Check input file and output file are of txt format
    if not input_file.endswith(".txt") or not output_file.endswith(".txt"):
        raise ValueError("Input and output files must be of .txt format.")

    return input_file, output_file


def main():
    """Main function to test the shares problem implementation."""
    # Read input and output filenames from the command line
    try:
        input_file, output_file = read_filenames_from_console()
        # Read input from a file
        # Validate the input file
        try:
            validate_input(input_file)
            shares, edges = read_input_from_file(input_file)
            # Write output to a file
            write_output_to_file(output_file, find_max_shares(shares, edges))
        except ValueError as e:
            print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
