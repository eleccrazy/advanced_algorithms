"""File: shares.py
This module contains the implementation of the shares problem.
Author: Gizachew Bayness Kassa
"""
from collections import defaultdict


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
    graph = build_graph(edges)
    components = find_wccs(graph)
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


def main():
    """Main function to test the shares problem implementation."""
    # Example usage
    shares = [10, 20, 30, 40, 50, 60]
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    print(find_max_shares(shares, edges)) # Result: (120, [1, 3, 5])

    # Another example
    shares = [10, 20, 30, 40, 50, 60, 70]
    edges = [(1, 0), (2, 3), (3, 4), (4, 5), (5, 2), (6, 2)]
    print(find_max_shares(shares, edges)) # Result: (190, [1, 3, 5, 6])

    # Another example
    shares = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    edges = [(1, 0), (2, 3), (3, 4), (4, 5), (5, 2), (6, 2), (7, 6), (8, 6)]
    print(find_max_shares(shares, edges)) # Result: (290, [1, 3, 5, 7, 8])


if __name__ == "__main__":
    main()
