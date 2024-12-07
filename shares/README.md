# Shares Problem

## Problem Description

In a joint-stock company, each shareholder spies on exactly one other shareholder, except for one shareholder who spies on no one. Each shareholder owns a certain number of shares, represented as positive integers. The objective is to find a subset of shareholders such that:

1. No two shareholders in the subset spy on one another.
2. The total sum of shares owned by the selected subset is maximized.

The company structure is represented as a directed graph where:
- **Nodes** represent shareholders.
- **Directed edges** indicate spying relationships.

This graph may contain cycles, tree-like structures, or a combination of both.

---

## Solution Approach

To solve the problem, the algorithm employs dynamic programming combined with graph decomposition:

1. **Graph Decomposition**:
   - The graph is decomposed into Weakly Connected Components (WCCs) using DFS or BFS.
   - Each WCC is classified as either containing a cycle or being acyclic (tree-like).

2. **Tree Components**:
   - For tree components, dynamic programming computes the maximum shares:
     - `dp[node][0]`: Maximum shares if the node is excluded.
     - `dp[node][1]`: Maximum shares if the node is included.
   - Backtracking identifies the selected shareholders.

3. **Cycle Components**:
   - For cyclic components, a representative node is chosen, and two configurations are simulated:
     - Include the representative node (excluding its neighbors).
     - Exclude the representative node.
   - Both configurations are converted to tree-like structures and solved using dynamic programming.

4. **Combining Results**:
   - Results from all WCCs are aggregated, ensuring no conflicts between selected shareholders.

---