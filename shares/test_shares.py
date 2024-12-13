import pytest, os
from shares import (
    build_graph,
    find_wccs,
    detect_cycle,
    process_tree_component,
    process_cycle_component,
    find_max_shares,
    read_input_from_file,
    write_output_to_file
)

# Let's use before hooks to create the input file
@pytest.fixture
def input_file(tmp_path):
    input_file = tmp_path / "sample_input.txt"
    input_data = """6
10
20 1
30 1
40 3
50 3
60 5"""
    input_file.write_text(input_data)
    return input_file

# Let's use before hooks to create the output file
@pytest.fixture
def output_file(tmp_path):
    return tmp_path / "sample_output.txt"


# Let's use before hooks to build one graph for testing
@pytest.fixture
def graph():
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    return build_graph(edges)

# Test build_graph with empty edges
def test_build_graph_empty_edges():
    edges = []
    expected_graph = {}
    assert build_graph(edges) == expected_graph


# Test build_graph with one edge
def test_build_graph_one_edge():
    edges = [(1, 0)]
    expected_graph = {1: [0], 0: [1]}
    assert build_graph(edges) == expected_graph


# Test build_graph with multiple edges
def test_build_graph_multiple_edges():
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4), (6, 2)]
    expected_graph = {
        1: [0],
        0: [1, 2],
        2: [0, 3, 4, 6],
        3: [2],
        4: [2, 5],
        5: [4],
        6: [2]
    }
    assert build_graph(edges) == expected_graph

# Test find_wccs for 1 connected component
def test_find_wccs():
    graph = {
        1: [0],
        0: [1, 2],
        2: [0, 3, 4],
        3: [2],
        4: [2, 5],
        5: [4]
    }
    expected_wccs = [[1, 0, 2, 4, 5, 3]]
    result = find_wccs(graph)
    assert len(result) == 1
    assert result == expected_wccs


# Test find_wccs for multiple connected components
def test_find_wccs_multiple_components():
    graph = {
        1: [0],
        0: [1],
        2: [3, 5, 6],
        3: [2, 4],
        4: [3, 5],
        5: [4, 2],
        6: [2]
    }
    expected_wccs = [[1, 0], [2, 6, 5, 4, 3]]
    result = find_wccs(graph)
    assert len(result) == 2
    assert result == expected_wccs

# Test find_wcc() for only one graph node
def test_find_wccs_one_node():
    graph = {1: []}
    expected_wccs = [[1]]
    result = find_wccs(graph)
    assert len(result) == 1
    assert result == expected_wccs


# Test detect_cycle
def test_detect_cycle():
    graph = {1: [0], 0: [1], 2: [3, 5, 6], 3: [2, 4], 4: [3, 5], 5: [4, 2], 6: [2]}
    component = [1, 0]
    assert detect_cycle(component, graph) is False
    
    component = [2, 6, 5, 4, 3]
    assert detect_cycle(component, graph) is True


# Test process_tree_component
def test_process_tree_component():
    graph = {
        0: [1, 2],
        1: [0],
        2: [0, 3, 4],
        3: [2],
        4: [2, 5],
        5: [4]
    }
    shares = [10, 20, 30, 40, 50, 60]
    tree = [0, 1, 2, 3, 4, 5]
    max_shares, selected_shareholders = process_tree_component(tree, graph, shares)
    assert max_shares == 120
    assert sorted(selected_shareholders) == [1, 3, 5]


# Test process_tree_component with only one node
def test_process_tree_component_one_node():
    graph = {0: []}
    shares = [10]
    tree = [0]
    max_shares, selected_shareholders = process_tree_component(tree, graph, shares)
    assert max_shares == 10
    assert selected_shareholders == [0]


# Test process_tree_component with two nodes
def test_process_tree_component_two_nodes():
    graph = {0: [1], 1: [0]}
    shares = [10, 20]
    tree = [0, 1]
    max_shares, selected_shareholders = process_tree_component(tree, graph, shares)
    assert max_shares == 20
    assert selected_shareholders == [1]

# Test process_tree_component with 0 nodes
def test_process_tree_component_zero_nodes():
    graph = {}
    shares = []
    tree = []
    max_shares, selected_shareholders = process_tree_component(tree, graph, shares)
    assert max_shares == 0
    assert selected_shareholders == []


# Test process_cycle_component
def test_process_cycle_component():
    shares = [10, 20, 30, 40, 50, 60, 70]
    edges = [(1, 0), (2, 3), (3, 4), (4, 5), (5, 2), (6, 2)]
    graph = build_graph(edges)
    components = find_wccs(graph)
    component = [comp for comp in components if len(comp) > 2][0]
    max_shares, selected_shareholders = process_cycle_component(component, graph, shares)
    assert max_shares == 170
    assert sorted(selected_shareholders) == [3, 5, 6]


# Test find_max_shares
def test_find_max_shares():
    shares = [10, 20, 30, 40, 50, 60]
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 120
    assert sorted(selected_shareholders) == [2, 4, 6]


# Test find_max_shares with only one node
def test_find_max_shares_one_node():
    shares = [10]
    edges = []
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 10
    assert selected_shareholders == [1]


# Test find_max_shares with two nodes
def test_find_max_shares_two_nodes():
    shares = [10, 20]
    edges = [(1, 0)]
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 20
    assert selected_shareholders == [2]


# Test find_max_shares with 0 nodes
def test_find_max_shares_zero_nodes():
    shares = []
    edges = []
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 0
    assert selected_shareholders == []

# Test find_max_shares with invalid spying relationships
def test_find_max_shares_invalid_edges():
    shares = [10, 20, 30, 40, 50, 60]
    edges = [(1, 0), (2,  1), (3, 2), (4, 3), (5, 4), (0, 5)]
    with pytest.raises(ValueError):
        find_max_shares(shares, edges)


# Test read_input_from_file
def test_read_input_from_file(tmp_path):
    input_file = tmp_path / "input.txt"
    input_data = """6
10
20 1
30 1
40 3
50 3
60 5"""
    input_file.write_text(input_data)
    
    shares, edges = read_input_from_file(input_file)
    assert shares == [10, 20, 30, 40, 50, 60]
    assert edges == [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]


# Test write_output_to_file
def test_write_output_to_file(tmp_path):
    output_file = tmp_path / "output.txt"
    output = (120, [1, 3, 5])
    write_output_to_file(output_file, output)
    
    content = output_file.read_text().strip()
    expected_content = "Maximum Shares: 120\nSelected Shareholders: [1, 3, 5]"
    assert content == expected_content


# Test the main functionality
def test_main_functionality():
    shares = [10, 20, 30, 40, 50, 60]
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 120
    assert sorted(selected_shareholders) == [2, 4, 6]

