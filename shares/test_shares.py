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

# Test build_graph
def test_build_graph():
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    expected_graph = {
        1: [0],
        0: [1, 2],
        2: [0, 3, 4],
        3: [2],
        4: [2, 5],
        5: [4]
    }
    assert build_graph(edges) == expected_graph


# Test find_wccs
def test_find_wccs():
    graph = {
        1: [0],
        0: [1, 2],
        2: [0, 3, 4],
        3: [2],
        4: [2, 5],
        5: [4]
    }
    expected_wccs = [[1, 0, 2, 3, 4, 5]]
    assert find_wccs(graph) == expected_wccs


# Test detect_cycle
def test_detect_cycle():
    graph = {
        0: [1],
        1: [2],
        2: [0]
    }
    component = [0, 1, 2]
    assert detect_cycle(component, graph) is True

    graph = {
        0: [1],
        1: [2]
    }
    component = [0, 1, 2]
    assert detect_cycle(component, graph) is False


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


# Test process_cycle_component
def test_process_cycle_component():
    graph = {
        0: [1],
        1: [2],
        2: [0]
    }
    shares = [10, 20, 30]
    component = [0, 1, 2]
    max_shares, selected_shareholders = process_cycle_component(component, graph, shares)
    assert max_shares == 50
    assert sorted(selected_shareholders) == [1, 2]


# Test find_max_shares
def test_find_max_shares():
    shares = [10, 20, 30, 40, 50, 60]
    edges = [(1, 0), (2, 0), (3, 2), (4, 2), (5, 4)]
    max_shares, selected_shareholders = find_max_shares(shares, edges)
    assert max_shares == 120
    assert sorted(selected_shareholders) == [1, 3, 5]


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

