import pytest
import networkx as nx
from table_reg.chain_executor import ChainExecutor

def test_full_case():
    def func1(ex1a, ex1b):
        print(
            f'Func 1 running with `{ex1a}` and `{ex1b}` as external dependencies')
        return 'Func 1 result'

    def func2(peer_dep, ex2):
        print(
            f"Func 2 running with `{peer_dep}` as peer dependency and `{ex2}` external dependencies")
        return 'Func 2 result'

    def func3(ex_arg, dep):
        print(
            f"Func 3 running with `{dep}` as peer dependency and `{ex_arg}` external dependencies")
        return 'Func 3 result'

    def func4(dep):
        print(f"Func 4 running with `{dep}` peer dependency")
        return 'Func 4 result a', 'Func 4 result b'

    def func5(dep):
        print(f"Func 5 running with `{dep}` peer dependency")
        return 'Func 5 result'

    def func6(ext_arg, peer_dep):
        print(
            f"Func 6 running with `{peer_dep}` as peer dependency and `{ext_arg}` external dependencies")
        return 'Func 6 result a', 'Func 6 result b'

    def func7(peer_dep, ext_arg):
        print(
            f"Func 7 running with `{peer_dep}` as peer dependency and `{ext_arg}` external dependencies")
        return 'Func 7 result'

    def func8(peer_1, peer_2):
        print(
            f'Func 8 running with `{peer_1}` and `{peer_2}` as peer dependency')
        return 'Func 8 result'

    def func9(ext_1, peer_dep, ext_2):
        print(
            f'Func 9 is running with `{peer_dep}` as peer dependency and `{ext_1}, {ext_2}` external dependencies')
        return 'Func 9 result'

    def func10(peer_dep_1, ext_arg, peer_dep_2):
        print(
            f'Func 9 is running with `{peer_dep_1}, {peer_dep_2}` as peer dependency and `{ext_arg}` external dependencies')
        return 'Func 10 result'

    G = nx.DiGraph()

    exe = ChainExecutor(G, print_to_console=False)

    exe.add_node(func1, args={"ex1a": "external argument 1a", "ex1b": "external argument 1b"}) \
        .add_node(func2, args={"ex2": "hello fn 2"}) \
        .add_node(func3, args={"ex_arg": "hello fn 3"}) \
        .add_node(func4) \
        .add_node(func5) \
        .add_node(func6, args={"ext_arg": "hello fn 6"}) \
        .add_node(func7, args={"ext_arg": "hello fn 7"}) \
        .add_node(func8) \
        .add_node(func9, args={"ext_1": "external argument 9a", "ext_2": "external argument 9b"}) \
        .add_node(func10, args={"ext_arg": "hello fn 10"})

    exe.add_edge('func1', 'func2') \
        .add_edge('func2', 'func3', arg_index=1) \
        .add_edge('func3', 'func4') \
        .add_edge('func4', 'func5', result_index=1) \
        .add_edge('func4', 'func6', result_index=0, arg_index=1) \
        .add_edge('func6', 'func7', result_index=1, arg_index=0) \
        .add_edge('func5', 'func8', arg_index=0) \
        .add_edge('func6', 'func8', result_index=0, arg_index=1) \
        .add_edge('func8', 'func9', arg_index=1) \
        .add_edge('func7', 'func10', arg_index=0) \
        .add_edge('func9', 'func10', arg_index=2)

    exe.compile_graph()
    exe.execute()
    assert exe.get_node_result('func10') == 'Func 10 result'