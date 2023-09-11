import networkx as nx
from typing import Optional, Any, TypedDict


def func1(args):
    print(f'Running func1 with external args {args}')
    func1_result = 'func 1 result'
    return func1_result


def func2(arg):
    print(f'Using {arg}')
    func2_result = 'func 2 result'
    return func2_result


def func3(arg):
    print(f'Using {arg}')
    a = 'func 3 result a'
    b = 'func 3 result b'
    return a, b


def func4(arg):
    print(f'Using {arg}')
    a = 'func 4 result a'
    b = 'func 4 result b'
    return a, b


def func5(arg):
    print(f'Using {arg}')
    func5_result = 'func 5 result'
    return func5_result


def func6(arg):
    print(f'Using {arg}')
    func6_result = 'func 6 result'
    return func6_result


def func7(arg):
    print(f'Using {arg}')
    func7_result = 'func 7 result'
    return func7_result


def func8(arg1, arg2):
    print(f'Using {arg1} as first dep, {arg2} as second dep')
    func8_result = 'func 8 result'
    return func8_result


def func9(arg):
    print(f'Using {arg}')
    func9_result = 'func 9 result'
    return func9_result


def func10(arg):
    print(f'Using {arg}')
    func10_result = 'func 10 result'
    return func10_result


ExternalArgumentInterface = TypedDict(
    'ExternalArgumentInterface', {"args": dict})


def create_node(func: callable, node_name: Optional[str] = None, external_argument: Optional[Any | ExternalArgumentInterface] = None):
    """
    Create node with attributes base on interface

    @ Node interface:
    - node_name `(Optional[str])`: Name of the node. `Default func.__name__`
    - executor `(callable)`: Function to be called. LIMITATION: If the function is not root, it not receive external argument during runtime
    - external_argument `(Optional[Any | {"args": dict}])`: External argument for root node. If multiple external argument provided, use
    keyword argument
    """
    if node_name is not None:
        return (node_name, {"executor": func, "result": None})
    return (func.__name__, {"executor": func, "result": None})


def add_functions_to_graph(graph: nx.DiGraph, fns: list[callable]):
    """
    Utility function to add all functions to `ChainExecutor` class, with all default parameters
    """
    graph.add_nodes_from([create_node(i) for i in fns])


def create_edge(g: nx.DiGraph, u, v, dependency: Optional[int] = None, dependency_map: Optional[tuple[int]] = None):
    """
    Create edge between two nodes `u` & `v` base in edge attribute interface

    @ Edge interface:

    - u `(str)`: Name of predecessor node
    - v `(str)`: Name of child node
    - dependency `(Optional[int])`: Use in case of `(multiple return) -> (pure function)`. If provided,
        - `u`'s executor function must return multiple results
        - `v`'s executor function must be a pure function
        - `v` will take the result from `u` at `dependency` index 

    - dependency_map `(Tuple[int | None, int])`: Use in case of `(multiple return | single return) -> (multi argument)`. If provided,
        - If `u` is `multiple return` type, provide `dependency_map[0]` with int
        - `v`'s executor function must have multi predecessor node
        - `v` will take the result from `u` at `dependency_map[0]` index, as `dependency_map[1]` argument index
    """
    if dependency is not None:
        g.add_edge(u, v, dependency=dependency)
        return
    if dependency_map is not None:
        g.add_edge(u, v, dependency_map=dependency_map)
        return
    g.add_edge(u, v)
    return


G = nx.DiGraph()
G.add_nodes_from([
    ('func1', {"executor": func1, "result": None}),
    ('func2', {"executor": func2, "result": None}),
    ('func3', {"executor": func3, "result": None}),
    ('func4', {"executor": func4, "result": None}),
    ('func5', {"executor": func5, "result": None}),
    ('func6', {"executor": func6, "result": None}),
    ('func7', {"executor": func7, "result": None}),
    ('func8', {"executor": func8, "result": None}),
    ('func9', {"executor": func9, "result": None}),
    ('func10', {"executor": func10, "result": None}),
])

G.add_edge('func1', 'func2')
G.add_edge('func2', 'func3')
G.add_edge('func3', 'func4')
G.add_edge('func3', 'func5', dependency=1)
G.add_edge('func4', 'func6', dependency=1)
G.add_edge('func5', 'func7')
G.add_edge('func7', 'func8', dependency_map=(0, 0))
G.add_edge('func4', 'func8', dependency_map=(0, 1))
G.add_edge('func8', 'func9')
G.add_edge('func9', 'func10')


class ChainExecutor:
    def __init__(self, dependency_graph: nx.DiGraph) -> None:
        self.g = dependency_graph

    def get_predecessors(self, node_name) -> list[str]:
        return [i for i in self.g.predecessors(node_name)]

    def sort_and_return_args_array(self, args_array):
        sorted_args_array = sorted(args_array, key=lambda x: x[0])
        return [i[1] for i in sorted_args_array]

    def set_external_dependencies(self, node_name, args):
        node_predecessor_labels = self.get_predecessors(node_name)
        if len(node_predecessor_labels):
            raise Exception(
                "Not a root node, cannot add external dependencies")
        self.g.nodes[node_name]["args"] = args

    def get_node_data(self, node_name):
        return self.g.nodes[node_name]['result']

    def execute(self, func: str):
        current_node_result = self.g.nodes[func]['result']
        if current_node_result is not None:
            return current_node_result
        else:
            node_predecessor_labels = self.get_predecessors(func)
            if len(node_predecessor_labels) > 0:
                print(
                    f'Found {len(node_predecessor_labels)} predecessor(s) for {func}: {", ".join(node_predecessor_labels)}')
                args_array = []
                for label in node_predecessor_labels:
                    edge_dependency = self.g.get_edge_data(label, func)
                    if 'dependency' in edge_dependency:
                        dependency_index = edge_dependency["dependency"]
                        arg_index = 0
                        print(
                            f'Using {dependency_index} positional result from {label} as pure dependency for {func}')
                    if 'dependency_map' in edge_dependency:
                        dependency_index = edge_dependency["dependency_map"][
                            0] if edge_dependency["dependency_map"][0] is not None else None
                        arg_index = edge_dependency["dependency_map"][1]
                        print(
                            f'Using {dependency_index} positional result of {label} as {arg_index} positional argument for {func}')
                    else:
                        dependency_index = None
                        arg_index = 0
                        print(
                            f'Using pure dependency between {label} and {func}')

                    def assert_prev_result():
                        prev_result = self.g.nodes[label]['result']
                        if prev_result is None:
                            self.execute(label)

                    assert_prev_result()
                    prev_result = self.g.nodes[label]['result'] if dependency_index is None else self.g.nodes[label]['result'][dependency_index]
                    args_array.append(
                        (arg_index, prev_result))
                sorted_args_array = self.sort_and_return_args_array(args_array)
                fn_result = self.g.nodes[func]['executor'](*sorted_args_array)
            else:
                print('First node reach')
                if 'args' in self.g.nodes[func]:
                    if type(self.g.nodes[func]['args']) == dict:
                        fn_result = self.g.nodes[func]['executor'](
                            **self.g.nodes[func]['args'])
                    else:
                        fn_result = self.g.nodes[func]['executor'](
                            self.g.nodes[func]['args'])
                else:
                    fn_result = self.g.nodes[func]['executor']()

            self.g.nodes[func]['result'] = fn_result

        return self.g.nodes[func]['result']


# print('Register chain executor for `func8`')
# chain_executor = ChainExecutor(G)
# chain_executor.set_external_dependencies('func1', {"args": "hello world"})
# a = chain_executor.execute('func10')
# print(f'Final answer:', a)
