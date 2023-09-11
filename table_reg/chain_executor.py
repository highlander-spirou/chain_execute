from functools import partial
from inspect import getfullargspec
import networkx as nx
from networkx import topological_sort
from typing import Optional, TypedDict, Callable, Any, Union, List


class PartialFunc:
    """
    Wrapper of `partial` function that receive both positional argument and keyword argument
    """

    def __init__(self, fn: callable, defaults: Optional[dict] = None) -> None:
        self.fn = fn
        self.defaults = defaults
        self.partial_fn = self.init_partial()

    def init_partial(self):
        if self.defaults is not None:
            return partial(self.fn, **self.defaults)
        return partial(self.fn)

    def positional_kw(self, fn_args):
        """
        Receive a positional argument `args` and turn them into keyword argument 
        (which will subjected into partial_fn)
        """
        args = getfullargspec(self.fn).args
        if self.defaults is not None:
            un_init_args = [i for i in args if i not in self.defaults.keys()]
            return dict(zip(un_init_args, fn_args))
        else:
            return dict(zip(args, fn_args))

    def __call__(self, *positional_args, **kw_arg):
        if len(positional_args) > 0:
            posix_kw = self.positional_kw(positional_args)
            kw_arg.update(posix_kw)

        return self.partial_fn(**kw_arg)


class NodeAttrInterface(TypedDict):
    executor: Callable
    result: Union[None, Any]
    partial_fn: partial
    args: Optional[Union[dict, Any]]


class EdgeAttrInterface(TypedDict):
    result_index: Optional[int]
    arg_index: Optional[int]


class ChainExecutor:
    def __init__(self, graph: nx.DiGraph, print_to_console: bool) -> None:
        self.g = graph
        self.log = print_to_console

    def __node_ref(self, node_name) -> NodeAttrInterface:
        """
        Return a node reference, with type hint
        """
        return self.g.nodes[node_name]

    def __edge_ref(self, u, v) -> EdgeAttrInterface:
        """
        Return a edge reference, with type hint
        """
        return self.g.get_edge_data(u, v)

    def __get_predecessors(self, node_name) -> list[str]:
        return [i for i in self.g.predecessors(node_name)]

    def __assert_prev_result(self, label):
        """
        Recursive function to run predecessor node if it were not ran
        """
        prev_result = self.__node_ref(label)['result']
        if prev_result is None:
            self.execute(label)

    def __sort_args_array(self, args_array):
        """
        Sort args array base on the the first ordered tuple
        """
        sorted_args_array = sorted(args_array, key=lambda x: x[0])
        return [i[1] for i in sorted_args_array]

    def add_node(self, func, node_name: Optional[str] = None, args: Optional[dict] = None):
        """
        Create a node attribute tuple with default attribute

        @ Return: `tuple` with NodeAttrInterface

        - node_name `(Optional[str])`: Name of the node. `Default func.__name__`
        - executor `(callable)`: Function to be called.
        - args: `Optional[dict]`: Keyword arguments as external dependencies for `executor`
        """
        attr = {"executor": func, "result": None}

        if args is not None:
            attr.update({"args": args})

        name = func.__name__ if node_name is None else node_name
        self.g.add_nodes_from([(name, attr)])
        return self

    def add_edge(self, u: str, v: str,  result_index: Optional[int] = None, arg_index: Optional[int] = None):
        """
        Re-implement of networkx's `add_edge` method, with parameters follow the `EdgeAttrInterface`

        @ Parameters:

        - `u`: name of parent node
        - `v`: name of child node
        - `result_index`: the result at `result_index` index position will be used. 
        If the `u`'s executor function return a single result, this will cause `Index error` !!!
        - `arg_index`: the `u`'s result will be injected as `arg_index` index argument in `v`'s executor function
        """
        self.g.add_edge(u, v)
        if result_index is not None:
            self.g[u][v]['result_index'] = result_index
        if arg_index is not None:
            self.g[u][v]['arg_index'] = arg_index

        return self

    def add_linear_edge(self, ordered_node_names: list):
        """
        Utility function to add edge linearly

        @ Example:
        - Rather add edge as: 

            G.add_edge('func1', 'func2')

            G.add_edge('func2', 'func3')

            G.add_edge('func3', 'func4')
        - It will add edge as:
            G.add_edge_linear('func 1', 'func 2', 'func 3', 'func 4')
        """
        i = 0
        while i < len(ordered_node_names) - 1:
            self.g.add_edge(ordered_node_names[i], ordered_node_names[i+1])
            i += 1

    def compile_graph(self):
        """
        Compile all node's executor with its external dependencies to partial function
        """
        for node_name in self.g.nodes:
            node_ref = self.__node_ref(node_name)
            if "args" in node_ref:
                node_ref['partial_fn'] = PartialFunc(
                    fn=node_ref['executor'], defaults=node_ref['args'])
            else:
                node_ref['partial_fn'] = PartialFunc(fn=node_ref['executor'])

    def get_topological_sort(self):
        return [i for i in topological_sort(self.g)]

    def get_node_result(self, node_name):
        return self.__node_ref(node_name)['result']

    def reset_graph(self):
        """
        Reset all nodes by change the node's `partial_fn` and `result` to `None`.
        Graph has to be re-compiled
        """
        for i in self.get_topological_sort():
            node_ref = self.__node_ref(i)
            node_ref['partial_fn'] = None
            node_ref['result'] = None

        return self

    def execute_node(self, func: str):
        """
        ## Run the graph recursively 

        Get the result of predecessor from the edge attribute. 

        If the function beforehand has not been ran (`result` is `None`), run the predecessor node

        Run the node's partial function with previous result (prev result is align as positional argument base on edge data)
        """
        current_node_ref = self.__node_ref(func)
        if current_node_ref['result'] is not None:
            return current_node_ref['result']
        else:
            node_predecessor_labels = self.__get_predecessors(func)
            if len(node_predecessor_labels) > 0:
                if self.log:
                    print(
                        f'Found {len(node_predecessor_labels)} predecessor(s) for {func}: {", ".join(node_predecessor_labels)}')
                args_array = []
                for pred_label in node_predecessor_labels:
                    node_ref = self.__node_ref(pred_label)
                    edge_ref = self.__edge_ref(pred_label, func)

                    self.__assert_prev_result(pred_label)

                    if ('result_index' in edge_ref):
                        prev_result = node_ref['result'][edge_ref['result_index']]
                    else:
                        prev_result = node_ref['result']

                    if ('arg_index' in edge_ref):
                        arg_index = edge_ref['arg_index']
                    else:
                        arg_index = 0

                    args_array.append((arg_index, prev_result))

                sorted_args_array = self.__sort_args_array(args_array)
                fn_result = current_node_ref['partial_fn'](*sorted_args_array)

            else:
                if self.log:
                    print(f'Root node reach: {func}')

                fn_result = current_node_ref['partial_fn']()

            current_node_ref['result'] = fn_result
            return current_node_ref['result']

    def execute(self):
        """
        Run the whole graph by topological sort the nodes
        """
        self.compile_graph()
        ordered_nodes = self.get_topological_sort()
        for i in ordered_nodes:
            self.execute_node(i)


if __name__ == "__main__":
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
    print('final result:', exe.get_node_result('func10'))
