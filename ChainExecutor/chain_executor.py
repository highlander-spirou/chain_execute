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
    """
    Execute chain of functions with Direct Acyclic Graph manners.

    @ Initialize parameters

    - g (`nx.DiGraph`)
    - print_to_console (`bool`): Enable terminal log 
    """

    def __init__(self, graph: nx.DiGraph, print_to_console: bool) -> None:
        self.g = graph
        self.log = print_to_console
        self.__node_add_order: List[str] = []

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

    def __reset_node(self, node_name: str):
        """
        Reset the node's partial function and result to `None`
        """
        node_ref = self.__node_ref(node_name)
        node_ref['partial_fn'] = None
        node_ref['result'] = None

    def __compile_node(self, node_name: str):
        node_ref = self.__node_ref(node_name)
        if "args" in node_ref:
            node_ref['partial_fn'] = PartialFunc(
                fn=node_ref['executor'], defaults=node_ref['args'])
        else:
            node_ref['partial_fn'] = PartialFunc(fn=node_ref['executor'])

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
        self.__node_add_order.append(name)
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

    def add_edge_from_node_order(self, exclude: Optional[Union[str, list]] = None, reset_after_add=False):
        """
        Add linear edge relationships base on the order of node adding to the object.

        @ Parameters:

        - exclude: Exclude the node

        - reset_after_add: clear the node history, so that this function can be ran in future. 
        Usually need when constructing complex graph having multi-stage 
        """
        if exclude is not None:
            if type(exclude) == str:
                node_order = [i for i in self.__node_add_order if i != exclude]
            else:
                node_order = [
                    i for i in self.__node_add_order if i not in exclude]
        else:
            node_order = self.__node_add_order

        self.add_linear_edge(node_order)

        if reset_after_add:
            self.__node_add_order = []

    def compile_graph(self):
        """
        Compile all node's executor with its external dependencies to partial function
        """
        for node_name in self.g.nodes:
            self.__compile_node(node_name)

    def get_topological_sort(self):
        return [i for i in topological_sort(self.g)]

    def get_node_result(self, node_name: Optional[str] = None):
        """
        Get node result by `node_name`.

        If `node_name` not provided, return the result of the last node from topological_sort
        """
        if node_name is None:
            last_node = self.get_topological_sort()[-1]
            return self.__node_ref(last_node)['result']
        return self.__node_ref(node_name)['result']

    def reset_graph(self):
        """
        Reset all nodes by change the node's `partial_fn` and `result` to `None`.
        Graph has to be re-compiled
        """
        for i in self.g.nodes:
            self.__reset_node(i)

    def reset_from_node(self, root_node_name: str, reset_root_node=True):
        """
        Reset all the nodes after the `root_node_name` from the topological sort.

        This method is used when there are a dependencies change, or there are an update in the `root_node_name` result

        @ Parameters
        - reset_root_node: If False, reset all the nodes after the `root_node_name`, excluding it. If True, reset the `root_node_name`.  
        """
        node_order = self.get_topological_sort()
        root_node_index = node_order.index(root_node_name)
        if reset_root_node:
            nodes_to_reset = node_order[root_node_index:]
        else:
            nodes_to_reset = node_order[root_node_index+1:]

        for i in nodes_to_reset:
            self.__reset_node(i)

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
