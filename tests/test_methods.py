from ChainExecutor.chain_executor import ChainExecutor


def test_update_node_args():
    def func1(arg1):
        print(
            f'Func 1 running with `{arg1}`')
        return f'Func 1 result is {arg1}'

    def func2(peer_dep):
        return f"Func 2 running with `{peer_dep}` as peer dependency"

    exe = ChainExecutor()

    exe.add_node(func1) \
        .add_node(func2)

    exe.add_edge_from_node_order()

    exe.update_node_args({'node_name': func1.__name__,
                         'args': {'arg1': 'hello world'}})

    exe.execute()

    result = exe.get_node_result('func2')
    assert result == """Func 2 running with `Func 1 result is hello world` as peer dependency"""


def test_update_multiple_nodes():
    def func1(arg_1):
        return f"hello {arg_1} from func 1"

    def func2(arg_2):
        return f"hello {arg_2} from func 2"

    def func3(arg_1, arg_2):
        return f"func 3 is returning {arg_1} and {arg_2}"

    g = ChainExecutor()
    g.add_node(func1, 'node_1')
    g.add_node(func2, 'node_2')
    g.add_node(func3, 'node_3')

    g.add_edge('node_1', 'node_3')
    g.add_edge('node_2', 'node_3')

    g.update_node_args([{'node_name': 'node_1', 'args': {'arg_1': 'Mập'}},
                        {'node_name': 'node_2', 'args': {'arg_2': 'Rex'}}])

    g.execute()

    result_1 = g.get_node_result()


    g.update_node_args({'node_name': 'node_1', 'args': {'arg_1': 'Shipapa'}})
    g.execute()

    result_2 = g.get_node_result()

    assert result_1 == "func 3 is returning hello Mập from func 1 and hello Rex from func 2"
    assert result_2 == 'func 3 is returning hello Shipapa from func 1 and hello Rex from func 2'


def test_execute_node():
    def func1(arg1):
        print(
            f'Func 1 running with `{arg1}`')
        return f'Func 1 result is {arg1}'

    def func2(peer_dep):
        return f"Func 2 running with `{peer_dep}` as peer dependency"

    exe = ChainExecutor()

    exe.add_node(func1) \
        .add_node(func2)

    exe.add_edge_from_node_order()

    exe.update_node_args({'node_name': func1.__name__,
                         'args': {'arg1': 'hello world'}})

    exe.execute_node('func1')

    result = exe.get_node_result('func1')
    print(result)


if __name__ == '__main__':
    test_update_node_args()
    test_execute_node()
    test_update_multiple_nodes()
