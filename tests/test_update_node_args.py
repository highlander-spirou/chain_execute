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

    exe.update_node_args(func1.__name__, args={'arg1': 'hello world'})

    exe.execute()

    result = exe.get_node_result('func2')
    assert result == """Func 2 running with `Func 1 result is hello world` as peer dependency"""


if __name__ == '__main__':
    test_update_node_args()