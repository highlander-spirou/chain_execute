# 0.4.3

## Fix bug 
- `execute_node` 🪲
    - Fix bug that `execute_node` not compiled, leading to `partial_fn` is `None`


# 0.4.2

## Fix bug 
- `update_node_args` 🪲
    - Fix bug that `update_node_args` not working when `args` is initially empty



# 0.4.1

## Change | Update
- `init` ✅
    - `init` method now automatically create `nx.DiGraph()` if not provided
    - `print_to_console` method now has default method as `False`

## New
- `update_node_args` ✅
    - Update the node's external dependency (args) using `spread operator`, and reset all nodes from the updated node using `reset_from_node` method


# 0.3.0

## Refactor
- `reset_graph` ✅
- `compile_graph` ✅

## Change | Update
- `get_node_result` ✅
    - `get_node_result` now return last topological sort node if `node_name` not provided
- `add_edge_from_node_order` ✅ 
    - `add_edge_from_node_order` now receive two extra arguments: `exclude` and `reset_after_add`

## New
- `reset_from_node` ✅
    - now the graph can be re-compile to run from `reset_root`, not entire graph