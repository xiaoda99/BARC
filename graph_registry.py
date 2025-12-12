# Graph registry - excluded from autoreload to persist state
# Usage: %aimport -graph_registry  (in notebook, before other imports)

_current_graph = None

def set_current_graph(graph):
    global _current_graph
    _current_graph = graph

def get_current_graph():
    return _current_graph

