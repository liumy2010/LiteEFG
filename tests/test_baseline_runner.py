"""Check runner reporting at actual completed-update boundaries."""

import pytest

from LiteEFG.baselines import utils


@pytest.mark.parametrize("iterations, frequency, reports", [
    (5, 2, [2, 4, 5]), (2, 10, [2]), (4, 2, [2, 4]),
])
def test_progress_and_final_report(monkeypatch, iterations, frequency, reports):
    class Graph:
        updates = 0

        def current_strategy(self):
            return "strategy"

        def update_graph(self, env):
            self.updates += 1

    graph = Graph()
    measured_at = []

    class Environment:
        def set_graph(self, graph):
            pass

        def update_strategy(self, *args, **kwargs):
            pass

        def exploitability(self, *args):
            measured_at.append(graph.updates)
            return [0.0, 0.0]

    class Progress:
        completed = 0
        closed = False

        def __init__(self, total):
            assert total == iterations

        def __enter__(self):
            return self

        def __exit__(self, *args):
            Progress.closed = True

        def set_description(self, description):
            pass

        def update(self, amount):
            Progress.completed += amount
            assert Progress.completed == graph.updates

    monkeypatch.setattr(utils, "tqdm", Progress)
    monkeypatch.setattr(utils.pyspiel, "load_game", lambda name: None)
    monkeypatch.setattr(utils.leg, "OpenSpielEnv", lambda *args, **kwargs: Environment())
    utils.train(graph, "Enumerate", "default", iterations, frequency)
    assert graph.updates == iterations
    assert measured_at == reports
    assert Progress.completed == iterations
    assert Progress.closed


@pytest.mark.parametrize("iterations, frequency", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_invalid_iteration_parameters_are_rejected(iterations, frequency):
    with pytest.raises(ValueError, match="must be positive"):
        utils.train(None, "Enumerate", "default", iterations, frequency)
