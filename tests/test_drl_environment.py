"""The abstract game contract and public concrete-environment exports."""

import inspect

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import LiteEFG as leg
from LiteEFG.drl import Environment, env
from LiteEFG.drl.environment import Environment as AbstractEnvironment
from LiteEFG.drl.env.goofspiel import Goofspiel
from LiteEFG.drl.env.dark_chess import DarkChess


def test_unfinished_environment_cannot_be_instantiated():
    class UnfinishedEnvironment(Environment):
        num_players = 2
        num_actions = 3
        max_steps = 3

    with pytest.raises(TypeError, match="abstract"):
        Environment()
    with pytest.raises(TypeError, match="abstract.*step"):
        UnfinishedEnvironment()


def test_concrete_games_share_the_abstract_contract_without_shadowing_native():
    assert Environment is AbstractEnvironment
    assert leg.Environment is not Environment
    assert leg.Goofspiel is env.Goofspiel is Goofspiel
    assert leg.DarkChess is env.DarkChess is DarkChess
    assert isinstance(leg.Goofspiel(), Environment)
    assert isinstance(leg.DarkChess(), Environment)
    for name in env.__all__:
        game = getattr(env, name)
        assert issubclass(game, Environment)
        assert not inspect.isabstract(game)
