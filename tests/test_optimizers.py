"""Tests for optimizer strategies."""
import pytest

from src.optimizers import (
    BaseOptimizer,
    OptimizerConfig,
    RandomSearchOptimizer,
    create_optimizer,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_seed(self):
        config = OptimizerConfig()
        assert config.seed == 0

    def test_custom_seed(self):
        config = OptimizerConfig(seed=42)
        assert config.seed == 42


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a standard random search optimizer for testing."""
        return RandomSearchOptimizer(
            param_bounds={"C": (0.01, 100.0), "max_iter": (10, 1000)},
            integer_keys={"max_iter"},
            is_higher_better=True,
            config=OptimizerConfig(seed=42),
        )

    def test_name(self, optimizer):
        assert optimizer.name == "random"

    def test_propose_returns_valid_structure(self, optimizer):
        proposal, source, token_usage = optimizer.propose(
            current_config={"C": 1.0, "max_iter": 100},
            last_score=0.5,
            history=[],
        )

        assert isinstance(proposal, dict)
        assert source == "random"
        assert token_usage is None

    def test_propose_respects_bounds(self, optimizer):
        """Proposed values should be within bounds."""
        for _ in range(100):
            proposal, _, _ = optimizer.propose({}, 0.0, [])
            assert 0.01 <= proposal["C"] <= 100.0
            assert 10 <= proposal["max_iter"] <= 1000

    def test_propose_integer_keys_are_integers(self, optimizer):
        """Integer keys should produce integer values."""
        for _ in range(100):
            proposal, _, _ = optimizer.propose({}, 0.0, [])
            assert isinstance(proposal["max_iter"], int)

    def test_propose_deterministic_with_same_seed(self):
        """Same seed should produce same sequence."""
        config = OptimizerConfig(seed=42)
        opt1 = RandomSearchOptimizer(
            param_bounds={"C": (0.01, 100.0)},
            integer_keys=set(),
            is_higher_better=True,
            config=config,
        )
        opt2 = RandomSearchOptimizer(
            param_bounds={"C": (0.01, 100.0)},
            integer_keys=set(),
            is_higher_better=True,
            config=config,
        )

        proposals1 = [opt1.propose({}, 0.0, [])[0] for _ in range(5)]
        proposals2 = [opt2.propose({}, 0.0, [])[0] for _ in range(5)]

        assert proposals1 == proposals2

    def test_propose_different_with_different_seed(self):
        """Different seeds should produce different sequences."""
        opt1 = RandomSearchOptimizer(
            param_bounds={"C": (0.01, 100.0)},
            integer_keys=set(),
            is_higher_better=True,
            config=OptimizerConfig(seed=42),
        )
        opt2 = RandomSearchOptimizer(
            param_bounds={"C": (0.01, 100.0)},
            integer_keys=set(),
            is_higher_better=True,
            config=OptimizerConfig(seed=123),
        )

        proposal1, _, _ = opt1.propose({}, 0.0, [])
        proposal2, _, _ = opt2.propose({}, 0.0, [])

        assert proposal1 != proposal2

    def test_reset_restarts_sequence(self, optimizer):
        """Reset should restart the random sequence."""
        proposals_before = [optimizer.propose({}, 0.0, [])[0] for _ in range(3)]
        optimizer.reset()
        proposals_after = [optimizer.propose({}, 0.0, [])[0] for _ in range(3)]

        assert proposals_before == proposals_after

    def test_propose_ignores_history(self, optimizer):
        """Random search should ignore history."""
        optimizer.reset()
        proposal_no_history, _, _ = optimizer.propose(
            current_config={"C": 1.0, "max_iter": 100},
            last_score=0.5,
            history=[],
        )

        optimizer.reset()
        proposal_with_history, _, _ = optimizer.propose(
            current_config={"C": 1.0, "max_iter": 100},
            last_score=0.9,
            history=[
                {"config": {"C": 10.0, "max_iter": 500}, "score": 0.8},
                {"config": {"C": 50.0, "max_iter": 200}, "score": 0.7},
            ],
        )

        # Same seed, same sequence - history doesn't affect output
        assert proposal_no_history == proposal_with_history


class TestCreateOptimizer:
    """Tests for the optimizer factory function."""

    def test_create_random_optimizer(self):
        optimizer = create_optimizer(
            optimizer_type="random",
            param_bounds={"C": (0.01, 100.0)},
            integer_keys=set(),
            is_higher_better=True,
            config=OptimizerConfig(seed=0),
        )

        assert isinstance(optimizer, RandomSearchOptimizer)
        assert optimizer.name == "random"

    def test_create_unknown_optimizer_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(
                optimizer_type="unknown",
                param_bounds={"C": (0.01, 100.0)},
                integer_keys=set(),
                is_higher_better=True,
                config=OptimizerConfig(seed=0),
            )

    def test_create_llm_optimizer_requires_benchmark(self):
        with pytest.raises(ValueError, match="benchmark is required"):
            create_optimizer(
                optimizer_type="llm",
                param_bounds={"C": (0.01, 100.0)},
                integer_keys=set(),
                is_higher_better=True,
                config=OptimizerConfig(seed=0),
                benchmark=None,
            )
