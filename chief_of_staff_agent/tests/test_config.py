"""Tests for configuration."""

from chief_of_staff_agent.config import Config


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.model == "claude-opus-4-20250514"
        assert config.max_tokens == 16384
        assert config.max_agent_turns == 30

    def test_validate_missing_keys(self):
        config = Config(tavily_api_key="", anthropic_api_key="")
        warnings = config.validate()
        assert len(warnings) == 2
        assert any("TAVILY_API_KEY" in w for w in warnings)
        assert any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_validate_all_set(self):
        config = Config(tavily_api_key="test", anthropic_api_key="test")
        warnings = config.validate()
        assert len(warnings) == 0

    def test_from_env(self):
        config = Config.from_env()
        assert isinstance(config, Config)

    def test_frozen(self):
        config = Config()
        try:
            config.model = "other"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
