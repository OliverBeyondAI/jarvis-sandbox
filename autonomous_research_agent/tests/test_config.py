"""Tests for Config — validation, defaults, and env overrides."""

from __future__ import annotations

import os
from unittest.mock import patch

from autonomous_research_agent.config import Config


class TestConfigDefaults:
    """Verify default values and immutability."""

    def test_default_model(self):
        cfg = Config()
        assert cfg.model == "claude-opus-4-7-20250501"

    def test_default_max_tokens(self):
        cfg = Config()
        assert cfg.max_tokens == 16384

    def test_default_max_agent_turns(self):
        cfg = Config()
        assert cfg.max_agent_turns == 40

    def test_default_min_searches(self):
        cfg = Config()
        assert cfg.min_searches == 4

    def test_default_min_deep_dives(self):
        cfg = Config()
        assert cfg.min_deep_dives == 2

    def test_frozen_immutability(self):
        cfg = Config()
        try:
            cfg.model = "other-model"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestConfigValidation:
    """Verify validation warnings."""

    def test_warns_missing_tavily_key(self):
        cfg = Config(tavily_api_key="", anthropic_api_key="sk-test")
        warnings = cfg.validate()
        assert any("TAVILY_API_KEY" in w for w in warnings)

    def test_warns_missing_anthropic_key(self):
        cfg = Config(tavily_api_key="tvly-test", anthropic_api_key="")
        warnings = cfg.validate()
        assert any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_no_warnings_when_keys_set(self):
        cfg = Config(tavily_api_key="tvly-test", anthropic_api_key="sk-test")
        warnings = cfg.validate()
        assert warnings == []

    def test_both_warnings_when_no_keys(self):
        cfg = Config(tavily_api_key="", anthropic_api_key="")
        warnings = cfg.validate()
        assert len(warnings) == 2


class TestConfigFromEnv:
    """Verify environment variable loading."""

    @patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-env", "ANTHROPIC_API_KEY": "sk-env"})
    def test_from_env_picks_up_keys(self):
        cfg = Config.from_env()
        assert cfg.tavily_api_key == "tvly-env"
        assert cfg.anthropic_api_key == "sk-env"

    @patch.dict(os.environ, {"RESEARCH_OUTPUT_DIR": "/tmp/custom_reports"})
    def test_from_env_custom_output_dir(self):
        cfg = Config.from_env()
        assert cfg.output_dir == "/tmp/custom_reports"

    def test_custom_overrides(self):
        cfg = Config(
            model="claude-opus-4-7-20250501",
            max_agent_turns=10,
            min_searches=6,
        )
        assert cfg.model == "claude-opus-4-7-20250501"
        assert cfg.max_agent_turns == 10
        assert cfg.min_searches == 6
