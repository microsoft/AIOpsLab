# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from clients.utils.llm import LiteLLMClient


def _fake_litellm(content="hello", raises=None):
    """Return a stub litellm module and the dict capturing its request."""
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        if raises is not None:
            raise raises
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    module = types.ModuleType("litellm")
    module.completion = completion
    return module, captured


class TestLiteLLMClientConfig(unittest.TestCase):
    """Model, endpoint and key resolution."""

    def test_defaults_to_gpt_4o_with_no_endpoint(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        self.assertEqual(client.model, "gpt-4o")
        self.assertEqual(client.api_base, "")
        self.assertEqual(client.api_key, "")

    def test_reads_the_environment(self):
        env = {
            "LITELLM_MODEL": "anthropic/claude-opus-4-7",
            "LITELLM_API_BASE": "http://localhost:4000",
            "LITELLM_API_KEY": "sk-gateway",
        }
        with patch.dict("os.environ", env, clear=True):
            client = LiteLLMClient(use_cache=False)
        self.assertEqual(client.model, "anthropic/claude-opus-4-7")
        self.assertEqual(client.api_base, "http://localhost:4000")
        self.assertEqual(client.api_key, "sk-gateway")

    def test_constructor_arguments_win(self):
        env = {"LITELLM_MODEL": "from-env", "LITELLM_API_BASE": "http://env:4000"}
        with patch.dict("os.environ", env, clear=True):
            client = LiteLLMClient(
                model="explicit", api_base="http://explicit:4000", use_cache=False
            )
        self.assertEqual(client.model, "explicit")
        self.assertEqual(client.api_base, "http://explicit:4000")

    def test_trailing_slash_is_trimmed(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(api_base="http://localhost:4000/", use_cache=False)
        self.assertEqual(client.api_base, "http://localhost:4000")

    def test_no_endpoint_is_required(self):
        """Unlike GenericOpenAIClient, constructing without a base_url is valid.

        That is the point of this client: reaching a provider needs no
        OpenAI-compatible endpoint to exist first.
        """
        with patch.dict("os.environ", {}, clear=True):
            LiteLLMClient(use_cache=False)  # must not raise


class TestLiteLLMClientRouting(unittest.TestCase):
    """Whether a configured gateway is actually used."""

    def test_direct_mode_sends_the_bare_model_name(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(model="anthropic/claude-opus-4-7", use_cache=False)
        self.assertEqual(client.route(), "anthropic/claude-opus-4-7")

    def test_gateway_mode_prefixes_the_model(self):
        """Regression guard for a silently bypassed gateway.

        Without the litellm_proxy/ prefix LiteLLM infers the provider from the
        model name and calls it directly, so api_base is ignored and the call
        fails on missing provider credentials.
        """
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(
                model="gemini-2.5-flash",
                api_base="http://localhost:4000",
                use_cache=False,
            )
        self.assertEqual(client.route(), "litellm_proxy/gemini-2.5-flash")

    def test_an_explicit_prefix_is_not_doubled(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(
                model="litellm_proxy/gpt-4o",
                api_base="http://localhost:4000",
                use_cache=False,
            )
        self.assertEqual(client.route(), "litellm_proxy/gpt-4o")


class TestLiteLLMClientInference(unittest.TestCase):
    """The request LiteLLM actually receives."""

    def _run(self, client, payload=None):
        module, captured = _fake_litellm()
        with patch.dict(sys.modules, {"litellm": module}):
            result = client.inference(payload or [{"role": "user", "content": "hi"}])
        return result, captured

    def test_returns_the_message_content(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        result, _ = self._run(client)
        self.assertEqual(result, ["hello"])

    def test_forwards_the_payload_unchanged(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        payload = [
            {"role": "system", "content": "you are an SRE"},
            {"role": "user", "content": "the service is down"},
        ]
        _, captured = self._run(client, payload)
        self.assertEqual(captured["messages"], payload)

    def test_drop_params_is_on(self):
        """Providers reject each other's parameters; without this the shared
        temperature/top_p/n settings cannot serve every model."""
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        _, captured = self._run(client)
        self.assertTrue(captured["drop_params"])

    def test_direct_mode_sends_no_endpoint_or_key(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(model="gpt-4o", use_cache=False)
        _, captured = self._run(client)
        self.assertNotIn("api_base", captured)
        self.assertNotIn("api_key", captured)

    def test_gateway_mode_forwards_endpoint_and_key(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(
                model="gpt-4o",
                api_base="http://localhost:4000",
                api_key="sk-gateway",
                use_cache=False,
            )
        _, captured = self._run(client)
        self.assertEqual(captured["api_base"], "http://localhost:4000")
        self.assertEqual(captured["api_key"], "sk-gateway")
        self.assertEqual(captured["model"], "litellm_proxy/gpt-4o")

    def test_max_tokens_is_configurable(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(max_tokens=256, use_cache=False)
        _, captured = self._run(client)
        self.assertEqual(captured["max_tokens"], 256)

    def test_provider_errors_propagate(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        module, _ = _fake_litellm(raises=RuntimeError("provider exploded"))
        with patch.dict(sys.modules, {"litellm": module}):
            with self.assertRaises(RuntimeError):
                client.inference([{"role": "user", "content": "hi"}])

    def test_missing_package_explains_how_to_install(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        with patch.dict(sys.modules, {"litellm": None}):
            with self.assertRaises(ImportError) as ctx:
                client.inference([{"role": "user", "content": "hi"}])
        self.assertIn("poetry install", str(ctx.exception))


class TestLiteLLMClientCaching(unittest.TestCase):
    """Caching matches the other clients' behaviour."""

    def test_run_stores_the_response(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        client.cache = MagicMock()
        client.cache.get_from_cache.return_value = None

        module, _ = _fake_litellm()
        with patch.dict(sys.modules, {"litellm": module}):
            result = client.run([{"role": "user", "content": "hi"}])

        self.assertEqual(result, ["hello"])
        client.cache.add_to_cache.assert_called_once()
        client.cache.save_cache.assert_called_once()

    def test_a_cache_hit_skips_the_provider(self):
        with patch.dict("os.environ", {}, clear=True):
            client = LiteLLMClient(use_cache=False)
        client.cache = MagicMock()
        client.cache.get_from_cache.return_value = ["cached"]

        # No litellm module installed: a cache hit must not need one.
        with patch.dict(sys.modules, {"litellm": None}):
            result = client.inference([{"role": "user", "content": "hi"}])

        self.assertEqual(result, ["cached"])


class TestLiteLLMAgentRegistration(unittest.TestCase):
    def test_agent_is_registered(self):
        from clients.registry import AgentRegistry
        from clients.litellm_agent import LiteLLMAgent

        self.assertIs(AgentRegistry().AGENT_REGISTRY["litellm"], LiteLLMAgent)


if __name__ == "__main__":
    unittest.main()
