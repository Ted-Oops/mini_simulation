import contextlib
import io
import http.client
import json
import math
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import main
from config import build_experiment_config
from llm_client import LLMClient, LLMConfig, LLMResponseError, load_llm_config
from market import OrderBookMarket
from models import DecisionMode, StrategyType
from visualization import _ensure_output_dir, save_experiment_report


def config_for(strategy=StrategyType.FUNDAMENTAL, mode=DecisionMode.RULE_BASED, **kwargs):
    return replace(build_experiment_config(strategy, mode), **kwargs)


def api_response(content):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


HOLD = json.dumps({"action": "hold", "quantity": 0, "limit_price": None, "reason": "test"})


class ConfigurationTests(unittest.TestCase):
    def test_invalid_experiment_values_fail_before_simulation(self):
        invalid = {
            "num_agents": [0, -1, 1.5, True], "num_steps": [0, -1],
            "bootstrap_steps": [-1, 0.5], "batch_size": [0, -1],
            "initial_shares_per_agent": [-1], "max_order_age_steps": [-1],
            "initial_price": [0, -1, float("nan"), float("inf")],
            "initial_cash": [-1, float("nan")], "price_tick": [0, -1, 0.001],
            "price_impact": [-1, float("inf")], "max_price_offset_ratio": [-1, 1],
            "seed": [True, 1.5], "strategy_type": ["unknown"],
            "decision_mode": ["unknown"],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        config_for(**{field: value})
                    config = config_for()
                    setattr(config, field, value)
                    with self.assertRaisesRegex(ValueError, field):
                        OrderBookMarket(config)

    def test_cli_rejects_invalid_counts_without_traceback(self):
        for option, value in (("--agents", "0"), ("--steps", "-1"),
                              ("--bootstrap-steps", "-1"), ("--agents", "nan")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    main.parse_args([option, value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(option, err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())
        self.assertEqual(main.parse_args(["--bootstrap-steps", "0"]).bootstrap_steps, 0)

    def test_invalid_llm_configuration(self):
        invalid = {"timeout_seconds": [0, -1, float("nan"), float("inf")],
                   "temperature": [-1, 3, float("nan")], "max_tokens": [0, -1, 1.5, True],
                   "base_url": ["invalid", "http://", "http://host:bad", "http://host/?q=1"],
                   "model": ["", " "], "api_key": [" "]}
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        LLMConfig(**({"api_key": None} | {field: value}))

    def test_llm_zero_temperature_is_preserved_and_bad_env_is_identified(self):
        with patch.dict(os.environ, {}, clear=True), patch("llm_client._load_local_api_config", return_value={"temperature": 0}):
            self.assertEqual(load_llm_config().temperature, 0)
            for env_name in ("MINI_SIM_LLM_TIMEOUT", "MINI_SIM_LLM_MAX_TOKENS"):
                with patch.dict(os.environ, {env_name: "bad-value"}):
                    with self.assertRaisesRegex(ValueError, env_name):
                        load_llm_config()


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(LLMConfig(api_key="test-key", base_url="http://example.test"))

    def test_malformed_responses_fall_back_without_disabling_next_request(self):
        for content in (None, [], {}, "not JSON", '{"action":'):
            with self.subTest(content=content):
                with patch("urllib.request.urlopen", return_value=api_response(content)):
                    self.assertIsNone(self.client.complete_json([]))
                self.assertEqual(self.client.last_error_code, "response_error")
                self.assertIsNone(self.client.disabled_reason)
                with patch("urllib.request.urlopen", return_value=api_response(HOLD)):
                    self.assertEqual(self.client.complete_json([])["action"], "hold")
                self.assertIsNone(self.client.last_error)
                self.assertIsNone(self.client.last_error_code)

    def test_bad_envelope_or_encoding_is_a_response_error(self):
        for data in (b"null", b"{}", b'{"choices":[]}', b"\xff"):
            with self.subTest(data=data), patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
                self.assertIsNone(self.client.complete_json([]))
                self.assertEqual(self.client.last_error_code, "response_error")

    def test_transport_failure_disables_further_requests(self):
        for error in (TimeoutError("timeout"), URLError("offline"), http.client.IncompleteRead(b"partial")):
            client = LLMClient(self.client.config)
            with self.subTest(error=error), patch("urllib.request.urlopen", side_effect=error) as request:
                self.assertIsNone(client.complete_json([]))
                self.assertEqual(client.last_error_code, "transport_error")
                self.assertIsNone(client.complete_json([]))
                self.assertEqual(client.last_error_code, "client_disabled")
                self.assertEqual(request.call_count, 1)

    def test_404_tries_alternate_endpoint(self):
        error = HTTPError("http://example.test/v1/chat/completions", 404, "missing", {}, io.BytesIO())
        with patch("urllib.request.urlopen", side_effect=[error, api_response(HOLD)]) as request:
            self.assertEqual(self.client.complete_json([])["action"], "hold")
            self.assertEqual(request.call_count, 2)
            self.assertTrue(request.call_args.args[0].full_url.endswith("/chat/completions"))
            self.assertNotIn("/v1/", request.call_args.args[0].full_url)

    def test_parallel_errors_remain_attached_to_their_request(self):
        barrier = threading.Barrier(2)

        def complete(messages):
            raise LLMResponseError(messages[0]["content"])

        def worker(label):
            self.client.complete_json([{"content": label}])
            barrier.wait(timeout=5)
            return self.client.last_error

        with patch.object(self.client, "_complete", side_effect=complete), ThreadPoolExecutor(2) as pool:
            self.assertEqual(list(pool.map(worker, ["first", "second"])), ["first", "second"])


class MarketStabilityTests(unittest.TestCase):
    def test_same_second_reports_do_not_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as folder, patch("visualization.datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261016_120000"
            first = _ensure_output_dir(folder, "experiment")
            (first / "summary.json").write_text("original", encoding="utf-8")
            second = _ensure_output_dir(folder, "experiment")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "summary.json").read_text(encoding="utf-8"), "original")

    def assert_summary(self, summary, config):
        self.assertEqual(summary["num_steps"], config.num_steps)
        self.assertEqual(len(summary["price_history"]), config.num_steps + 1)
        self.assertEqual(len(summary["volume_history"]), config.num_steps)
        self.assertEqual(len(summary["bootstrap_price_history"]), config.bootstrap_steps + 1)
        self.assertTrue(all(math.isfinite(p) and p > 0 for p in summary["price_history"]))
        self.assertTrue(all(a["cash"] >= -1e-6 and a["shares"] >= 0 for a in summary["agents"]))
        for buy, sell, hold in zip(summary["buy_count_history"], summary["sell_count_history"], summary["hold_count_history"]):
            self.assertEqual(buy + sell + hold, config.num_agents)
        json.dumps(summary, allow_nan=False)

    def test_nine_combinations_with_default_and_boundary_configurations(self):
        cases = [{}, {"num_agents": 1, "num_steps": 3, "bootstrap_steps": 0},
                 {"num_agents": 5, "num_steps": 7, "bootstrap_steps": 1},
                 {"num_agents": 7, "num_steps": 8, "bootstrap_steps": 3},
                 {"num_agents": 20, "num_steps": 100, "bootstrap_steps": 24},
                 {"num_agents": 1, "num_steps": 1, "bootstrap_steps": 0,
                  "initial_cash": 0, "initial_shares_per_agent": 0,
                  "max_order_age_steps": 0, "batch_size": 5}]
        client = LLMClient(LLMConfig(api_key=None))
        with patch("agents.base.get_default_llm_client", return_value=client):
            for strategy in StrategyType:
                for mode in DecisionMode:
                    for overrides in cases:
                        with self.subTest(strategy=strategy, mode=mode, overrides=overrides):
                            config = config_for(strategy, mode, **overrides)
                            market = OrderBookMarket(config)
                            market.run_simulation()
                            summary = market.get_summary()
                            self.assert_summary(summary, config)
                            count = 0 if mode == DecisionMode.RULE_BASED else config.num_steps * config.num_agents
                            self.assertEqual(summary["llm_stats"]["fallback_count"], count)
                            self.assertEqual(summary["llm_stats"]["success_count"], 0)
                            self.assertEqual(sum(summary["llm_stats"]["fallback_reasons"].values()), count)

    def test_successes_and_invalid_values_are_counted_in_all_llm_modes(self):
        payloads = [({"action": "hold", "quantity": 0}, None),
                    ({"action": "buy", "quantity": "inf"}, "invalid_quantity"),
                    ({"action": "buy", "quantity": 1, "limit_price": float("nan")}, "invalid_limit_price"),
                    ({"action": "hold", "signal_strength": "NaN"}, "invalid_signal_strength"),
                    ({"action": "hold", "quantity": True}, "invalid_quantity"),
                    ({"action": "unknown"}, "invalid_action")]
        for strategy in StrategyType:
            for mode in (DecisionMode.HALF_RULE_BASED, DecisionMode.OPEN_ENDED):
                for payload, reason in payloads:
                    with self.subTest(strategy=strategy, mode=mode, payload=payload):
                        client = LLMClient(LLMConfig(api_key="test-key"))
                        with patch("agents.base.get_default_llm_client", return_value=client), patch.object(client, "_complete", return_value=json.dumps(payload)):
                            config = config_for(strategy, mode, num_steps=2, num_agents=4)
                            market = OrderBookMarket(config)
                            market.run_simulation()
                        summary = market.get_summary()
                        self.assert_summary(summary, config)
                        stats = summary["llm_stats"]
                        self.assertEqual(stats["decision_count"], 8)
                        self.assertEqual(stats["success_count"], 8 if reason is None else 0)
                        self.assertEqual(stats["fallback_reasons"], {} if reason is None else {reason: 8})
                        self.assertEqual(sum(summary["llm_success_count_history"]), stats["success_count"])

    def test_mixed_results_include_holds_and_executed_orders(self):
        client = LLMClient(LLMConfig(api_key="test-key"))

        def complete(messages):
            context = json.loads(messages[-1]["content"].split("Market context JSON:\n")[1])
            agent_id = context["agent_state"]["agent_id"]
            if agent_id.endswith("_1"):
                return HOLD
            if agent_id.endswith("_2"):
                return '{"action":"buy","quantity":1,"signal_strength":1.0}'
            raise LLMResponseError("invalid response")

        with patch("agents.base.get_default_llm_client", return_value=client), patch.object(client, "_complete", side_effect=complete):
            config = config_for(mode=DecisionMode.OPEN_ENDED, num_steps=3, num_agents=3)
            market = OrderBookMarket(config)
            market.run_simulation()
        summary = market.get_summary()
        self.assert_summary(summary, config)
        self.assertEqual(summary["llm_stats"]["success_count"], 6)
        self.assertEqual(summary["llm_stats"]["fallback_reasons"], {"response_error": 3})
        self.assertTrue(any(t["buyer_id"] == "fundamental_2" for t in summary["trade_history"]))
        with tempfile.TemporaryDirectory() as folder:
            output = save_experiment_report(summary, folder)
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["llm_stats"], summary["llm_stats"])
            self.assertGreater((output / "report.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
