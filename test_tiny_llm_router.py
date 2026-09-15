"""Tests for tiny-llm-router — zero external deps."""
import unittest
import time
from tiny_llm_router import (
    Task, RoleRule, Router, RoutingError, infer_role,
)


class TestInferRole(unittest.TestCase):
    def test_vision_when_images(self):
        self.assertEqual(infer_role(Task(has_images=True)), "vision")

    def test_code_when_code_hints(self):
        t = Task(prompt="def hello(): return 1")
        self.assertEqual(infer_role(t), "code")

    def test_long_when_huge_prompt(self):
        t = Task(prompt="x" * 5000)
        self.assertEqual(infer_role(t), "long")

    def test_json_when_needs_json_flag(self):
        self.assertEqual(infer_role(Task(needs_json=True)), "json")

    def test_default_when_empty(self):
        self.assertEqual(infer_role(Task(prompt="")), "default")


class TestPick(unittest.TestCase):
    def test_explicit_role_used(self):
        r = Router([RoleRule("code", "gpt-4o-mini"), RoleRule("default", "haiku")])
        self.assertEqual(r.pick(Task(role="code")), "gpt-4o-mini")

    def test_fallback_when_no_role_match(self):
        r = Router([RoleRule("code", "gpt-4o-mini")], fallback="haiku")
        self.assertEqual(r.pick(Task(role="vision")), "haiku")

    def test_only_if_excludes_rule(self):
        r = Router(
            [RoleRule("code", "expensive-model", only_if=lambda t: not t.has_images)],
            fallback="cheap-model",
        )
        # only_if is False -> skip -> fallback used
        self.assertEqual(r.pick(Task(role="code", has_images=True)), "cheap-model")

    def test_cost_cap_excludes_model(self):
        r = Router(
            [RoleRule("code", "expensive", max_cost_per_1k=0.001)],
            fallback="cheap",
            pricing={"expensive": {"input_per_1k": 0.01, "output_per_1k": 0.02}},
        )
        self.assertEqual(r.pick(Task(role="code")), "cheap")

    def test_circuit_break_when_many_failures(self):
        r = Router([RoleRule("code", "fragile")], fallback="safe", failure_threshold=2, failure_window_s=60.0)
        # simulate two errors
        r.record("fragile", 100, 0.0, errored=True)
        r.record("fragile", 100, 0.0, errored=True)
        # now fragile is broken -> use safe
        self.assertEqual(r.pick(Task(role="code")), "safe")

    def test_circuit_break_window_expiry(self):
        r = Router([RoleRule("code", "fragile")], fallback="safe",
                   failure_threshold=2, failure_window_s=0.05)
        r.record("fragile", 100, 0.0, errored=True)
        r.record("fragile", 100, 0.0, errored=True)
        time.sleep(0.06)
        # window expired -> fragile is healthy again
        self.assertEqual(r.pick(Task(role="code")), "fragile")

    def test_no_candidates_raises(self):
        r = Router([RoleRule("code", "fragile")], failure_threshold=1)
        r.record("fragile", 100, 0.0, errored=True)
        with self.assertRaises(RoutingError):
            r.pick(Task(role="code"))


class TestRecordAndStats(unittest.TestCase):
    def test_stats_updated_after_record(self):
        r = Router([RoleRule("default", "m")])
        r.record("m", 50, 0.001)
        r.record("m", 150, 0.003)
        s = r.stats("m")
        self.assertEqual(s["n_calls"], 2)
        self.assertEqual(s["n_errors"], 0)
        self.assertGreater(s["avg_latency_ms"], 0)
        self.assertGreater(s["avg_cost_usd"], 0)
        self.assertGreater(s["p95_latency_ms"], 0)

    def test_error_counted(self):
        r = Router([RoleRule("default", "m")])
        r.record("m", 50, 0.0, errored=True)
        self.assertEqual(r.stats("m")["n_errors"], 1)


if __name__ == "__main__":
    unittest.main()
