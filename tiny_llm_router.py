"""
tiny-llm-router — Smart Model Routing for Python AI Agents
---------------------------------------------------------
Pick the right model for each task based on role, cost, latency, and capability.
Zero deps. One file. MIT.

Designed for the multi-model era: route chat/coding/vision/reasoning across
fast/cheap (Kimi-K2.6, Haiku), heavy reasoning (Nemotron-3 Super, o1),
vision (Qwen3.6-VL, GPT-4o), and code-specialized (Qwen3-Coder, GPT-4o).

Usage:
    router = Router([
        RoleRule("code",     "gpt-4o-mini"),
        RoleRule("think",    "o1-mini"),
        RoleRule("vision",   "gpt-4o"),
        RoleRule("default",  "gpt-4o-mini"),
    ], fallback="gpt-4o")
    model = router.pick(task=Task(role="code", prompt="...", has_images=False))
"""
from __future__ import annotations
import time, threading, re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---- Task / Rule model -----------------------------------------------------

@dataclass
class Task:
    role: str = "default"            # code | think | vision | chat | default | <custom>
    prompt: str = ""                 # used for heuristics (length, has_code, has_images)
    has_images: bool = False         # explicit signal (often the source knows best)
    estimated_input_tokens: int = 0  # optional cost-aware signal
    needs_json: bool = False         # JSON-mode-required task
    latency_budget_ms: Optional[int] = None  # None = no SLA
    metadata: dict = field(default_factory=dict)


@dataclass
class RoleRule:
    role: str
    model: str
    max_cost_per_1k: Optional[float] = None    # $ cap per 1k tokens (input+output avg)
    max_latency_p95_ms: Optional[int] = None  # observed p95 latency
    only_if: Optional[Callable[[Task], bool]] = None  # extra guard


# ---- Statistics (for adaptive routing) -------------------------------------

@dataclass
class ModelStat:
    n_calls: int = 0
    n_errors: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    quality_score: float = 1.0     # set externally or from eval

    def record(self, latency_ms: float, cost_usd: float, errored: bool = False) -> None:
        with _STAT_LOCK:
            self.n_calls += 1
            if errored:
                self.n_errors += 1
            # rolling avg
            n = self.n_calls
            self.avg_latency_ms = self.avg_latency_ms + (latency_ms - self.avg_latency_ms) / n
            self.avg_cost_usd = self.avg_cost_usd + (cost_usd - self.avg_cost_usd) / n
            # simple p95 proxy = max of last 20
            last = getattr(self, "_last_latencies", [])
            last.append(latency_ms)
            if len(last) > 20:
                last.pop(0)
            self._last_latencies = last
            self.p95_latency_ms = sorted(last)[int(0.95 * (len(last) - 1))]

_STAT_LOCK = threading.Lock()


# ---- Heuristics for role inference -----------------------------------------

_CODE_HINTS = re.compile(r"\b(def|class|import|from|return|if|for|while|\{|\}|=>|\.py|\.js|\.ts)\b")
_JSON_HINTS = re.compile(r"^\s*(\{|\[|\")", re.MULTILINE)
_LONG_PROMPT = 4000  # chars


def infer_role(task: Task) -> str:
    """Best-effort role inference when caller didn't pick."""
    if task.has_images:
        return "vision"
    if task.needs_json:
        return "json"
    if task.prompt:
        if _CODE_HINTS.search(task.prompt):
            return "code"
        if len(task.prompt) > _LONG_PROMPT:
            return "long"
        if _JSON_HINTS.search(task.prompt):
            return "json"
    return "default"


# ---- Router ----------------------------------------------------------------

class RoutingError(Exception):
    """Raised when no model can be selected (all rules fail)."""


class Router:
    def __init__(self, rules, fallback: Optional[str] = None,
                 pricing: Optional[dict] = None,
                 failure_window_s: float = 60.0,
                 failure_threshold: int = 3):
        """
        rules: list[RoleRule]
        fallback: model name used when no role rule matches OR all role rules' only_if fails
        pricing: model -> {input_per_1k, output_per_1k}; optional
        """
        self._by_role = {r.role: r for r in rules}
        self._rules = list(rules)
        self._fallback = fallback
        self._pricing = pricing or {}
        self._stats: dict[str, ModelStat] = {}
        # failure tracking -> circuit-break per model
        self._failures: dict[str, list[float]] = {}
        self._failure_window_s = failure_window_s
        self._failure_threshold = failure_threshold

    # ---- public API ----

    def pick(self, task: Task) -> str:
        """Return model name for the task. Applies role rules + cost/latency guards
        + failure circuit-break. Falls through candidates on exclusion.
        Raises RoutingError if nothing fits."""
        role = task.role if task.role != "default" else infer_role(task)
        primary_rule = self._by_role.get(role) or self._by_role.get("default")

        # Build ordered candidate list with their source rule.
        candidates: list[tuple[str, Optional[RoleRule]]] = []
        if primary_rule is not None and self._rule_applies(primary_rule, task):
            candidates.append((primary_rule.model, primary_rule))
        if self._fallback and not any(m == self._fallback for m, _ in candidates):
            fb_rule = RoleRule("__fallback__", self._fallback)
            candidates.append((self._fallback, fb_rule))

        for model, rule in candidates:
            if self._is_circuit_broken(model):
                continue
            if self._over_cost_cap(task, rule) or self._over_latency_cap(task, rule):
                continue
            return model
        raise RoutingError(f"No model available for role={role!r}; candidates={[m for m, _ in candidates]}")

    def _rule_applies(self, rule: RoleRule, task: Task) -> bool:
        """True when rule has no `only_if`, or only_if(task) is True."""
        if rule.only_if is None:
            return True
        try:
            return bool(rule.only_if(task))
        except Exception:
            return False

    def record(self, model: str, latency_ms: float, cost_usd: float,
               errored: bool = False) -> None:
        """Update internal stats + failure tracker after a real call."""
        st = self._stats.setdefault(model, ModelStat())
        st.record(latency_ms, cost_usd, errored=errored)
        if errored:
            now = time.time()
            buf = self._failures.setdefault(model, [])
            buf.append(now)
            # windowed prune
            cutoff = now - self._failure_window_s
            self._failures[model] = [t for t in buf if t >= cutoff]

    def stats(self, model: Optional[str] = None) -> dict:
        if model is None:
            return {m: s.__dict__.copy() for m, s in self._stats.items()}
        st = self._stats.get(model)
        return st.__dict__.copy() if st else {}

    def is_healthy(self, model: str) -> bool:
        return not self._is_circuit_broken(model)

    # ---- internal ----

    def _is_circuit_broken(self, model: str) -> bool:
        now = time.time()
        buf = [t for t in self._failures.get(model, []) if now - t < self._failure_window_s]
        self._failures[model] = buf
        return len(buf) >= self._failure_threshold

    def _over_cost_cap(self, task: Task, rule: Optional[RoleRule]) -> bool:
        if rule is None or rule.max_cost_per_1k is None:
            return False
        p = self._pricing.get(rule.model)
        if not p:
            return False
        avg_per_1k = (p.get("input_per_1k", 0) + p.get("output_per_1k", 0)) / 2
        return avg_per_1k > rule.max_cost_per_1k

    def _over_latency_cap(self, task: Task, rule: Optional[RoleRule]) -> bool:
        if rule is None or rule.max_latency_p95_ms is None:
            return False
        st = self._stats.get(rule.model)
        if not st or st.p95_latency_ms == 0:
            return False
        return st.p95_latency_ms > rule.max_latency_p95_ms


# ---- Convenience: integrate with tiny-llm and tiny-cost --------------------

def make_tiny_llm_adapter(router: Router, llm_module):
    """
    Wrap an instance of tiny_llm.LLM so .chat() auto-routes based on Task hints.

    llm.chat_auto(task) -> str  (sets role/prompt/etc from Task, then dispatches)
    """
    class _Adapter:
        def __init__(self):
            self.router = router
            self.llm = llm_module.LLM()
            self._last_model = None

        def chat_auto(self, task: Task, **kw):
            model = self.router.pick(task)
            self._last_model = model
            t0 = time.time()
            err = False
            try:
                resp = self.llm.chat(model=model, messages=[{"role": "user", "content": task.prompt}], **kw)
                return resp
            except Exception:
                err = True
                raise
            finally:
                dt = (time.time() - t0) * 1000
                cost = getattr(resp, "cost_usd", 0.0) if not err else 0.0
                self.router.record(model, dt, cost, errored=err)

        @property
        def last_model(self):
            return self._last_model

    return _Adapter()
