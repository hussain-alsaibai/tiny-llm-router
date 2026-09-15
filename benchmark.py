"""Benchmark tiny_llm_router pick + record throughput."""
import time, random
from tiny_llm_router import Router, RoleRule, Task

random.seed(42)


def bench():
    rules = [
        RoleRule("code", "gpt-4o-mini"),
        RoleRule("think", "o1-mini"),
        RoleRule("vision", "gpt-4o"),
        RoleRule("long", "gemini-2.5-flash"),
        RoleRule("json", "haiku"),
        RoleRule("default", "gpt-4o-mini"),
    ]
    r = Router(rules, fallback="gpt-4o-mini")
    tasks = [
        Task(role="code", prompt="def hello(): return 1"),
        Task(role="think"),
        Task(role="vision", has_images=True),
        Task(role="default", prompt="hi"),
        Task(role="json", needs_json=True),
        Task(role="long", prompt="x" * 5000),
    ] * 200  # 1200 tasks

    t0 = time.time()
    picked = {}
    for t in tasks:
        m = r.pick(t)
        picked[m] = picked.get(m, 0) + 1
        r.record(m, latency_ms=random.uniform(50, 500),
                 cost_usd=random.uniform(0.0001, 0.01))
    dt = time.time() - t0

    print(f"Tasks processed: {len(tasks)} in {dt*1000:.2f} ms")
    print(f"Throughput:      {len(tasks)/dt:.0f} picks/sec")
    print(f"Avg latency:     {dt/len(tasks)*1e6:.1f} us/pick")
    print(f"Distribution:    {picked}")
    print(f"Failure window:  healthy after {r._failure_window_s}s of no errors")


if __name__ == "__main__":
    bench()
