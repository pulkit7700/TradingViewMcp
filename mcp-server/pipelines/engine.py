"""
Pipeline execution engine — modular, chainable, async.

Each step is a coroutine: async (ctx: PipelineContext) -> PipelineContext.
Steps share state via ctx.data dict and accumulate results in ctx.results.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared context flowing through all pipeline steps."""
    ticker: str
    config: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)       # intermediate data (ohlcv, features, etc.)
    results: dict = field(default_factory=dict)    # computed outputs (signals, forecasts, etc.)
    errors: dict = field(default_factory=dict)     # step_name → error str (non-fatal)
    meta: dict = field(default_factory=dict)       # timing, source, etc.

    def put(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def result(self, key: str, value: Any):
        self.results[key] = value

    def get_result(self, key: str, default=None):
        return self.results.get(key, default)

    def record_error(self, step: str, error: Exception):
        self.errors[step] = str(error)
        logger.warning("Pipeline step '%s' failed: %s", step, error)


StepFn = Callable[[PipelineContext], Coroutine[Any, Any, PipelineContext]]


@dataclass
class PipelineStep:
    name: str
    fn: StepFn
    optional: bool = False  # if True, failure doesn't abort pipeline


class Pipeline:
    """
    Ordered chain of async steps.
    Each step receives the full context and returns it (possibly enriched).
    Optional steps log errors and continue; required steps re-raise.
    """

    def __init__(self, name: str, steps: list[PipelineStep]):
        self.name = name
        self.steps = steps

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.meta["pipeline"] = self.name
        ctx.meta["started_at"] = time.time()
        logger.info("Pipeline '%s' started for ticker=%s", self.name, ctx.ticker)

        for step in self.steps:
            step_start = time.time()
            try:
                ctx = await step.fn(ctx)
                elapsed = time.time() - step_start
                logger.debug("Step '%s' completed in %.2fs", step.name, elapsed)
                ctx.meta[f"step_{step.name}_elapsed"] = elapsed
            except Exception as e:
                if step.optional:
                    ctx.record_error(step.name, e)
                else:
                    logger.error("Required step '%s' failed: %s", step.name, e)
                    raise

        ctx.meta["total_elapsed"] = time.time() - ctx.meta["started_at"]
        logger.info(
            "Pipeline '%s' done in %.2fs | errors: %s",
            self.name,
            ctx.meta["total_elapsed"],
            list(ctx.errors.keys()) or "none",
        )
        return ctx


class PipelineEngine:
    """
    Manages named pipelines and supports concurrent execution.
    Results are cached for config-specified TTL.
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._pipelines: dict[str, Pipeline] = {}
        self._cache: dict[str, tuple[float, dict]] = {}  # key → (ts, results)
        self._ttl = config.get("pipelines", {}).get("result_ttl_seconds", 600)
        self._max_concurrent = config.get("pipelines", {}).get("max_concurrent", 4)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    def register(self, pipeline: Pipeline):
        self._pipelines[pipeline.name] = pipeline
        logger.info("Registered pipeline: %s", pipeline.name)

    async def run(self, name: str, ticker: str, extra: dict | None = None) -> dict:
        cache_key = f"{name}:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            ts, results = cached
            if time.time() - ts < self._ttl:
                logger.debug("Cache hit for %s", cache_key)
                return results

        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not registered. Available: {list(self._pipelines)}")

        pipeline = self._pipelines[name]
        ctx = PipelineContext(
            ticker=ticker,
            config=self._cfg,
            data=extra or {},
        )

        async with self._semaphore:
            ctx = await pipeline.run(ctx)

        output = {
            "ticker": ticker,
            "pipeline": name,
            "results": ctx.results,
            "errors": ctx.errors,
            "meta": ctx.meta,
        }
        self._cache[cache_key] = (time.time(), output)
        return output

    async def run_multi(
        self, name: str, tickers: list[str], extra: dict | None = None
    ) -> dict[str, dict]:
        tasks = {t: self.run(name, t, extra) for t in tickers}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            ticker: (res if not isinstance(res, Exception) else {"error": str(res)})
            for ticker, res in zip(tasks.keys(), results)
        }

    def list_pipelines(self) -> list[str]:
        return list(self._pipelines.keys())

    def invalidate_cache(self, name: str | None = None, ticker: str | None = None):
        if name and ticker:
            self._cache.pop(f"{name}:{ticker}", None)
        elif name:
            for k in list(self._cache.keys()):
                if k.startswith(f"{name}:"):
                    del self._cache[k]
        else:
            self._cache.clear()
