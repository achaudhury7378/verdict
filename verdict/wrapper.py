"""
verdict/wrapper.py - Reusable wrappers for arbitrary LLM-powered callables.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from verdict.core import Experiment, Variant


def _default_normalize(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "output" in raw:
            return raw
        if "answer" in raw:
            return {"output": str(raw["answer"])}
        return {"output": str(raw)}
    return str(raw)


@dataclass
class VerdictWrapper:
    """
    Attach verdict benchmarking behavior to any callable.

    The wrapped callable remains directly callable, but it can also be
    converted into a `Variant` or combined into an `Experiment`.
    """

    fn: Callable
    name: str
    model: Optional[str] = None
    pricing: Optional[dict] = None
    config: dict = field(default_factory=dict)
    prepare: Optional[Callable[[Any, dict], Any]] = None
    normalize: Optional[Callable[[Any], Any]] = None
    metadata: dict = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def variant(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        pricing: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> Variant:
        merged_config = dict(self.config)
        if config:
            merged_config.update(config)

        return Variant(
            name=name or self.name,
            fn=self._variant_call,
            config=merged_config,
            model=model or self.model,
            pricing=pricing or self.pricing,
        )

    def experiment(
        self,
        *,
        against: Optional[Iterable[Any]] = None,
        queries: list,
        judge,
        name: str = "verdict experiment",
        runs: int = 20,
        monthly_volume: int = 10_000,
        seed: Optional[int] = None,
        max_workers: int = 10,
    ) -> Experiment:
        variants = [self.variant()]
        for item in against or []:
            if isinstance(item, VerdictWrapper):
                variants.append(item.variant())
            elif isinstance(item, Variant):
                variants.append(item)
            else:
                raise TypeError(
                    "against= items must be VerdictWrapper or Variant instances."
                )

        return Experiment(
            name=name,
            variants=variants,
            queries=queries,
            judge=judge,
            runs=runs,
            monthly_volume=monthly_volume,
            seed=seed,
            max_workers=max_workers,
        )

    def _variant_call(self, config: dict, query: Any):
        payload = self.prepare(query, config) if self.prepare else query
        raw = self._invoke(payload, config)
        normalizer = self.normalize or _default_normalize
        return normalizer(raw)

    def _invoke(self, payload: Any, config: dict):
        try:
            signature = inspect.signature(self.fn)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            params = list(signature.parameters.values())
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                return self.fn(config, payload)

            positional = [
                p
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            required = [p for p in positional if p.default is inspect._empty]

            if len(required) >= 2 or len(positional) >= 2:
                return self.fn(config, payload)
            if len(positional) >= 1:
                return self.fn(payload)
            return self.fn()

        try:
            return self.fn(config, payload)
        except TypeError:
            return self.fn(payload)


def verdict_wrap(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    model: Optional[str] = None,
    pricing: Optional[dict] = None,
    config: Optional[dict] = None,
    prepare: Optional[Callable[[Any, dict], Any]] = None,
    normalize: Optional[Callable[[Any], Any]] = None,
    metadata: Optional[dict] = None,
):
    """
    Decorator/helper that turns a callable into a VerdictWrapper.

    Example
    -------
    @verdict_wrap(name="rag-bot", model="gpt-4o")
    def answer(question):
        ...
    """

    def _decorate(callable_obj: Callable) -> VerdictWrapper:
        return VerdictWrapper(
            fn=callable_obj,
            name=name or getattr(callable_obj, "__name__", "wrapped-callable"),
            model=model,
            pricing=pricing,
            config=dict(config or {}),
            prepare=prepare,
            normalize=normalize,
            metadata=dict(metadata or {}),
        )

    if fn is not None:
        return _decorate(fn)
    return _decorate


def build_experiment(
    wrappers: Iterable[Any],
    *,
    queries: list,
    judge,
    name: str = "verdict experiment",
    runs: int = 20,
    monthly_volume: int = 10_000,
    seed: Optional[int] = None,
    max_workers: int = 10,
) -> Experiment:
    """
    Build an Experiment from VerdictWrapper and/or Variant objects.
    """

    variants = []
    for item in wrappers:
        if isinstance(item, VerdictWrapper):
            variants.append(item.variant())
        elif isinstance(item, Variant):
            variants.append(item)
        else:
            raise TypeError("wrappers must contain only VerdictWrapper or Variant.")

    return Experiment(
        name=name,
        variants=variants,
        queries=queries,
        judge=judge,
        runs=runs,
        monthly_volume=monthly_volume,
        seed=seed,
        max_workers=max_workers,
    )
