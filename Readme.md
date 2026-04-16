# Verdict

**A/B testing for LLM systems with statistical proof, cost tracking, and human-readable reporting.**

Verdict helps you compare prompts, models, RAG pipelines, chatbots, and agentic systems on the same tasks. It runs repeated evaluations, scores outputs with a judge model, estimates quality with confidence intervals, and now explains the metrics in plain English so results are easier to act on.

## What It Does

- Runs repeatable experiments across multiple variants
- Measures judged quality, latency, failures, and cost
- Supports OpenAI token pricing, including cached-input pricing when usage metadata includes cached token counts
- Produces a terminal report with a metric cheat sheet, quick-read takeaways, and plain-English statistical explanations
- Lets you benchmark existing app callables with a wrapper instead of forcing everything into manual `Variant(...)` setup

## Install

```bash
pip install llm-arbiter
```

## Quick Start

```python
from openai import OpenAI
from verdict import Experiment, Judge, Variant

client = OpenAI()


def call_model(config, query):
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": query}],
        temperature=0.2,
    )
    return {
        "output": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": getattr(
                    getattr(response.usage, "prompt_tokens_details", None),
                    "cached_tokens",
                    0,
                )
            },
        },
    }


experiment = Experiment(
    name="Simple benchmark",
    variants=[
        Variant("gpt-4o", call_model, config={"model": "gpt-4o"}, model="gpt-4o"),
        Variant(
            "gpt-4o-mini",
            call_model,
            config={"model": "gpt-4o-mini"},
            model="gpt-4o-mini",
        ),
    ],
    queries=[
        "Explain recursion simply.",
        "Summarize the tradeoffs of electric cars.",
    ],
    judge=Judge(
        criteria=[
            "Is the answer correct?",
            "Is it clear and easy to follow?",
            "Does it answer the question directly?",
        ]
    ),
    runs=3,
)

analysis = experiment.run()
analysis.display()
```

## Wrapper Pattern

Use `verdict_wrap` when you already have an app callable and want to benchmark it without rewriting your system around `Variant`.

```python
from verdict import Judge, build_experiment, verdict_wrap


@verdict_wrap(name="rag-bot", model="gpt-4o")
def rag_answer(question):
    return my_rag_pipeline(question)


@verdict_wrap(name="agent-bot", model="gpt-4.1-mini")
def agent_answer(question):
    return my_agent.run(question)


experiment = build_experiment(
    [rag_answer, agent_answer],
    queries=[
        "Summarize the customer complaint.",
        "What action should we take next?",
    ],
    judge=Judge(
        criteria=[
            "Is the answer correct?",
            "Does it use the available context well?",
            "Is it clear and actionable?",
        ]
    ),
    runs=3,
)

analysis = experiment.run()
analysis.display()
```

### Wrapper Hooks

- `prepare=query_transformer`
  Use this when your app expects a richer payload than a raw string.
- `normalize=response_transformer`
  Use this when your callable returns a custom object and you want to convert it into either a plain string or `{"output": ..., "usage": ...}`.
- Wrapped callables stay callable
  You can keep using them normally in your app and also benchmark them later.

## Supported Return Shapes

Verdict accepts either of these callable return formats:

```python
"plain text output"
```

or

```python
{
    "output": "...",
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 220,
        "cached_input_tokens": 800,
    },
}
```

It also understands OpenAI-style usage keys such as:

- `prompt_tokens`
- `completion_tokens`
- `prompt_tokens_details.cached_tokens`
- `input_tokens_details.cached_tokens`

## Cost Calculation

Verdict calculates per-run cost as:

```text
(uncached_input_tokens * input_price)
+ (cached_input_tokens * cached_input_price)
+ (output_tokens * output_price)
```

If cached token details are not present, Verdict falls back to normal input pricing for all prompt tokens.

## OpenAI Compatibility Helper

Some newer models such as `gpt-5-mini` expect `max_completion_tokens` instead of `max_tokens`.
Verdict includes a small helper for that:

```python
from verdict import output_token_limit_arg

response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Hello"}],
    **output_token_limit_arg(model_name, 80),
)
```

The built-in `Judge` already uses this helper automatically.

## Explainable Report

The terminal report now includes:

- `METRIC CHEAT SHEET`
  Short explanations of what each metric means
- `QUICK READ`
  A compact dashboard-style interpretation for each variant
- `STATS IN PLAIN ENGLISH`
  A readable explanation of pairwise comparisons

This makes the results easier to understand for non-specialists and stakeholders who are not comfortable reading p-values or confidence intervals directly.

## Built-In Pricing Table

The library includes built-in pricing entries for these OpenAI model families:

- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-5.1`
- `gpt-5.2`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`
- `o3`
- `o3-mini`
- `o4-mini`

You can always override pricing per variant:

```python
Variant(
    name="custom-model",
    fn=call_model,
    pricing={
        "input": 2.0 / 1_000_000,
        "cached_input": 0.5 / 1_000_000,
        "output": 8.0 / 1_000_000,
    },
)
```

## Core API

### `Variant`

Represents one system under test.

```python
Variant(
    name="gpt-4o",
    fn=call_model,
    config={"model": "gpt-4o"},
    model="gpt-4o",
    pricing=None,
)
```

### `Experiment`

Runs all variants across all queries for a fixed number of repetitions.

Key parameters:

- `variants`
- `queries`
- `judge`
- `runs`
- `monthly_volume`
- `seed`
- `max_workers`

### `Judge`

Scores outputs on a 1-10 scale using an OpenAI model.

```python
judge = Judge(
    criteria=[
        "Is it correct?",
        "Is it specific?",
        "Is it actionable?",
    ],
    model="gpt-4o-mini",
)
```

### `Analysis`

Returned by `experiment.run()`.

Useful methods:

- `summary(variant_name)`
- `compare(name_a, name_b)`
- `all_comparisons()`
- `query_breakdown()`
- `recommendation()`
- `display()`

## Example Files In This Repo

- `test.py`
  Wrapper-based OpenAI benchmark example
- `test_api.py`
  Simple API connectivity test

## Environment

Set your API key before running examples:

```bash
export OPENAI_API_KEY="sk-..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

## Notes

- If your terminal is not UTF-8, Unicode progress bars may render poorly
- Cached pricing only applies when your usage payload includes cached token details
- If you use a provider or SDK that does not return usage, Verdict falls back to rough token estimation for plain string outputs

## License

MIT
