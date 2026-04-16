
"""
verdict/judge.py — LLM Judge using OpenAI.
"""

import re
import json
from openai import OpenAI
from verdict.openai_compat import output_token_limit_arg


class Judge:
    """
    Scores model outputs using GPT-4o-mini.

    Parameters
    ----------
    criteria : list[str]
        What to evaluate.
        Example: ["Does it cite specific numbers?",
                  "Is the analysis correct?"]
    model : str
        Judge model. Default "gpt-4o-mini".
    """

    def __init__(self, criteria: list, model: str = "gpt-4o-mini"):
        self.criteria = criteria
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def score(self, output: str, query_text: str) -> float:
        """
        Score a single output. Returns float 1.0 to 10.0.

        Parameters
        ----------
        output : str
            The model's response to evaluate.
        query_text : str
            The question that was asked.
        """
        criteria_text = "\n".join(
            f"  {i+1}. {c}" for i, c in enumerate(self.criteria)
        )

        prompt = f"""Score this answer from 1 to 10. Be strict.

QUESTION:
{query_text}

ANSWER:
{output[:3000]}

CRITERIA:
{criteria_text}

RULES:
- 10 = perfect on ALL criteria
- 5 = acceptable but flawed
- Below 3 = seriously wrong or incomplete
- Be strict. Vague answers get low scores.

Respond with ONLY this JSON:
{{"score": <number>, "reason": "<one sentence>"}}"""

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You evaluate AI outputs. Respond with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                **output_token_limit_arg(self.model, 80),
            )
            return self._parse(r.choices[0].message.content.strip())

        except Exception as e:
            print(f"    ⚠️ Judge error: {e}")
            return 5.0

    def _parse(self, text: str) -> float:
        """Extract score from judge response."""
        # Try JSON first
        try:
            match = re.search(r"\{[^}]+\}", text)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 5))
                return max(1.0, min(10.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: any number 1-10
        for n in re.findall(r"\b(\d+\.?\d*)\b", text):
            val = float(n)
            if 1 <= val <= 10:
                return val

        return 5.0




