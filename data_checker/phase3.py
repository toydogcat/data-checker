import random
import json
import re
from typing import List, Dict, Any, Optional


DEFAULT_JUDGE_PROMPT_TEMPLATE = """You are an expert Data Quality Evaluator ("LLM Judge").
Evaluate the following text sample across 3 dimensions on a scale of 1 to 5 stars:
1. Fluency (流暢度): Grammar, natural phrasing, readable formatting.
2. Logic (邏輯性): Coherence, factual consistency, non-contradictory.
3. Value (資訊價值): Information density, usefulness, educational or practical worth.

Text Sample:
---
{text}
---

Return ONLY a raw valid JSON object in the following format (no extra codeblocks or explanations):
{{"fluency": <1-5 integer>, "logic": <1-5 integer>, "value": <1-5 integer>, "reason": "<brief 1-sentence critique>"}}
"""


class Phase3Checker:
    """
    階段三：引入「Llama / LLM as a Judge」抽樣審查
    Phase 3: LLM Judge sampling evaluation for content quality, fluency, logic, and value.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        prompt_template: str = DEFAULT_JUDGE_PROMPT_TEMPLATE
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_template = prompt_template
        self._openai_client = None

        if self.api_key or self.base_url:
            self._init_client()

    def _init_client(self):
        try:
            import openai
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._openai_client = openai.OpenAI(**kwargs)
        except Exception:
            self._openai_client = None

    def _mock_judge_evaluate(self, text: str) -> Dict[str, Any]:
        """
        Heuristic mock judge for standalone local/Colab execution when API key is missing.
        """
        text_str = str(text).strip()
        length = len(text_str)

        # Fluency heuristic: penalize broken characters, non-printable text, excessive repetition
        fluency = 5
        if "\ufffd" in text_str or re.search(r"[\x00-\x08]", text_str):
            fluency -= 2
        if len(set(text_str)) < 5 and length > 10:
            fluency -= 2
        fluency = max(1, min(5, fluency))

        # Logic heuristic
        logic = 4
        if length < 5:
            logic = 2
        elif "謝謝" in text_str and length < 6:
            logic = 3

        # Value heuristic: reward moderate text length and diverse vocabulary
        words = re.findall(r"\w+", text_str)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0.5

        if length < 10:
            value = 2
        elif ttr > 0.6 and length > 30:
            value = 5
        elif ttr > 0.4:
            value = 4
        else:
            value = 3

        reasons = [
            "語意完整，資訊密度良好。",
            "句子流暢，邏輯清晰。",
            "字數偏短，資訊價值有限。",
            "含有較多重複詞彙，整體品質中等。",
            "文字結構完整且具參考價值。"
        ]
        reason = random.choice(reasons)

        return {
            "fluency": fluency,
            "logic": logic,
            "value": value,
            "reason": reason
        }

    def evaluate_sample(self, text: str) -> Dict[str, Any]:
        """Evaluate a single text record using OpenAI API or fallback heuristic."""
        if self._openai_client is None:
            return self._mock_judge_evaluate(text)

        prompt = self.prompt_template.format(text=text)
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality judge. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            # Parse JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "fluency": int(result.get("fluency", 3)),
                    "logic": int(result.get("logic", 3)),
                    "value": int(result.get("value", 3)),
                    "reason": str(result.get("reason", ""))
                }
        except Exception:
            pass

        return self._mock_judge_evaluate(text)

    def audit_corpus(self, texts: List[str], sample_size: int = 50, random_seed: int = 42) -> Dict[str, Any]:
        """
        Randomly sample N records and run LLM Judge evaluation.
        """
        clean_texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not clean_texts:
            return {"error": "Corpus is empty"}

        if len(clean_texts) > sample_size:
            random.seed(random_seed)
            sampled_indices = random.sample(range(len(clean_texts)), sample_size)
        else:
            sampled_indices = list(range(len(clean_texts)))

        evaluations = []
        fluency_scores = []
        logic_scores = []
        value_scores = []

        for idx in sampled_indices:
            sample_text = clean_texts[idx]
            eval_res = self.evaluate_sample(sample_text)
            eval_res["sample_index"] = idx
            eval_res["text_snippet"] = sample_text[:60] + "..." if len(sample_text) > 60 else sample_text
            evaluations.append(eval_res)

            fluency_scores.append(eval_res["fluency"])
            logic_scores.append(eval_res["logic"])
            value_scores.append(eval_res["value"])

        avg_fluency = round(sum(fluency_scores) / len(fluency_scores), 2)
        avg_logic = round(sum(logic_scores) / len(logic_scores), 2)
        avg_value = round(sum(value_scores) / len(value_scores), 2)
        overall_avg = round((avg_fluency + avg_logic + avg_value) / 3.0, 2)

        return {
            "sample_count": len(evaluations),
            "execution_mode": "api" if self._openai_client is not None else "mock_heuristic",
            "model_used": self.model if self._openai_client is not None else "heuristic_engine",
            "average_scores": {
                "fluency": avg_fluency,
                "logic": avg_logic,
                "value": avg_value,
                "overall_score": overall_avg
            },
            "judge_pass": overall_avg >= 4.0,
            "sample_evaluations": evaluations
        }
