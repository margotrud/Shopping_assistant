from __future__ import annotations

"""Controlled local generative fallback for mention polarity.

The model receives only mentions already extracted upstream and can return only
LIKE / DISLIKE / UNKNOWN. It never creates mentions or ranks products.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from Shopping_assistant.utils.optional_deps import require

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_ALLOWED_LABELS = {
    "LIKE",
    "DISLIKE",
    "UNKNOWN",
}


def _canon_key(value: str) -> str:
    return value.strip().lower()


@lru_cache(maxsize=2)
def _load_local_model(model_name: str):
    """Load and cache the local causal language model on first fallback use."""
    torch = require(
        "torch",
        extra="torch",
        purpose="Needed for the local generative polarity fallback.",
    )

    if not torch.cuda.is_available():
        cpu_threads = int(
            os.environ.get(
                "SA_LLM_CPU_THREADS",
                str(os.cpu_count() or 1),
            )
        )

        torch.set_num_threads(
            max(1, cpu_threads)
        )

    transformers = require(
        "transformers",
        extra="transformers",
        purpose="Needed for the local generative polarity fallback.",
    )

    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
    )

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def _extract_json_object(
    raw_text: str,
) -> Optional[Dict[str, Any]]:
    """Parse a JSON object, allowing harmless text around one object."""
    if not isinstance(raw_text, str):
        return None

    text = raw_text.strip()

    if not text:
        return None

    try:
        parsed = json.loads(text)

    except json.JSONDecodeError:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1

            parsed = json.loads(
                text[start:end]
            )

        except (
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def _normalize_llm_output(
    raw_text: str,
    mentions: List[str],
) -> Dict[str, Optional[str]]:
    """Validate model output against the fixed caller-provided mention set."""
    parsed = _extract_json_object(
        raw_text
    )

    if parsed is None:
        return {
            mention: None
            for mention in mentions
        }

    parsed_by_key = {
        _canon_key(str(key)): value
        for key, value in parsed.items()
        if isinstance(key, str)
    }

    out: Dict[str, Optional[str]] = {}

    for mention in mentions:
        value = parsed_by_key.get(
            _canon_key(mention)
        )

        if not isinstance(value, str):
            out[mention] = None
            continue

        label = value.strip().upper()

        if (
            label not in _ALLOWED_LABELS
            or label == "UNKNOWN"
        ):
            out[mention] = None
            continue

        out[mention] = label

    return out


@lru_cache(maxsize=4)
def make_local_generative_polarity_fn(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 96,
    debug: bool = False,
):
    """Build a local generative fallback returning LIKE / DISLIKE / None."""
    max_new_tokens_i = max(
        16,
        int(max_new_tokens),
    )

    def _fn(
        clause_text: str,
        mentions: List[str],
    ) -> Dict[str, Optional[str]]:
        if not mentions:
            return {}

        mentions_u = list(
            dict.fromkeys(mentions)
        )

        tokenizer, model, device = (
            _load_local_model(
                model_name
            )
        )

        expected_schema = {
            mention: "LIKE|DISLIKE|UNKNOWN"
            for mention in mentions_u
        }

        expected_schema_json = json.dumps(
            expected_schema,
            ensure_ascii=False,
        )

        system_prompt = (
            "You are a deterministic preference-polarity classifier.\n"
            "You receive one user clause and a fixed list of mentions.\n\n"

            "For every mention, classify the user's preference as exactly one of:\n"
            "- LIKE\n"
            "- DISLIKE\n"
            "- UNKNOWN\n\n"

            "Definitions:\n"
            "- LIKE: the user wants, likes, prefers, accepts, is attracted to, "
            "or expresses a positive preference toward the mention.\n"
            "- DISLIKE: the user rejects, dislikes, avoids, excludes, "
            "or expresses a negative preference toward the mention.\n"
            "- UNKNOWN: the clause does not provide enough evidence to determine "
            "the user's preference toward the mention.\n\n"

            "OUTPUT CONTRACT:\n"
            "- Return JSON only.\n"
            "- Use exactly the keys shown in EXPECTED_SCHEMA.\n"
            "- Do not add keys.\n"
            "- Do not remove keys.\n"
            "- Do not rename keys.\n"
            "- Every value must be exactly LIKE, DISLIKE, or UNKNOWN.\n"
            "- Do not write explanations.\n"
            "- Do not write markdown.\n"
        )
        user_prompt = (
            f"CLAUSE:\n{clause_text}\n\n"
            f"MENTIONS:\n{json.dumps(mentions_u, ensure_ascii=False)}\n\n"
            f"EXPECTED_SCHEMA:\n{expected_schema_json}\n\n"
            "Fill the values in EXPECTED_SCHEMA with the correct labels. "
            "Return the completed JSON object only."
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(device)
            for key, value
            in model_inputs.items()
        }

        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens_i,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_length = model_inputs[
            "input_ids"
        ].shape[1]

        generated_tokens = generated[
            0,
            prompt_length:,
        ]

        raw_output = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        if debug:
            log.debug(
                "[polarity][local-generative] "
                "text=%r mentions=%r raw_output=%r",
                clause_text,
                mentions_u,
                raw_output,
            )

        result_u = _normalize_llm_output(
            raw_output,
            mentions_u,
        )

        return {
            mention: result_u.get(mention)
            for mention in mentions
        }

    return _fn


__all__ = [
    "DEFAULT_MODEL_NAME",
    "make_local_generative_polarity_fn",
]