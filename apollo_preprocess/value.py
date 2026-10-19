from __future__ import annotations

import re
from dataclasses import dataclass

from apollo_preprocess.models import FormRule, HospitalConfig

_STOP = {"AND", "OF", "THE"}


@dataclass
class FormHit:
    family: str
    title: str
    tier: str
    score: int


def normalize(text: str) -> str:
    text = text.upper().replace("&", " AND ").replace("/", " ")
    text = text.replace("'", " ")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [tok for tok in normalize(text).split() if tok not in _STOP and len(tok) > 1]


def _levenshtein(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _token_match(wanted: str, found: str) -> bool:
    if wanted == found:
        return True
    if len(wanted) >= 4 and len(found) >= 4 and _levenshtein(wanted, found) <= 1:
        return True
    return False


def _phrase_in(text_tokens: list[str], phrase: str) -> bool:
    wanted = _tokens(phrase)
    if not wanted:
        return False
    index = 0
    for token in wanted:
        found = False
        while index < len(text_tokens):
            if _token_match(token, text_tokens[index]):
                index += 1
                found = True
                break
            index += 1
        if not found:
            return False
    return True


def _all_present(text_tokens: list[str], words: list[str]) -> bool:
    pool = list(text_tokens)
    for word in _tokens(" ".join(words)):
        for index, token in enumerate(pool):
            if _token_match(word, token):
                pool.pop(index)
                break
        else:
            return False
    return True


def _rule_matches(rule: FormRule, text_tokens: list[str]) -> int:
    best = 0
    for phrase in rule.phrases:
        if _phrase_in(text_tokens, phrase):
            best = max(best, len(_tokens(phrase)))
    if rule.require_all and _all_present(text_tokens, rule.require_all):
        best = max(best, 1)
    return best


def match_forms(text: str, cfg: HospitalConfig) -> list[FormHit]:
    tokens = _tokens(text)
    hits: list[FormHit] = []
    for rule in cfg.forms:
        score = _rule_matches(rule, tokens)
        if score:
            hits.append(FormHit(rule.id, rule.title, rule.tier, score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits


def choose_tier(
    text: str,
    form_code: str | None,
    cfg: HospitalConfig,
) -> tuple[str | None, str | None, str | None, bool]:
    hits = match_forms(text, cfg)
    by_tier = {"high": [], "medium": [], "low": []}
    for hit in hits:
        by_tier.setdefault(hit.tier, []).append(hit)
    code_tier = cfg.code_tiers.get(form_code) if form_code else None
    for tier in ("high", "medium"):
        if by_tier.get(tier):
            best = by_tier[tier][0]
            return best.tier, best.title, best.family, False
    if by_tier.get("low"):
        best = by_tier["low"][0]
        if code_tier in ("high", "medium"):
            return None, best.title, best.family, True
        return best.tier, best.title, best.family, False
    if code_tier in ("high", "medium", "low"):
        return code_tier, None, None, False
    return None, None, None, False


def apply_value_status(pages: list) -> None:
    for page in pages:
        if page.terminal:
            continue
        if page.value_tier == "low":
            page.status = "skipped_low_value"
        else:
            page.status = "ocr_required"
            if page.value_tier not in ("high", "medium") and "unidentified_form" not in page.reasons:
                page.reasons.append("unidentified_form")
