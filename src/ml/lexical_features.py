"""Simple, deterministic code-pattern counts computed from ONE function's source text.

Why: the five Lizard metrics mostly measure size. These counts describe what the code does
(buffer/string calls, allocation, pointer use, error checks, control flow). Chapter Two cites text-based
features for vulnerability prediction (Scandariato et al., 2014). Comments and string literals are removed first.
The same function is used at training time and at prediction time (uploaded complete files).
"""
from __future__ import annotations

import re

LEX_NAMES = ["string_copy_calls", "alloc_free_calls", "total_calls", "arrow_ops", "array_index_ops", "pointer_casts",
             "goto_count", "return_count", "if_count", "loop_count", "switch_case_count", "sizeof_count", "null_checks"]

STRING_COPY = {"strcpy", "strncpy", "strcat", "strncat", "sprintf", "vsprintf", "snprintf", "vsnprintf", "gets",
               "memcpy", "memmove", "memset", "bcopy", "wcscpy", "wcscat"}
ALLOC_FREE = {"malloc", "calloc", "realloc", "free", "alloca", "kmalloc", "kzalloc", "kfree", "g_malloc", "g_free",
              "xmalloc", "xfree"}
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else", "case", "typeof", "alignof",
            "defined", "static_assert", "decltype", "noexcept", "throw", "new", "delete"}

_NOISE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_CAST = re.compile(r"\(\s*(?:(?:const|struct|unsigned|signed|volatile)\s+)*[A-Za-z_]\w*(?:\s*::\s*\w+)*\s*\*+\s*\)")


def strip_noise(text: str) -> str:
    return _NOISE.sub(" ", text)


def _count(pattern: str, s: str) -> int:
    return len(re.findall(pattern, s))


def extract(text: str) -> dict:
    s = strip_noise(text)
    calls = [c for c in _CALL.findall(s) if c not in KEYWORDS]
    return {
        "string_copy_calls": sum(c in STRING_COPY for c in calls),
        "alloc_free_calls": sum(c in ALLOC_FREE for c in calls) + _count(r"\b(?:new|delete)\b", s),
        "total_calls": len(calls),
        "arrow_ops": s.count("->"),
        "array_index_ops": s.count("["),
        "pointer_casts": len(_CAST.findall(s)),
        "goto_count": _count(r"\bgoto\b", s),
        "return_count": _count(r"\breturn\b", s),
        "if_count": _count(r"\bif\b", s),
        "loop_count": _count(r"\b(?:for|while)\b", s),
        "switch_case_count": _count(r"\b(?:switch|case)\b", s),
        "sizeof_count": _count(r"\bsizeof\b", s),
        "null_checks": _count(r"\b(?:NULL|nullptr)\b", s),
    }
