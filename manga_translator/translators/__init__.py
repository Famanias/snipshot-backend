from typing import Optional, List, Dict

import py3langid as langid  # kept for compatibility; may be used elsewhere

# Re-export key symbols from common
from .common import (
    CommonTranslator,
    OfflineTranslator,
    VALID_LANGUAGES,
    ISO_639_1_TO_VALID_LANGUAGES,
)

from .groq import GroqTranslator

from ..config import Translator, TranslatorConfig, TranslatorChain
from ..utils import Context

# --- Minimal translator registry: only Groq ---

TRANSLATORS: Dict[Translator, type[CommonTranslator]] = {
    Translator.groq: GroqTranslator,
}

translator_cache: Dict[Translator, CommonTranslator] = {}


def get_translator(key: Translator, *args, **kwargs) -> CommonTranslator:
    if key not in TRANSLATORS:
        available = ", ".join(t.name for t in TRANSLATORS.keys())
        raise ValueError(
            f'Could not find translator for: "{key}". '
            f"Choose from the following: {available}"
        )
    if key not in translator_cache:
        cls = TRANSLATORS[key]
        translator_cache[key] = cls(*args, **kwargs)
    return translator_cache[key]


async def prepare(chain: TranslatorChain):
    """
    Prepare translators in the chain.

    Original version also handled offline model downloads.
    In this Groq-only build we just validate language support.
    """
    for key, tgt_lang in chain.chain:
        translator = get_translator(key)
        # GroqTranslator should implement supports_languages; if it's a no-op,
        # this still keeps signature compatibility.
        translator.supports_languages("auto", tgt_lang, fatal=True)


async def dispatch(
    chain: TranslatorChain,
    queries: List[str],
    translator_config: Optional[TranslatorConfig] = None,
    use_mtpe: bool = False,
    args: Optional[Context] = None,
    device: str = "cpu",
) -> List[str]:
    """
    Groq-only dispatch: apply translators in the chain sequentially.
    """
    if not queries:
        return queries

    if args is not None and "translations" not in args:
        args["translations"] = {}

    for key, tgt_lang in chain.chain:
        translator = get_translator(key)

        # No offline load/unload: Groq is API-based
        if translator_config:
            translator.parse_args(translator_config)

        # GroqTranslator.translate(src_lang, tgt_lang, queries, use_mtpe)
        queries = await translator.translate("auto", tgt_lang, queries, use_mtpe)

        if args is not None:
            args["translations"][tgt_lang] = queries

    return queries


async def dispatch_batch(
    chain: TranslatorChain,
    batch_queries: List[List[str]],
    translator_config: Optional[TranslatorConfig] = None,
    use_mtpe: bool = False,
    args: Optional[Context] = None,
    device: str = "cpu",
) -> List[List[str]]:
    """
    Batch version of dispatch:

    Flattens all queries, runs dispatch once, then re-groups results back into
    the original batch structure.
    """
    if not batch_queries or not any(batch_queries):
        return batch_queries

    flat_queries: List[str] = []
    mapping: List[int] = []

    for batch_idx, qs in enumerate(batch_queries):
        for q in qs:
            flat_queries.append(q)
            mapping.append(batch_idx)

    flat_results = await dispatch(
        chain,
        flat_queries,
        translator_config=translator_config,
        use_mtpe=use_mtpe,
        args=args,
        device=device,
    )

    batch_results: List[List[str]] = [[] for _ in batch_queries]
    for res, batch_idx in zip(flat_results, mapping):
        batch_results[batch_idx].append(res)

    return batch_results


async def unload(key: Translator):
    """
    Unload a translator instance from the cache.
    For GroqTranslator (API-based), there's nothing heavy to unload,
    but we keep this for API compatibility.
    """
    translator_cache.pop(key, None)


__all__ = [
    "CommonTranslator",
    "OfflineTranslator",
    "VALID_LANGUAGES",
    "ISO_639_1_TO_VALID_LANGUAGES",
    "GroqTranslator",
    "TRANSLATORS",
    "get_translator",
    "prepare",
    "dispatch",
    "dispatch_batch",
    "unload",
]
