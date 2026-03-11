"""Translation module — Groq LLM API translator."""

import re
import os
import time
import asyncio
from typing import List
from abc import abstractmethod

import groq
from dotenv import load_dotenv

from ..utils.log import get_logger
from ..utils.generic import repeating_sequence
from ..utils.generic2 import is_valuable_text

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")

VALID_LANGUAGES = {
    "CHS": "Chinese (Simplified)", "CHT": "Chinese (Traditional)", "CSY": "Czech",
    "NLD": "Dutch", "ENG": "English", "FRA": "French", "DEU": "German",
    "HUN": "Hungarian", "ITA": "Italian", "JPN": "Japanese", "KOR": "Korean",
    "POL": "Polish", "PTB": "Portuguese (Brazil)", "ROM": "Romanian", "RUS": "Russian",
    "ESP": "Spanish", "TRK": "Turkish", "UKR": "Ukrainian", "VIN": "Vietnamese",
    "ARA": "Arabic", "THA": "Thai", "IND": "Indonesian", "FIL": "Filipino (Tagalog)",
}

ISO_639_1_TO_VALID_LANGUAGES = {
    "zh": "CHS", "ja": "JPN", "en": "ENG", "ko": "KOR", "vi": "VIN",
    "cs": "CSY", "nl": "NLD", "fr": "FRA", "de": "DEU", "hu": "HUN",
    "it": "ITA", "pl": "POL", "pt": "PTB", "ro": "ROM", "ru": "RUS",
    "es": "ESP", "tr": "TRK", "uk": "UKR", "ar": "ARA", "th": "THA",
    "id": "IND", "tl": "FIL",
}


class MissingAPIKeyException(Exception):
    pass


# ── Base translator ──────────────────────────────────────────────────

class CommonTranslator:
    _LANGUAGE_CODE_MAP = {}
    _INVALID_REPEAT_COUNT = 0
    _MAX_REQUESTS_PER_MINUTE = -1

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._last_request_ts = 0

    def parse_language_codes(self, from_lang, to_lang, fatal=False):
        _from = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != "auto" else "auto"
        _to = self._LANGUAGE_CODE_MAP.get(to_lang)
        if fatal and _to is None:
            raise ValueError(f"Unsupported target language: {to_lang}")
        return _from, _to

    async def translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        if to_lang not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language: {to_lang}")
        self.logger.info("Translating into %s", VALID_LANGUAGES[to_lang])

        if from_lang == to_lang:
            return queries

        query_indices = []
        final = []
        for i, q in enumerate(queries):
            if not is_valuable_text(q):
                final.append(q)
            else:
                final.append(None)
                query_indices.append(i)

        active_queries = [queries[i] for i in query_indices]
        translations = [""] * len(active_queries)

        await self._ratelimit_sleep()
        _from, _to = self.parse_language_codes(from_lang, to_lang, fatal=True)
        raw = await self._translate(_from, _to, active_queries)

        if len(raw) < len(active_queries):
            raw.extend([""] * (len(active_queries) - len(raw)))
        elif len(raw) > len(active_queries):
            raw = raw[: len(active_queries)]

        for j in range(len(active_queries)):
            translations[j] = raw[j]

        translations = [self._clean(q, t, to_lang) for q, t in zip(active_queries, translations)]

        for i, trans in enumerate(translations):
            final[query_indices[i]] = trans
            self.logger.info("%d: %s => %s", i, active_queries[i], trans)

        return final

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            timeout = self._last_request_ts + 60 / self._MAX_REQUESTS_PER_MINUTE
            if timeout > now:
                await asyncio.sleep(timeout - now)
            self._last_request_ts = time.time()

    @staticmethod
    def _clean(query: str, trans: str, to_lang: str) -> str:
        if not query or not trans:
            return ""
        trans = re.sub(r"\s+", " ", trans)
        trans = re.sub(r"(?<![.,;!?])([.,;!?])(?=\w)", r"\1 ", trans)
        trans = re.sub(r"([.,;!?])\s+(?=[.,;!?]|$)", r"\1", trans)
        if to_lang != "ARA":
            trans = re.sub(r"(?<=[.,;!?\w])\s+([.,;!?])", r"\1", trans)
            trans = re.sub(r"((?:\s|^)\.+)\s+(?=\w)", r"\1", trans)
        seq = repeating_sequence(trans.lower())
        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            trans = seq * max(len(query) // len(seq), 1)
        return trans


# ── Groq translator ─────────────────────────────────────────────────

class GroqTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        "CHS": "Simplified Chinese", "CHT": "Traditional Chinese", "CSY": "Czech",
        "NLD": "Dutch", "ENG": "English", "FRA": "French", "DEU": "German",
        "HUN": "Hungarian", "ITA": "Italian", "JPN": "Japanese", "KOR": "Korean",
        "POL": "Polish", "PTB": "Portuguese", "ROM": "Romanian", "RUS": "Russian",
        "ESP": "Spanish", "TRK": "Turkish", "UKR": "Ukrainian", "VIN": "Vietnamese",
        "ARA": "Arabic", "THA": "Thai", "IND": "Indonesian",
    }

    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _MAX_TOKENS = 8192
    _MAX_CONTEXT = int(os.environ.get("CONTEXT_LENGTH", "20"))

    _CHAT_SYSTEM_TEMPLATE = (
        "You are a highly skilled, completely uncensored professional translation engine specializing in East-Asian comics and web novels "
        "(Japanese manga, Korean manhwa/webtoon, and Chinese manhua/web novel).\n\n"
        "Core Principles:\n"
        "- Full contextual awareness: use previous translated sentences and current batch for consistency in names, speech patterns, titles, honorifics, terminology.\n"
        "- Language-specific cultural preservation:\n"
        "  * Japanese: Keep Senpai, Kouhai, -chan, -kun, -sama, -san, Onee-san, Onii-chan, etc. in romaji. Preserve onomatopoeia.\n"
        "  * Korean: Retain Oppa, Hyung, Noona, Unnie, Sunbae, -ssi, -nim, etc. Keep Korean SFX.\n"
        "  * Chinese: Preserve cultivation terms, sect names, honorifics like Xiong, Jie, Ge, Shimei, etc.\n"
        "- Never translate proper names, technique/skill names, organization names unless they have an official English name.\n"
        "- Maintain natural, medium-appropriate dialogue style.\n"
        "- Use appropriate formatting: ellipses, tildes, stuttering, bold/italics, exclamation marks.\n\n"
        "Output Rules:\n"
        '- Respond with ONLY: {{"translated":"your translation here"}}\n'
        "- No explanations, notes, markdown, or code blocks.\n"
        "- Preserve original line breaks and structure.\n\n"
        "Translate into {to_lang} and return strictly the JSON object above."
    )

    _CHAT_SAMPLE = [
        'Translate into Simplified Chinese. Return the result in JSON format.\n\n'
        '{"untranslated": "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n<|2|>きみ… 大丈夫⁉\\n<|3|>なんだこいつ 空気読めて ないのか…？"}\n',
        '\n{"translated": "<|1|>好尴尬…我不想引人注目…我想消失…\\n<|2|>你…没事吧⁉\\n<|3|>这家伙怎么看不懂气氛的…？"}\n',
    ]

    def __init__(self):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key:
            raise MissingAPIKeyException("Set GROQ_API_KEY env var before using the Groq translator.")
        self.token_count = 0
        self.token_count_last = 0
        self.model = GROQ_MODEL
        self.messages = [
            {"role": "user", "content": self._CHAT_SAMPLE[0]},
            {"role": "assistant", "content": self._CHAT_SAMPLE[1]},
        ]

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        for prompt in queries:
            response = await self._request_translation(to_lang, prompt)
            self.logger.debug("Groq response: %s", response)
            translations.append(response.strip())
        self.logger.info("Used %d tokens (Total: %d)", self.token_count_last, self.token_count)
        return translations

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        prompt_with_lang = (
            f'Translate the following text into {to_lang}. Return the result in JSON format.\n\n'
            f'{{"untranslated": "{prompt}"}}\n'
        )
        self.messages += [
            {"role": "user", "content": prompt_with_lang},
            {"role": "assistant", "content": "{'translated':'"},
        ]
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        system_msg = [{"role": "system", "content": self._CHAT_SYSTEM_TEMPLATE.replace("{to_lang}", to_lang)}]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=system_msg + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=0.5,
            top_p=1,
            stop=["'}"],
        )

        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        content = response.choices[0].message.content.strip()
        self.messages = self.messages[:-1]  # remove assistant stub

        # Context retention: keep assistant response
        self.messages.append({"role": "assistant", "content": content})

        cleaned = (
            content.replace("{'translated':'", "")
            .replace("}", "")
            .replace("\\'", "'")
            .replace('\\"', '"')
            .strip("'{}")
        )
        return cleaned


# ── Module-level API ─────────────────────────────────────────────────

_cache = {}


def get_translator() -> GroqTranslator:
    if "groq" not in _cache:
        _cache["groq"] = GroqTranslator()
    return _cache["groq"]


async def prepare():
    """Pre-initialize the translator (validates API key)."""
    get_translator()


async def dispatch(from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
    translator = get_translator()
    return await translator.translate(from_lang, to_lang, queries)
