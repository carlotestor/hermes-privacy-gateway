"""Presidio-backed text pseudonymization and local alias rehydration."""

from __future__ import annotations

import hashlib
import re
import sys
import threading
import types
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

# Presidio is imported lazily in PrivacyEngine so that importing this module
# (which plugin registration does for the constants/helpers below) never
# depends on it. If Presidio is missing, engine creation fails inside the
# middleware and requests are blocked, instead of register() raising and the
# plugin silently not registering at all.
if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

if __package__:
    from .vault import AliasVault, alias_ranges
else:  # direct local test/import outside Hermes plugin loader
    from vault import AliasVault, alias_ranges


BLOCKED_REQUEST = "[PRIVACY_GATEWAY_BLOCKED_REQUEST]"
BLOCKED_TOOL_RESULT = "[PRIVACY_GATEWAY_BLOCKED_TOOL_RESULT]"
NON_TEXT_BLOCK = "[PRIVACY_GATEWAY_BLOCKED_NON_TEXT_CONTENT]"

# Conservative structured-secret patterns. These are deliberately narrow;
# domain-specific secrets should be added as Presidio recognizers or local rules.
_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_AUTH_HEADER = re.compile(
    r"(?im)^authorization\s*:\s*(?:bearer|basic)\s+\S+\s*$"
)

_NON_TEXT_TYPES = {
    "image_url",
    "input_image",
    "image",
    "input_audio",
    "audio",
    "input_file",
    "file",
}
_NON_TEXT_KEYS = {
    "image_url",
    "file_data",
    "b64_json",
    "input_audio",
    "audio_url",
}

# Provider request objects mix model-visible text with protocol metadata. Only
# these known fields are text/payload boundaries; unknown scalar fields (IDs,
# enum values, model names, item references, etc.) must remain byte-for-byte
# unchanged or the provider may reject an otherwise valid request.
_MODEL_TEXT_KEYS = {
    "content",
    "description",
    "input_text",
    "instructions",
    "output_text",
    "prompt",
    "refusal",
    "summary",
    "text",
}
_MODEL_PAYLOAD_KEYS = {
    "args",
    "arguments",
    "input",
    "output",
}


def walk_strings(value: Any, string_fn) -> Any:
    """Recursively transform strings while preserving container structure."""
    if isinstance(value, str):
        return string_fn(value)
    if isinstance(value, list):
        return [walk_strings(item, string_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(walk_strings(item, string_fn) for item in value)
    if isinstance(value, dict):
        return {key: walk_strings(item, string_fn) for key, item in value.items()}
    return value


def walk_model_text(value: Any, string_fn, *, mode: str = "content") -> Any:
    """Transform model-visible strings without rewriting protocol metadata.

    ``content`` accepts a bare string or a list of content/message items.
    ``scan`` searches nested protocol containers but preserves unknown scalar
    fields. ``payload`` sanitizes every string value inside tool arguments or
    outputs, where arbitrary user-defined object keys are expected.
    """
    if isinstance(value, str):
        return string_fn(value) if mode in {"content", "payload"} else value
    if isinstance(value, list):
        child_mode = "payload" if mode == "payload" else mode
        return [walk_model_text(item, string_fn, mode=child_mode) for item in value]
    if isinstance(value, tuple):
        child_mode = "payload" if mode == "payload" else mode
        return tuple(walk_model_text(item, string_fn, mode=child_mode) for item in value)
    if isinstance(value, dict):
        if mode == "payload":
            return {
                key: walk_model_text(item, string_fn, mode="payload")
                for key, item in value.items()
            }

        transformed = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _MODEL_TEXT_KEYS:
                item_mode = "content"
            elif normalized_key in _MODEL_PAYLOAD_KEYS:
                item_mode = "payload"
            else:
                item_mode = "scan"
            transformed[key] = walk_model_text(item, string_fn, mode=item_mode)
        return transformed
    return value


def contains_non_text_model_content(value: Any) -> bool:
    """Detect common provider payload shapes carrying image/audio/file content."""
    if isinstance(value, list) or isinstance(value, tuple):
        return any(contains_non_text_model_content(item) for item in value)
    if isinstance(value, dict):
        item_type = str(value.get("type", "")).lower()
        if item_type in _NON_TEXT_TYPES:
            return True
        if any(key in value for key in _NON_TEXT_KEYS):
            return True
        return any(contains_non_text_model_content(item) for item in value.values())
    return False


def blocked_content_for_key(key: str) -> Any:
    """Return a conservative provider-compatible replacement for common text roots."""
    if key == "messages":
        return [{"role": "user", "content": BLOCKED_REQUEST}]
    return BLOCKED_REQUEST


def select_non_overlapping_results(
    results: Sequence[RecognizerResult],
) -> list[RecognizerResult]:
    """Choose a deterministic, highest-confidence set of entity spans.

    Presidio can return overlapping recognizer results. Prefer confidence first,
    then the longer span, then the earlier span, and return results in source
    order for callers that need stable output.
    """
    selected: list[RecognizerResult] = []
    for result in sorted(
        results,
        key=lambda item: (
            -float(getattr(item, "score", 0.0)),
            -(int(item.end) - int(item.start)),
            int(item.start),
            int(item.end),
            str(getattr(item, "entity_type", "")),
        ),
    ):
        # Presidio normally returns few spans, so this clear O(n²) check keeps
        # overlap policy explicit and is preferable to a more fragile sweep.
        if any(result.start < other.end and result.end > other.start for other in selected):
            continue
        selected.append(result)
    return sorted(selected, key=lambda item: (int(item.start), int(item.end)))


DEFAULT_SANITIZE_CACHE_CHARS = 16_000_000

# Held in sys.modules so every loaded copy of this plugin (one per Hermes
# profile in a multi-profile gateway) shares one analyzer and one NLP lock.
_SHARED_STATE_NAME = "_hermes_privacy_gateway_shared"


def _shared_state() -> Any:
    state = sys.modules.get(_SHARED_STATE_NAME)
    if state is None:
        fresh: Any = types.ModuleType(_SHARED_STATE_NAME)
        fresh.lock = threading.Lock()
        fresh.analyzers = {}
        state = sys.modules.setdefault(_SHARED_STATE_NAME, fresh)
    return state


class _SanitizeCache:
    """Bounded LRU of sanitize() results keyed by a digest of the input text."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(0, int(max_chars))
        self._items: OrderedDict[bytes, str] = OrderedDict()
        self._chars = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(text: str) -> bytes:
        return hashlib.blake2b(
            text.encode("utf-8", "surrogatepass"), digest_size=16
        ).digest()

    def get(self, key: bytes) -> str | None:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: bytes, value: str) -> None:
        cost = len(value)
        if cost > self.max_chars:
            return
        with self._lock:
            if key in self._items:
                return
            self._items[key] = value
            self._chars += cost
            while self._chars > self.max_chars:
                _, evicted = self._items.popitem(last=False)
                self._chars -= len(evicted)


NLP_ENGINES = ("spacy", "none")


def _build_analyzer(
    nlp_config_path: Path,
    language: str,
    enable_langextract: bool,
    langextract_config_path: str,
    nlp_engine_name: str = "spacy",
):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    if nlp_engine_name == "none":
        # Pattern/checksum recognizers only: no model load, no NER entities.
        from presidio_analyzer.nlp_engine import NoOpNlpEngine

        nlp_engine = NoOpNlpEngine(models=[{"lang_code": language, "model_name": "none"}])
        nlp_engine.load()
        return _finish_analyzer(
            AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[language]),
            enable_langextract,
            langextract_config_path,
        )

    provider = NlpEngineProvider(conf_file=str(nlp_config_path))
    # Presidio's spaCy engine silently runs `spacy.cli.download` (an
    # unpinned pip install into the Hermes runtime) for a missing model.
    # Refuse instead; the model is a documented manual install.
    if provider.nlp_configuration.get("nlp_engine_name") == "spacy":
        import spacy.util

        for model in provider.nlp_configuration.get("models") or []:
            name = str(model.get("model_name", ""))
            if not (spacy.util.is_package(name) or Path(name).exists()):
                raise RuntimeError(
                    f"spaCy model {name!r} is not installed; install it as "
                    "described in docs/INSTALL.md (the plugin will not "
                    "download it at runtime)"
                )
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=[language],
    )
    return _finish_analyzer(analyzer, enable_langextract, langextract_config_path)


def _finish_analyzer(analyzer, enable_langextract: bool, langextract_config_path: str):
    if enable_langextract:
        from presidio_analyzer.predefined_recognizers.third_party.basic_langextract_recognizer import (
            BasicLangExtractRecognizer,
        )

        if langextract_config_path:
            recognizer = BasicLangExtractRecognizer(
                config_path=langextract_config_path
            )
        else:
            recognizer = BasicLangExtractRecognizer()
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


def _shared_analyzer(
    nlp_config_path: Path,
    language: str,
    enable_langextract: bool,
    langextract_config_path: str,
    nlp_engine_name: str = "spacy",
):
    """Return (analyzer, analysis_lock), building the analyzer once per process."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(nlp_config_path)):
        digest.update(path.read_bytes() if path.exists() else b"-")
        digest.update(b"\0")
    key = (
        digest.hexdigest(),
        language,
        bool(enable_langextract),
        langextract_config_path,
        nlp_engine_name,
    )

    state = _shared_state()
    with state.lock:
        entry = state.analyzers.get(key)
        if entry is None:
            analyzer = _build_analyzer(
                nlp_config_path,
                language,
                enable_langextract,
                langextract_config_path,
                nlp_engine_name,
            )
            entry = (analyzer, threading.Lock())
            state.analyzers[key] = entry
    return entry


class PrivacyEngine:
    """Local detector + stable pseudonymizer."""

    def __init__(
        self,
        nlp_config_path: Path,
        *,
        language: str = "en",
        score_threshold: float = 0.55,
        enable_langextract: bool = False,
        langextract_config_path: str = "",
        vault: AliasVault | None = None,
        sanitize_cache_chars: int = DEFAULT_SANITIZE_CACHE_CHARS,
        nlp_engine: str = "spacy",
    ) -> None:
        nlp_engine = str(nlp_engine or "spacy").strip().lower()
        if nlp_engine not in NLP_ENGINES:
            raise ValueError(f"nlp_engine must be one of {NLP_ENGINES}, got {nlp_engine!r}")
        self.language = language
        self.score_threshold = score_threshold
        self.vault = vault or AliasVault()
        # Aliases depend on the vault key, so the cache is per engine.
        self._cache = _SanitizeCache(sanitize_cache_chars)
        # spaCy pipelines are not thread-safe and each concurrent forward pass
        # allocates its own activations; analyze one text at a time.
        self.analyzer, self._analysis_lock = _shared_analyzer(
            nlp_config_path,
            language,
            enable_langextract,
            langextract_config_path,
            nlp_engine,
        )

    @staticmethod
    def remove_high_confidence_secrets(text: str) -> str:
        text = _PEM_PRIVATE_KEY.sub("[SECRET_REMOVED]", text)
        return _AUTH_HEADER.sub("Authorization: [SECRET_REMOVED]", text)

    def sanitize(self, text: str) -> str:
        """Replace detected PII with stable aliases; never alias secrets."""
        if not text:
            return text

        # Conversation history is re-sent on every request; reuse prior results.
        cache_key = self._cache.key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        sanitized = self._sanitize_uncached(text)
        self._cache.put(cache_key, sanitized)
        if sanitized != text:
            # Sanitized tool results come back verbatim in later history; they are already final.
            self._cache.put(self._cache.key(sanitized), sanitized)
        return sanitized

    def _sanitize_uncached(self, text: str) -> str:
        text = self.remove_high_confidence_secrets(text)
        protected = alias_ranges(text)
        with self._analysis_lock:
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                score_threshold=self.score_threshold,
            )

        # Do not feed already-issued aliases back through Presidio.
        if protected:
            results = [
                result
                for result in results
                if not any(
                    result.start < end and result.end > start
                    for start, end in protected
                )
            ]

        if not results:
            return text

        results = select_non_overlapping_results(results)

        # Replace from right to left so Presidio's character offsets remain
        # valid while each detected span is converted to its stable alias.
        sanitized = text
        for result in sorted(results, key=lambda item: item.start, reverse=True):
            value = text[result.start : result.end]
            alias = self.vault.alias(result.entity_type, value)
            sanitized = sanitized[: result.start] + alias + sanitized[result.end :]
        return sanitized

    def sanitize_tree(self, value: Any) -> Any:
        return walk_strings(value, self.sanitize)

    def sanitize_model_content(self, value: Any) -> Any:
        """Sanitize provider text while preserving protocol IDs and enums."""
        return walk_model_text(value, self.sanitize)

    def rehydrate(self, text: str) -> str:
        return self.vault.rehydrate(text)

    def rehydrate_tree(self, value: Any) -> Any:
        return walk_strings(value, self.rehydrate)
