from types import SimpleNamespace
import base64
import secrets
import sqlite3
import threading
from pathlib import Path

import pytest

pytest.importorskip("presidio_analyzer")

from privacy import (
    BLOCKED_REQUEST,
    PrivacyEngine,
    _SanitizeCache,
    blocked_content_for_key,
    contains_non_text_model_content,
    select_non_overlapping_results,
    walk_model_text,
    walk_strings,
)
from vault import AliasVault


def test_walk_strings_preserves_structure():
    value = {"a": ["john@example.com", {"b": "John"}], "n": 4}
    out = walk_strings(value, lambda s: f"<{s}>")
    assert out == {"a": ["<john@example.com>", {"b": "<John>"}], "n": 4}


def test_model_text_walk_preserves_protocol_ids_and_sanitizes_content():
    value = [
        {
            "type": "message",
            "id": "person_911469234FCA8DD79AE3CE8306D7ED40",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Alice Example"},
            ],
        },
        {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_456",
            "name": "lookup_customer",
            "arguments": {"customer_id": "Alice Example"},
        },
    ]

    out = walk_model_text(value, lambda text: f"<{text}>")

    assert out[0]["id"] == value[0]["id"]
    assert out[0]["type"] == "message"
    assert out[0]["role"] == "user"
    assert out[0]["content"][0]["type"] == "input_text"
    assert out[0]["content"][0]["text"] == "<Alice Example>"
    assert out[1]["id"] == "fc_123"
    assert out[1]["call_id"] == "call_456"
    assert out[1]["name"] == "lookup_customer"
    assert out[1]["arguments"]["customer_id"] == "<Alice Example>"


def test_model_text_walk_preserves_unknown_scalar_metadata():
    value = {
        "metadata": {
            "opaque_reference": "Alice Example",
            "ids": ["Alice Example", "Bob Example"],
        },
        "content": "Alice Example",
    }

    out = walk_model_text(value, lambda text: f"<{text}>")

    assert out["metadata"] == value["metadata"]
    assert out["content"] == "<Alice Example>"


def test_detects_image_payload():
    payload = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    assert contains_non_text_model_content(payload)


def test_plain_text_payload_not_flagged_as_non_text():
    payload = [{"role": "user", "content": "hello"}]
    assert not contains_non_text_model_content(payload)


def test_blocked_messages_shape():
    assert blocked_content_for_key("messages") == [
        {"role": "user", "content": BLOCKED_REQUEST}
    ]


def test_secret_filter_removes_authorization_header():
    text = "Authorization: Bearer secret-token\nHello"
    out = PrivacyEngine.remove_high_confidence_secrets(text)
    assert "secret-token" not in out
    assert "[SECRET_REMOVED]" in out


def test_overlapping_results_prefer_confidence_then_longer_span():
    results = [
        SimpleNamespace(start=0, end=5, score=0.70),
        SimpleNamespace(start=0, end=11, score=0.70),
        SimpleNamespace(start=20, end=24, score=0.99),
        SimpleNamespace(start=21, end=25, score=0.50),
    ]
    selected = select_non_overlapping_results(results)
    assert [(item.start, item.end) for item in selected] == [(0, 11), (20, 24)]


def test_overlapping_ties_use_entity_type_deterministically():
    results = [
        SimpleNamespace(start=0, end=5, score=0.70, entity_type="ZETA"),
        SimpleNamespace(start=0, end=5, score=0.70, entity_type="ALPHA"),
    ]
    selected = select_non_overlapping_results(results)
    assert [item.entity_type for item in selected] == ["ALPHA"]


def test_sanitize_preserves_offsets_and_existing_aliases():
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    vault = AliasVault(connection=sqlite3.connect(":memory:"), encoded_key=key)
    existing_alias = vault.alias("PERSON", "Alice Example")

    engine = object.__new__(PrivacyEngine)
    engine.language = "en"
    engine.score_threshold = 0.55
    engine.vault = vault
    engine._cache = _SanitizeCache(1_000)
    engine._analysis_lock = threading.Lock()

    class FakeAnalyzer:
        def analyze(self, *, text, language, score_threshold):
            alias_start = text.index(existing_alias)
            email_start = text.index("alice@example.com")
            return [
                SimpleNamespace(
                    start=alias_start,
                    end=alias_start + len(existing_alias),
                    score=0.99,
                    entity_type="PERSON",
                ),
                SimpleNamespace(
                    start=email_start,
                    end=email_start + len("alice@example.com"),
                    score=0.90,
                    entity_type="EMAIL_ADDRESS",
                ),
            ]

    engine.analyzer = FakeAnalyzer()
    source = f"Contact {existing_alias} at alice@example.com."
    sanitized = engine.sanitize(source)

    assert sanitized.startswith(f"Contact {existing_alias} at ")
    assert "alice@example.com" not in sanitized
    assert engine.rehydrate(sanitized) == f"Contact Alice Example at alice@example.com."


def test_missing_spacy_model_is_refused_not_downloaded(monkeypatch):
    import spacy.cli
    import spacy.util
    from pathlib import Path

    def no_download(*args, **kwargs):
        raise AssertionError("spaCy model download attempted at runtime")

    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    monkeypatch.setattr(spacy.cli, "download", no_download)
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    vault = AliasVault(connection=sqlite3.connect(":memory:"), encoded_key=key)

    with pytest.raises(RuntimeError, match="en_core_web_lg"):
        PrivacyEngine(Path(__file__).resolve().parent.parent / "nlp.yaml", vault=vault)


def _engine_with(analyzer, *, cache_chars=10_000):
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    engine = object.__new__(PrivacyEngine)
    engine.language = "en"
    engine.score_threshold = 0.55
    engine.vault = AliasVault(connection=sqlite3.connect(":memory:", check_same_thread=False), encoded_key=key)
    engine._cache = _SanitizeCache(cache_chars)
    engine._analysis_lock = threading.Lock()
    engine.analyzer = analyzer
    return engine


def test_repeated_text_is_analyzed_once():
    calls = []

    class CountingAnalyzer:
        def analyze(self, *, text, language, score_threshold):
            calls.append(text)
            start = text.index("alice@example.com")
            return [SimpleNamespace(start=start, end=start + 17, score=0.9, entity_type="EMAIL_ADDRESS")]

    engine = _engine_with(CountingAnalyzer())
    first = engine.sanitize("mail alice@example.com now")
    second = engine.sanitize("mail alice@example.com now")
    assert first == second
    assert "alice@example.com" not in first
    assert len(calls) == 1


def test_sanitize_cache_is_bounded():
    cache = _SanitizeCache(10)
    for i in range(5):
        cache.put(cache.key(str(i)), "abcd")
    assert cache._chars <= 10
    assert cache.get(cache.key("0")) is None
    assert cache.get(cache.key("4")) == "abcd"
    cache.put(cache.key("big"), "x" * 11)
    assert cache.get(cache.key("big")) is None


def test_analysis_is_serialized_across_threads():
    state = {"active": 0, "peak": 0}
    guard = threading.Lock()
    gate = threading.Event()

    class SlowAnalyzer:
        def analyze(self, *, text, language, score_threshold):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            gate.wait(0.05)
            with guard:
                state["active"] -= 1
            return []

    engine = _engine_with(SlowAnalyzer())
    threads = [threading.Thread(target=engine.sanitize, args=(f"text {i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert state["peak"] == 1


def test_engines_share_one_analyzer(monkeypatch):
    import privacy

    built = []
    monkeypatch.setitem(privacy.sys.modules, privacy._SHARED_STATE_NAME, None)
    privacy.sys.modules.pop(privacy._SHARED_STATE_NAME)
    monkeypatch.setattr(privacy, "_build_analyzer", lambda *a: built.append(a) or object())
    config = Path(__file__).resolve().parent.parent / "nlp.yaml"
    a = privacy._shared_analyzer(config, "en", False, "")
    b = privacy._shared_analyzer(config, "en", False, "")
    assert a is b
    assert len(built) == 1
