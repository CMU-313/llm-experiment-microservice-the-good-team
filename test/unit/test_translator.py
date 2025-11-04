import pytest
from src.translator import translate_content
import src.translator as tr


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message."


def test_llm_normal_response(monkeypatch):
    """
    Happy path:
      - LLM correctly detects non-English language
      - LLM returns a valid English translation
    Expect: (False, <translated text>)
    """
    # Mock the LLM helpers your translate_content uses
    monkeypatch.setattr(tr, "get_language", lambda _txt: "German")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "This is a German message")

    is_english, translated = translate_content("Dies ist eine Nachricht auf Deutsch")
    assert is_english is False
    assert translated == "This is a German message"

    # Also verify the "already English" branch
    monkeypatch.setattr(tr, "get_language", lambda _txt: "English")
    # get_translation shouldn't be called, but keep a default anyway
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "SHOULD NOT USE")

    original = "This is an English message"
    is_english, translated = translate_content(original)
    assert is_english is True
    assert translated == original


@pytest.mark.parametrize(
    "lang_resp, trans_resp, input_text",
    [
        # Classifier denies / unknown → failure
        ("unknown", "ignored", "Dies ist eine Nachricht auf Deutsch"),
        ("n/a", "ignored", "Ceci est un message en français"),
        ("error", "ignored", "これは日本語のメッセージです"),
        # Classifier says non-English but translator returns empty → failure
        ("German", "", "Dies ist eine Nachricht auf Deutsch"),
        # Translator returns unchanged text → failure
        ("French", "Ceci est un message en français", "Ceci est un message en français"),
        # Translator returns denylisted gibberish → failure
        ("Spanish", "i don't understand your request", "Esta es un mensaje en español"),
        # Translator returns non-English-looking output → failure
        ("Korean", "이것은 한국어 메시지입니다", "이것은 한국어 메시지입니다"),
    ],
)
def test_llm_gibberish_response(monkeypatch, lang_resp, trans_resp, input_text):
    """
    Gibberish/malformed paths:
      - language = unknown/denylisted
      - translation empty / unchanged / denylisted / not English-looking
    Expect: (False, "MODEL FAILED")  per your current translate_content policy.
    """
    monkeypatch.setattr(tr, "get_language", lambda _txt: lang_resp)
    monkeypatch.setattr(tr, "get_translation", lambda _txt: trans_resp)

    is_english, translated = translate_content(input_text)
    assert is_english is False
    assert translated == "MODEL FAILED"