import pytest
from src.translator import translate_content
import src.translator as tr


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
        #test translation empty
        ("German", "", "Hier ist dein erstes Beispiel."),
        #test translation whitespace only
        ("German", "   ", "สวัสดี"), # Assuming get_language is not None (your example had a bad type mock)
        #identical output, translation failed
        ("German", "Hier ist dein erstes Beispiel.", "Hier ist dein erstes Beispiel."),
        #translation value returned unexpected output
        ("German", "I don't understand your request", "Hier ist dein erstes Beispiel."),
        #unknown language
        ("I don't understand your request", "First example", "Hier ist dein erstes Beispiel."),
        #test translation returns gibberish
        ("French", "🚀✨🎉 汉字测试", "Bonjour."),

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
    if trans_resp is None:
        monkeypatch.setattr(tr, "get_translation", lambda _txt: None)
    else:
        monkeypatch.setattr(tr, "get_translation", lambda _txt: trans_resp)

    is_english, translated = translate_content(input_text)
    assert is_english is False
    assert translated == "MODEL FAILED"

    # English content does not need translation
def test_echo_when_english_success(monkeypatch):
    """
    Happy path: Input is English, language model correctly identifies it,
    and the original text is returned.
    """
    monkeypatch.setattr(tr, "get_language", lambda _txt: "english")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "SHOULD NOT USE")

    original_text = "Hello!"
    is_english, translated = translate_content(original_text)
    assert is_english is True
    assert translated == original_text

#Runtime error Test
def test_exception_is_graceful_must_raise(monkeypatch):
    """
    Fails path: The underlying LLM helper (get_language) raises a RuntimeError.
    Since source code cannot be changed, we must assert that the exception
    is raised and NOT caught by translate_content.
    """
    def raise_runtime_error(_txt):
        raise RuntimeError("LLM exploded during language detection!")

    monkeypatch.setattr(tr, "get_language", raise_runtime_error)
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "SHOULD NOT USE")

    with pytest.raises(RuntimeError) as excinfo:
        translate_content("hola")

    assert "LLM exploded during language detection!" in str(excinfo.value)