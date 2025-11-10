import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # must include scheme
MODEL_NAME  = os.getenv("OLLAMA_MODEL", "llama3")                 # e.g., "llama3", "qwen2.5:0.5b", etc.
TIMEOUT_SEC = float(os.getenv("TRANSLATOR_TIMEOUT_SEC", "8"))

_TRANSLATE_SYS = (
    "You are a translator. Translate the input into natural, idiomatic English. "
    "Preserve meaning and tone. Keep numbers, code, and URLs unchanged. "
    "Reply with English text only—no preface, labels, or explanations."
)

_LANG_SYS = (
    "You are a language classifier. Detect the language of the input text and reply "
    "only with the English name of that language on a single line. No punctuation, "
    "no explanations, no ISO codes.\n\n"
    "Example:\nINPUT: Bonjour, je m'appelle Bob\nOUTPUT: French\n"
    "INPUT: Können Sie mir bitte helfen?\nOUTPUT: German"
)

def _ollama_chat(system_msg: str, user_msg: str) -> str | None:
    """Call Ollama /api/chat and return assistant content or None."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        # Expected: {"message": {"content": "..."}}
        msg = (data.get("message") or {}).get("content")
        return msg.strip() if isinstance(msg, str) else None
    except requests.RequestException:
        return None

def get_translation(text: str) -> str | None:
    return _ollama_chat(_TRANSLATE_SYS, f"Text:\n{text}")

def get_language(text: str) -> str | None:
    return _ollama_chat(_LANG_SYS, f"Text:\n{text}")

def _looks_english(s: str, min_ratio: float = 0.6) -> bool:
    if not s:
        return False
    ascii_letters_spaces = sum(ch.isascii() and (ch.isalpha() or ch.isspace()) for ch in s)
    return (ascii_letters_spaces / max(1, len(s))) >= min_ratio


def translate_content(content: str) -> tuple[bool, str]:
    """
    Return (is_english, translated_content).
    Policy for this assignment/tests:
      - On classifier gibberish/denylist → (False, "MODEL FAILED")
      - On translation gibberish/empty/unchanged/non-English-looking → (False, "MODEL FAILED")
      - On English → (True, original)
      - Otherwise → (False, translated)
    """
    content = (content or "").strip()
    if not content:
        return True, content

    denylist = {
        "i don't understand your request", "i dont understand your request",
        "unknown", "n/a", "error", "cannot translate", "unable to comply",
    }

    # 1) language detection
    lang = (get_language(content) or "").strip().lower()

    if lang.startswith("english"):
        return True, content

    # If classifier gives gibberish / denylisted tokens → FAIL per tests
    if lang in denylist:
        return False, "MODEL FAILED"

    # Optional heuristic shortcut (keep if you want; does not affect the failing cases)
    if not lang and _looks_english(content):
        return True, content

    # 2) translation
    translated = (get_translation(content) or "").strip()

    # Reject bad translator outputs → FAIL per tests
    if not translated:
        return False, "MODEL FAILED"
    if translated.casefold() == content.casefold():
        return False, "MODEL FAILED"
    if translated.casefold() in denylist:
        return False, "MODEL FAILED"
    if not _looks_english(translated):
        return False, "MODEL FAILED"

    # Looks good
    return False, translated

