from src.config import MAX_CONTEXT_WORDS

def truncate_context(text):
    words = text.split()
    return " ".join(words[:MAX_CONTEXT_WORDS])