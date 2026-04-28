# YOLOv5 utilities

def emojis(s):
    """Return string with emojis stripped for terminal compatibility."""
    raise NotImplementedError


def threaded(func):
    """Decorator to run function in a background thread."""
    raise NotImplementedError


class TryExcept:
    """Context manager to catch exceptions and optionally print them."""
    def __init__(self, msg=''):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, value, tb):
        raise NotImplementedError
