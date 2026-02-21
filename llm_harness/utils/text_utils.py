from functools import cache

from opencc import OpenCC


@cache
def _s2hk_converter() -> OpenCC:
    return OpenCC("s2hk")


def s2hk(content: str) -> str:
    """Convert simplified Chinese to traditional Chinese (Hong Kong standard).

    Args:
        content (str): Text in simplified Chinese.

    Returns:
        str: Text converted to traditional Chinese (Hong Kong standard).
    """
    return _s2hk_converter().convert(content)
