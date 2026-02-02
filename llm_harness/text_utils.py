import opencc


def s2hk(content: str) -> str:
    """Convert simplified Chinese to traditional Chinese (Hong Kong standard).

    Args:
        content (str): Text in simplified Chinese.

    Returns:
        str: Text converted to traditional Chinese (Hong Kong standard).
    """
    converter = opencc.OpenCC("s2hk")
    return converter.convert(content)
