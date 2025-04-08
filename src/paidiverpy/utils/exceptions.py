"""Exception classes."""


def raise_value_error(message: str) -> None:
    """Raise a ValueError with the given message.

    Args:
        message (str): The message to raise the ValueError with.
    """
    raise ValueError(message)
