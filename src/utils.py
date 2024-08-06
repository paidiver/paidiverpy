import logging
import sys


def initialise_logging(verbose=False):
    """Initialise logging configuration."""
    logging_level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        stream=sys.stdout,
        format=(
            "☁ paidiverpy ☁  | %(levelname)10s | "
            "%(asctime)s | %(message)s"
        ),
        level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

# class ClassInstanceMethod:
#     def __init__(self, func):
#         self.func = func

#     def __get__(self, instance, cls):
#         if instance is not None:
#             return lambda *args, **kwargs: self.func(instance, *args, **kwargs)
#         return lambda *args, **kwargs: self.func(cls, *args, **kwargs)
