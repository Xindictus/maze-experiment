from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


class Singleton(type):
    """_summary_

    Args:
        type (_type_): _description_

    Returns:
        _type_: _description_
    """

    _instances: Dict[Type[T], T] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]
