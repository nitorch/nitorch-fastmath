from typing import TypeVar, Union, Tuple, Sequence

T = TypeVar('T')
OneOrTwo = Union[T, Tuple[T, T]]
OneOrSeveral = Union[T, Sequence[T]]