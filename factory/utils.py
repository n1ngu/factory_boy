# Copyright: See the LICENSE file.


import typing as t
import collections
import importlib


T = t.TypeVar("T")


def import_object(module_name: str, attribute_name: str) -> t.Any:
    """Import an object from its absolute path.

    Example:
        >>> import_object('datetime', 'datetime')
        <type 'datetime.datetime'>
    """
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


class log_pprint:
    """Helper for properly printing args / kwargs passed to an object.

    Since it is only used with factory.debug(), the computation is
    performed lazily.
    """
    __slots__ = ['args', 'kwargs']

    def __init__(self, args: t.Sequence[t.Any] = (), kwargs: dict[str, t.Any] | None = None):
        self.args = args
        self.kwargs = kwargs or {}

    def __repr__(self) -> str:
        return repr(str(self))

    def __str__(self) -> str:
        return ', '.join(
            [
                repr(arg) for arg in self.args
            ] + [
                '%s=%s' % (key, repr(value))
                for key, value in self.kwargs.items()
            ]
        )


class ResetableIterator:
    """An iterator wrapper that can be 'reset()' to its start."""
    def __init__(self, iterator: t.Iterator[T], **kwargs: t.Any):
        super().__init__(**kwargs)
        self.iterator = iter(iterator)
        self.past_elements: collections.deque[T] = collections.deque()
        self.next_elements: collections.deque[T] = collections.deque()

    def __iter__(self) -> t.Generator[T, None, None]:
        while True:
            if self.next_elements:
                yield self.next_elements.popleft()
            else:
                try:
                    value = next(self.iterator)
                except StopIteration:
                    break
                else:
                    self.past_elements.append(value)
                    yield value

    def reset(self) -> None:
        self.next_elements.clear()
        self.next_elements.extend(self.past_elements)


class OrderedBase:
    """Marks a class as being ordered.

    Each instance (even from subclasses) will share a global creation counter.
    """

    CREATION_COUNTER_FIELD = '_creation_counter'

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        if type(self) is not OrderedBase:
            self.touch_creation_counter()

    def touch_creation_counter(self) -> None:
        bases = type(self).__mro__
        root = bases[bases.index(OrderedBase) - 1]
        if not hasattr(root, self.CREATION_COUNTER_FIELD):
            setattr(root, self.CREATION_COUNTER_FIELD, 0)
        next_counter = getattr(root, self.CREATION_COUNTER_FIELD)
        setattr(self, self.CREATION_COUNTER_FIELD, next_counter)
        setattr(root, self.CREATION_COUNTER_FIELD, next_counter + 1)


def sort_ordered_objects(items: t.Iterable[T], getter: t.Callable[[T], t.Any] = lambda x: x) -> list[T]:
    """Sort an iterable of OrderedBase instances.

    Args:
        items (iterable): the objects to sort
        getter (callable or None): a function to extract the OrderedBase instance from an object.

    Examples:
        >>> sort_ordered_objects([x, y, z])
        >>> sort_ordered_objects(v.items(), getter=lambda e: e[1])
    """
    return sorted(items, key=lambda x: getattr(getter(x), OrderedBase.CREATION_COUNTER_FIELD, -1))
