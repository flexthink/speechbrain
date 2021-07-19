"""
An implementation of a flexible context, making it posisble
to seamleslly give access to internal representation within a model

Authors
 * Artem Ploujnikov 2021
"""

import threading
import torch
from contextlib import contextmanager

_ctx = threading.local()
_ctx.current = None
_ctx.stack = []

@torch.jit.ignore
class Context:
    """
    A dynamic context object. Contexts can be used to provide access
    to intermediate representations within a model for specific scenarios
    that need them without having to change the model's output, such as
    attention matrices, the output of intermediate layers within deep networks, etc.

    The functionality is inspired by frameworks based on computational graphs
    that provide access to named tensors.

    In most cases, a context object should not be instantiated directly by
    user code.

    Arguments
    ---------
    name: str
        the name of this context (applicable to models with multiple nested
        modules)

    items: list[str]
        a list of names of items that will be registered by this context

    """
    __slots__ = [
        "name",
        "_items",
        "_filter_items",
        "_children",
        "_child_filter_items"]
    current = None
    dummy = None

    def __init__(
        self,
        name=None,
        items=None,
        raw_filter_items=None,
        raw_child_filter_items=None
    ):
        self.name = name
        if items is not None:
            self._filter_items, self._child_filter_items = self._parse_filter_items(items)
        else:
            self._filter_items = raw_filter_items
            self._child_filter_items = raw_child_filter_items or {}
        self._items = {}
        self._children = {}

    def _parse_filter_items(self, items):
        child_items = {item for item in items if "." in item}
        local_items = set(items) - child_items
        return local_items, self._parse_child_filter_items(child_items)

    def _parse_child_filter_items(self, items):
        item_components = [item.split(".") for item in items]
        result = {}
        for item in item_components:
            current = result
            for component in item[:-2]:
                if component not in current:
                    current[component] = {}
                current = current[component]
            component = item[-2]
            if component not in current:
                current[component] = set()
            current[component].add(item[-1])
        return result

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        elif self._filter_items is None or name in self._filter_items:
            self._items[name] = value

    def __getattr__(self, name):
        try:
            result = self._items.get(name)
            if result is None:
                result = self._children[name]
                if len(result) == 1:
                    result = result[0]
        except KeyError as e:
            raise AttributeError(e.args[0])
        return result

    def as_dict(self):
        """
        Returns the items stored in this context as a dictionary

        Returns
        -------
        items: dict
            a dictionary of item names and values
        """
        return dict(
            self._items,
            **dict(self._get_child_items()))

    def get_child_filter_items(self, name):
        """
        Retrieves the child filter items for the
        specified child

        Attributes
        ----------
        name: str
            the name of the child context
        """
        return self._child_filter_items.get(name)

    def _get_child_items(self):
        for context_name, contexts in self._children.items():
            if len(contexts) == 1:
                for item_name, value in contexts[0].as_dict().items():
                    yield f"{context_name}.{item_name}", value
            else:
                for idx, context in enumerate(contexts):
                    for item_name, value in context.as_dict().items():
                        yield f"{context_name}.{idx}.{item_name}", value

    @property
    def has_filters(self):
        """
        Determines if this context has filters
        """
        return (self._filter_items is not None or self._child_filter_items is not None)

    def child(self, context):
        """
        Adds a child context to this context

        Arguments
        ---------
        context: Context
            a child contect
        """
        children = self._children.get(context.name)
        if not children:
            children = []
            self._children[context.name] = children
        children.append(context)


_dummy = Context(items=[])

@torch.jit.ignore
@contextmanager
def context(name=None):
    empty = not any(_ctx.stack)
    can_stack = name is not None and not empty
    if can_stack:
        previous_top = _ctx.stack[-1]
        bottom = _ctx.stack[0]
        raw_filter_items= None
        raw_child_filter_items = None
        if bottom.has_filters:
            child_items = previous_top.get_child_filter_items(name)
            if child_items is not None:
                if isinstance(child_items, set):
                    raw_filter_items = child_items
                else:
                    raw_filter_items = set()
                    raw_child_filter_items = child_items
        context = Context(
            name=name,
            raw_filter_items=raw_filter_items,
            raw_child_filter_items=raw_child_filter_items)
        _ctx.stack.append(context)
    yield _dummy if empty else _ctx.stack[-1]
    if can_stack:
        _ctx.stack.pop()
        _ctx.stack[-1].child(context)


@torch.jit.ignore
@contextmanager
def use_context(items=None):
    """
    Creates a top-level context. This function is meant to be run at the
    top level, when a model is being created (as opposed to inside a model).
    Calls to use_model cannot be nested
    """
    if any(_ctx.stack):
        raise ValueError("A context is already open")

    context = Context(items=items)
    _ctx.stack.append(context)
    yield context
    _ctx.stack[:] = []

