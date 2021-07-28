"""
An implementation of a flexible context, making it posisble
to seamleslly give access to internal representation within a model

Authors
 * Artem Ploujnikov 2021
"""

import threading
import torch
from torch import nn
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

    def as_dict(self, nested=False):
        """
        Returns the items stored in this context as a dictionary

        Arguments
        ---------
        nested: bool
            if set to False (the default), nested contexts will be
            flattened with a dot notation. This is suitable for
            general usage

            Example:
            {"parent.x": tensor([1, 2]),
             "parent.child.x": tensor([1, 2, 3])}

            if set to True, the nesting of contexts will be preserved
            This is suitable for marshalling a nested context.

            Example:
            {"parent": {
                "items": {
                    "x": tensor([1, 2])
                },
                "children": {
                    "child": {
                        "items": {
                            "x": tensor([1, 2, 3])
                        }
                    }
                }
            }}

        Returns
        -------
        items: dict
            a dictionary of item names and values
        """
        return self._as_dict_nested() if nested else self._as_dict_flat()

    def _as_dict_nested(self):
        return {
            "items": self._items,
            "filter_items": self._filter_items,
            "children": {
                key: [child.as_dict(nested=True) for child in value]
                for key, value in self._children.items()
            }
        }

    def _as_dict_flat(self):
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

    @classmethod
    def from_dict(cls, context_dict, name=None):
        """
        Converts a nested context dictionary to a context
        object

        Arguments
        ---------
        context_dict: dict
            A context dictionary (similar to the one produced
            by .as_dict(nested=True))
        name: str
            the name of the context (optional)

        Returns
        -------
        context: Context
            the new context
        """
        context = cls(name=name)
        context.update_from_dict(context_dict, name=name)
        return context

    def update_from_dict(self, context_dict, name=None):
        """
        Update this context from a dictionary

        Arguments
        ---------
        context_dict: dict
            A context dictionary (similar to the one produced
            by .as_dict(nested=True))
        name: str
            the name of the context (optional)

        Returns
        -------
        context: Context
            the new context
        """
        print(">>>  NAME", name)
        self.name = name or context_dict.get("name")
        self._items = context_dict.get("items", {})
        self._filter_items = context_dict.get("filter_items")
        self._child_filter_items = context_dict.get("child_filter_items")
        context_dict_children = context_dict.get("children") or {}
        for child_name, child_dicts in context_dict_children.items():
            for child_dict in child_dicts:
                child_context = Context.from_dict(child_dict, name=child_name)
                self.child(child_context)



_dummy = Context(items=[])


def _ensure_context_store():
    if not hasattr(_ctx, "stack"):
        _ctx.stack = []
    if not hasattr(_ctx, "current"):
        _ctx.current = None


@torch.jit.ignore
@contextmanager
def context(name=None):
    _ensure_context_store()
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
    try:
        yield _dummy if empty else _ctx.stack[-1]
    finally:
        if can_stack:
            _ctx.stack.pop()
            _ctx.stack[-1].child(context)


def root_context():
    """
    The root context (i.e. the one at the *bottom* of the stack)
    """
    _ensure_context_store()
    return _ctx.stack[0] if _ctx.stack else None


def set_context(context):
    """
    Sets the current context to the specified value, replacing any existing context

    Arguments
    ---------
    context: Context
        the new context
    """
    _ensure_context_store()
    _ctx.stack[:] = [context]


@torch.jit.ignore
@contextmanager
def use_context(items=None):
    """
    Creates a top-level context. This function is meant to be run at the
    top level, when a model is being created (as opposed to inside a model).
    Calls to use_model cannot be nested
    """
    _ensure_context_store()
    if any(_ctx.stack):
        raise ValueError("A context is already open")

    context = Context(items=items)
    _ctx.stack.append(context)
    try:
        yield context
    finally:
        _ctx.stack[:] = []


class ParallelContextWrapper(nn.Module):
    """
    A wrapper that makes it possible to use the context
    with data-parallel models

    Arguments
    ---------
    parallel_module: nn.Module
        a module that parallelizes operations (data-parallel or
        distributed data-parallel)
    """
    def __init__(self, parallel_module):
        super().__init__()
        self.parallel_module = parallel_module

    def forward(self, *args, **kwargs):
        """
        A modified forward pass that deconstructs the context and puts
        it on the arguments
        """
        top_ctx = _ctx.stack[-1]
        ctx_dict = top_ctx.as_dict(nested=True)
        result, context_dict = self.parallel_module(*args, _context=ctx_dict, **kwargs)
        if _ctx.stack:
            _ctx.stack[-1].update_from_dict(context_dict)
        else:
            context = Context.from_dict(context_dict)
            set_context(context)
        return result


class ParallelContextUnwrapper(nn.Module):
    """
    An unwrapper correponding to to ParallelContextWrapper

    Arguments
    ---------
    module: torch.nn.Module
        the underlying module
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """
        Unwraps the context, computes the forward pass through the model,
        returns both the model and the context

        Arguments
        ---------
        _context: dict

        """
        context_dict = kwargs.get("_context")
        if context_dict is not None:
            context = Context.from_dict(context_dict)
            set_context(context)
            kwargs = dict(kwargs)
            del kwargs["_context"]
        result = self.module(*args, **kwargs)
        context = root_context().as_dict(nested=True)
        return result, context


def data_parallel_with_context(dp_module, module, **kwargs):
    """
    Enables data-parallel processing with a context by "sandwiching" the DP
    between a wrapper and an unwrapper, which results in the context being
    passed as an argument to the DP module.
    """
    return ParallelContextWrapper(
        parallel_module=dp_module(
            ParallelContextUnwrapper(module=module),
            **kwargs
        )
    )