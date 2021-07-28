from copy import deepcopy
from speechbrain.nnet.context import use_context
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import math
import itertools
import torch
import pytest

def testcontext_basic():
    from speechbrain.nnet.context import Context

    context = Context()
    foo = torch.tensor([1, 2, 3])
    bar = torch.tensor([3, 4])
    context.foo = foo
    context.bar = bar
    assert (context.foo == foo).all()
    assert (context.bar == bar).all()


def test_context_with():
    from speechbrain.nnet.context import context, use_context
    foo = torch.tensor([1, 2, 3])

    with context() as ctx:
        # Context operations are a no-op when no context is activated
        ctx.foo = foo

        # The tensors do not get saved on a context
        with pytest.raises(AttributeError):
            ctx.foo

    # Now that the context has been activated, values should be retained
    with use_context() as root_ctx:
        with context() as ctx:
            ctx.foo = foo
        assert (root_ctx.foo == foo).all()


def test_context_as_dict():
    from speechbrain.nnet.context import Context

    foo = torch.tensor([1, 2, 3])
    bar = torch.tensor([3, 4])
    context = Context()
    context.foo = foo
    context.bar = bar
    items = context.as_dict()
    assert (items["foo"] == foo).all()
    assert (items["bar"] == bar).all()


def test_context_nested():
    from speechbrain.nnet.context import use_context, context

    encoder_out = torch.arange(1, 3)
    decoder_out = torch.arange(1, 4)
    decoder_inner_x = torch.arange(1, 5)
    decoder_inner_y = torch.arange(1, 6)
    with use_context() as root_ctx:
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
        with context(name="decoder") as decoder_ctx:
            decoder_ctx.out = decoder_out
            with context(name="inner") as decoder_inner_ctx:
                decoder_inner_ctx.x = decoder_inner_x
                decoder_inner_ctx.y = decoder_inner_y

        assert (root_ctx.encoder.out == encoder_out).all()
        assert (root_ctx.decoder.out == decoder_out).all()
        assert (root_ctx.decoder.inner.x == decoder_inner_x).all()
        assert (root_ctx.decoder.inner.y == decoder_inner_y).all()


def test_context_nested_multi():
    from speechbrain.nnet.context import use_context, context

    encoder_out = torch.arange(1, 3)
    layer_x = [
        torch.arange(1, 7) + x for x in range(1, 3)]
    layer_y = [
        torch.arange(1, 7) + x for x in range(1, 4)]

    with use_context() as root_ctx:
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
            for y in layer_y:
                with context(name="layer") as layer_ctx:
                    layer_ctx.y = y

        for x in layer_x:
            with context(name="layer") as layer_ctx:
                layer_ctx.x = x

        assert (root_ctx.encoder.out == encoder_out).all()
        for idx, x in enumerate(layer_x):
            assert (root_ctx.layer[idx].x == x).all()
        for idx, y in enumerate(layer_y):
            assert (root_ctx.encoder.layer[idx].y == y).all()


def test_context_nested_as_dict():
    from speechbrain.nnet.context import use_context, context

    x = torch.arange(1, 2)
    encoder_out = torch.arange(1, 3)
    decoder_out = torch.arange(1, 4)
    decoder_inner_x = torch.arange(1, 5)
    decoder_inner_y = torch.arange(1, 6)
    with use_context() as root_ctx:
        root_ctx.x = x
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
        with context(name="decoder") as decoder_ctx:
            decoder_ctx.out = decoder_out
            with context(name="inner") as decoder_inner_ctx:
                decoder_inner_ctx.x = decoder_inner_x
                decoder_inner_ctx.y = decoder_inner_y

    items = root_ctx.as_dict()
    assert (items["x"] == x).all()
    assert (items["encoder.out"] == encoder_out).all()
    assert (items["decoder.out"] == decoder_out).all()
    assert (items["decoder.inner.x"] == decoder_inner_x).all()
    assert (items["decoder.inner.y"] == decoder_inner_y).all()


def test_context_nested_as_dict_nested():
    from speechbrain.nnet.context import use_context, context

    x = torch.arange(1, 2)
    encoder_out = torch.arange(1, 3)
    decoder_out = torch.arange(1, 4)
    decoder_inner_x = torch.arange(1, 5)
    decoder_inner_y = torch.arange(1, 6)
    with use_context() as root_ctx:
        root_ctx.x = x
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
        with context(name="decoder") as decoder_ctx:
            decoder_ctx.out = decoder_out
            with context(name="inner") as decoder_inner_ctx:
                decoder_inner_ctx.x = decoder_inner_x
                decoder_inner_ctx.y = decoder_inner_y

    context_dict = root_ctx.as_dict(nested=True)
    assert (context_dict["items"]["x"] == x).all()
    assert (
        context_dict["children"]["encoder"][0]["items"]["out"]
        == encoder_out).all()
    decoder_dict = context_dict["children"]["decoder"][0]
    assert (decoder_dict["items"]["out"] == decoder_out).all()
    inner_dict = decoder_dict["children"]["inner"][0]
    assert (inner_dict["items"]["x"] == decoder_inner_x).all()
    assert (inner_dict["items"]["y"] == decoder_inner_y).all()


def test_context_from_dict():
    from speechbrain.nnet.context import Context

    x = torch.arange(1, 2)
    encoder_out = torch.arange(1, 3)
    decoder_out = torch.arange(1, 4)
    bogus = [
        torch.arange(1, 5),
        torch.arange(1, 6),
    ]
    context_dict = {
        "items": {"x": x},
        "children": {
            "encoder": [
                {
                    "items": {"out": encoder_out}
                }
            ],
            "bogus_layers": [
                {
                    "items": {"out": value}
                }
                for value in bogus
            ],
            "decoder": [
                {
                    "items": {"out": decoder_out}
                }
            ]
        }
    }
    context = Context.from_dict(context_dict)
    assert (context.x == x).all()
    assert (context.encoder.out == encoder_out).all()
    assert (context.decoder.out == decoder_out).all()
    for idx, value in enumerate(bogus):
        assert (context.bogus_layers[idx].out == value).all()

def test_context_nested_as_dict_multi():
    from speechbrain.nnet.context import use_context, context

    encoder_out = torch.arange(1, 3)
    layer_x = [
        torch.arange(1, 7) + x for x in range(1, 3)]
    layer_y = [
        torch.arange(1, 7) + x for x in range(1, 4)]

    with use_context() as root_ctx:
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
            for y in layer_y:
                with context(name="layer") as layer_ctx:
                    layer_ctx.y = y

        for x in layer_x:
            with context(name="layer") as layer_ctx:
                layer_ctx.x = x

    items = root_ctx.as_dict()
    assert (items["encoder.out"] == encoder_out).all()
    for idx, x in enumerate(layer_x):
        assert (items[f"layer.{idx}.x"] == layer_x[idx]).all()
    for idx, y in enumerate(layer_y):
        assert (items[f"encoder.layer.{idx}.y"] == layer_y[idx]).all()


def test_context_nested_filters():
    from speechbrain.nnet.context import use_context, context

    encoder_out = torch.arange(1, 3)
    decoder_out = torch.arange(1, 4)
    decoder_inner_x = torch.arange(1, 5)
    decoder_inner_y = torch.arange(1, 6)
    with use_context(items=["encoder.out", "decoder.inner.x"]) as root_ctx:
        with context(name="encoder") as encoder_ctx:
            encoder_ctx.out = encoder_out
        with context(name="decoder") as decoder_ctx:
            decoder_ctx.out = decoder_out
            with context(name="inner") as decoder_inner_ctx:
                decoder_inner_ctx.x = decoder_inner_x
                decoder_inner_ctx.y = decoder_inner_y

    assert (root_ctx.encoder.out == encoder_out).all()
    assert (root_ctx.decoder.inner.x == decoder_inner_x).all()
    with pytest.raises(AttributeError):
        root_ctx.decoder.inner.y
    with pytest.raises(AttributeError):
        root_ctx.decoder.out


def scatter_inner(value, chunk_count):
    if value is None:
        result = [None] * chunk_count
    elif isinstance(value, tuple):
        result = tuple(scatter_inner(child, chunk_count) for child in value)
    elif isinstance(value, dict):
        result = {key: scatter_inner(child, chunk_count) for key, child in value.items()}
    elif isinstance(value, torch.Tensor):
        chunk_size = math.ceil(len(value) / chunk_count)
        result = [value[idx:idx+chunk_size] for idx in range(0, len(value), chunk_size)]
    else:
        raise ValueError(value.__class__)
    return result


def pluck(value, idx):
    if isinstance(value, tuple):
        result = tuple(pluck(child, idx) for child in value)
    elif isinstance(value, dict):
        result = {key: pluck(child, idx) for key, child in value.items()}
    else:
        result = value[idx]
    return result


def scatter(value, chunk_count):
    scattered_value = scatter_inner(value, chunk_count)
    return [
        pluck(scattered_value, idx) for idx in range(chunk_count)
    ]


def gather(value):
    if isinstance(value[0], tuple):
        result = tuple(
            gather(
                [child[idx]
                for child in value])
            for idx in range(len(value[0]))
        )

    elif isinstance(value[0], dict):
        result = {
            key: gather([child[key] for child in value])
            for key in value[0]
        }
    elif isinstance(value[0], torch.Tensor):
        result = torch.cat(value)
    elif isinstance(value, list) and all(item is None for item in value):
        result = None
    else:
        result = value
    return result


class FakeDP(nn.Module):
    def __init__(self, module, n_parallel=2):
        super().__init__()
        self.n_parallel = n_parallel
        self.module = module
        self.module_replicas = [
            deepcopy(self.module) for module in range(self.n_parallel)]
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel)

    def forward(self, *args, **kwargs):
        args_scattered = scatter(args, self.n_parallel)
        kwargs_scattered = scatter(kwargs, self.n_parallel)
        results = self.executor.map(
            lambda model, args, kwargs: model(*args, **kwargs),
            self.module_replicas,
            args_scattered,
            kwargs_scattered)
        results_scattered = tuple(result for result in results)
        return gather(results_scattered)


def test_context_parallel():
    from speechbrain.nnet.context import context
    from speechbrain.nnet.context import data_parallel_with_context

    class FakeModule(nn.Module):
        def forward(self, x):
            with context() as ctx:
                ctx.foo = x + 1
            return x * 2

    with use_context() as ctx:
        module = FakeModule()
        module = data_parallel_with_context(FakeDP, module)

        x = torch.tensor([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ])
        y = module(x)
        y_ref = torch.tensor([
            [2, 4],
            [6, 8],
            [10, 12],
            [14, 16]])
        assert torch.all(y == y_ref)
        foo_ref = torch.tensor([
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
        assert torch.all(ctx.foo == foo_ref)
