from speechbrain.nnet.context import use_context
import torch
import pytest

def test_context_basic():
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



