from .resnet import get_resnet


_backbones={
    'resnet': get_resnet,
    'resnext': get_resnet,
}


def get_backbone(name, **kwargs):
    name = name.lower()
    assert name in _backbones.keys(), f'Backbone {name} is Not implemented!'
    return _backbones[name](**kwargs)
