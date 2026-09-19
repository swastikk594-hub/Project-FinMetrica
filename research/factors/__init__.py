from . import value, quality, momentum, low_risk

FACTOR_REGISTRY = {
    'value': value,
    'quality': quality,
    'momentum': momentum,
    'low_risk': low_risk
}
