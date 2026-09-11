# -*- coding: utf-8 -*-
"""XiChen: end-to-end AI weather forecasting, radiance observation operators
and cascade data assimilation (1.0°, inference-only open-source release)."""

__version__ = "1.0.0"

_LAZY = {
    "XiChenForecast": "xichen.models.forecast",
    "XiChenObsOp": "xichen.models.obsoperator",
    "XiChenDA": "xichen.models.da",
    "Solver": "xichen.models.cascade",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "XiChenForecast", "XiChenObsOp", "XiChenDA", "Solver"]
