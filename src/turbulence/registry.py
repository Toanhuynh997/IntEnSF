from __future__ import annotations

from typing import Any, Callable, Dict, Type


_REGISTRY: Dict[str, Type[Any]] = {}


def register(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a turbulence model class by name."""

    def _decorator(cls: Type[Any]) -> Type[Any]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise KeyError(f"Duplicate turbulence model registration: {name!r}")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def make(name: str, **kwargs: Any) -> Any:
    """Construct a registered turbulence model by name."""

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown turbulence model {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def registered_names() -> list[str]:
    return sorted(_REGISTRY.keys())

