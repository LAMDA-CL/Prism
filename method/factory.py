# method/factory.py
"""
Factory for continual-learning integrations.

Loads implementations from ``method/custom/*/integration.py`` via registration.
"""
from typing import Any, Dict, Optional, Set
import importlib
from pathlib import Path

from .base.integration import CLIntegration


class CLMethodFactory:
    """
    Builds ``CLIntegration`` instances from method names / aliases.
    """

    # name / alias -> integration class
    _method_registry: Dict[str, type] = {}
    _discovered_modules: Set[str] = set()
    _discovery_done: bool = False

    @classmethod
    def register(cls, *method_names: str):
        """
        Decorator to register a CL integration.

        Usage: ``@CLMethodFactory.register("sp", "alias")``
        """
        def decorator(integration_class: type):
            if not method_names:
                raise ValueError("register() requires at least one method_name")
            for method_name in method_names:
                cls._method_registry[method_name.lower()] = integration_class
            return integration_class
        return decorator

    @classmethod
    def _discover_method_modules(cls) -> None:
        """Import ``method/custom/*/integration.py`` so ``@register`` decorators run."""
        if cls._discovery_done:
            return

        custom_root = Path(__file__).parent / "custom"
        if not custom_root.is_dir():
            cls._discovery_done = True
            return

        for child in custom_root.iterdir():
            if not child.is_dir() or child.name.startswith("_") or child.name == "__pycache__":
                continue
            if not (child / "integration.py").exists():
                continue
            module_name = f"method.custom.{child.name}.integration"
            if module_name in cls._discovered_modules:
                continue
            importlib.import_module(module_name)
            cls._discovered_modules.add(module_name)

        cls._discovery_done = True

    @classmethod
    def get_available_methods(cls) -> list:
        cls._discover_method_modules()
        return list(cls._method_registry.keys())

    @classmethod
    def create_integration(
        cls,
        method_name: str,
        config: Dict[str, Any],
    ) -> CLIntegration:
        method_name = method_name.lower()
        cls._discover_method_modules()

        if method_name not in cls._method_registry:
            available = cls.get_available_methods()
            raise ValueError(
                f"Unknown method: {method_name}\nAvailable: {available}"
            )

        integration_class = cls._method_registry[method_name]
        return integration_class(config)

    @classmethod
    def load_method_config(cls, method_name: str) -> Dict[str, Any]:
        try:
            mod = importlib.import_module(f"config.methods.{method_name.lower()}")
            cfg = getattr(mod, "METHOD_CONFIG", None)
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass

        config_dir = Path(__file__).parent.parent / "config" / "methods"
        config_file = config_dir / f"{method_name.lower()}.yaml"
        if config_file.exists():
            import yaml

            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config or {}

        raise FileNotFoundError(
            f"Method config not found: {config_dir / (method_name.lower() + '.py')}"
        )

    @classmethod
    def create_from_config_file(
        cls,
        method_name: str,
        override_config: Optional[Any] = None,
    ) -> CLIntegration:
        config = cls.load_method_config(method_name)

        if override_config:
            if isinstance(override_config, dict):
                config.update(override_config)
            else:
                config.update({k: v for k, v in vars(override_config).items() if v is not None})

        return cls.create_integration(method_name, config)
