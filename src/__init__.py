"""NHL xG source package.

Keep package initialization lightweight and avoid eager submodule imports.
This prevents circular imports during app startup.
"""

__all__ = ["data", "database", "models"]
