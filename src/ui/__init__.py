"""ui package initializer

This file makes the `src/ui` directory a Python package so imports like
`from ui.diabetes_app import diabetes_ui` will work when the package root is
on sys.path.
"""

__all__ = [
    "app_ui",
    "diabetes_app",
    "heart_disease_app",
]
