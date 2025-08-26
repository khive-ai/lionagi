def check_networkx_available():
    try:
        from networkx import DiGraph  # noqa: F401

        return True
    except Exception:
        return ImportError(
            "The 'networkx' package is required for this feature. "
            "Please install `networkx` or `'lionagi[graph]'`."
        )


def check_matplotlib_available():
    try:
        import matplotlib.pyplot as plt

        return True
    except Exception:
        return ImportError(
            "The 'matplotlib' package is required for this feature. "
            "Please install `matplotlib` or `'lionagi[graph]'`."
        )
