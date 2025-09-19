def check_async_postgres_available():
    from lionagi.utils import is_import_installed

    all_import_present = 0
    for pkg in ("sqlalchemy", "asyncpg"):
        if is_import_installed(pkg):
            all_import_present += 1
    if all_import_present == 2:
        return True
    return ImportError(
        "This adapter requires postgres option to be installed. "
        'Please install them using `uv pip install "lionagi[postgres]"`.'
    )
