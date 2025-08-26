def check_docling_available():
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401

        return True
    except Exception:
        return ImportError(
            "The 'docling' package is required for this feature. "
            "Please install it via 'pip install lionagi[reader]'."
        )
