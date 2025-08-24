from lionagi.utils import import_module, is_import_installed

_HAS_PDF2IMAGE = is_import_installed("pdf2image")


def pdf_to_images(
    pdf_path: str, output_folder: str, dpi: int = 300, fmt: str = "jpeg"
) -> list:
    """
    Convert a PDF file into images, one image per page.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Directory to save the output images.
        dpi (int): Dots per inch (resolution) for conversion (default: 300).
        fmt (str): Image format (default: 'jpeg'). Use 'png' if preferred.

    Returns:
        list: A list of file paths for the saved images.
    """
    if not _HAS_PDF2IMAGE:
        raise ModuleNotFoundError(
            "pdf2image is not installed, please install it with `pip install lionagi[unstructured]`"
        )

    import os

    convert_from_path = import_module(
        "pdf2image", import_name="convert_from_path"
    )

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to a list of PIL Image objects
    images = convert_from_path(pdf_path, dpi=dpi)

    saved_paths = []
    for i, image in enumerate(images):
        # Construct the output file name
        image_file = os.path.join(output_folder, f"page_{i + 1}.{fmt}")
        image.save(image_file, fmt.upper())
        saved_paths.append(image_file)

    return saved_paths
