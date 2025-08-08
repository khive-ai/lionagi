from pathlib import Path

from lionagi.utils import is_import_installed

_HAS_OPENCV = is_import_installed("cv2")


__all__ = ("read_image_to_base64",)


def read_image_to_base64(image_path: str | Path) -> str:
    if not _HAS_OPENCV:
        raise ModuleNotFoundError(
            "OpenCV is not installed, please install it with `pip install lionagi[unstructured]`"
        )

    import base64

    import cv2

    image_path = str(image_path)
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    file_extension = "." + image_path.split(".")[-1]

    success, buffer = cv2.imencode(file_extension, image)
    if not success:
        raise ValueError(f"Could not encode image to {file_extension} format.")
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image
