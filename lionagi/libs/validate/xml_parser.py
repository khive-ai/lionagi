from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any


def to_xml(
    obj: dict | list | str | int | float | bool | None,
    root_name: str = "root",
) -> str:
    """
    Convert a dictionary into an XML formatted string.

    Rules:
    - A dictionary key becomes an XML tag.
    - If the dictionary value is:
      - A primitive type (str, int, float, bool, None): it becomes the text content of the tag.
      - A list: each element of the list will repeat the same tag.
      - Another dictionary: it is recursively converted to nested XML.
    - root_name sets the top-level XML element name.

    Args:
        obj: The Python object to convert (typically a dictionary).
        root_name: The name of the root XML element.

    Returns:
        A string representing the XML.

    Examples:
        >>> to_xml({"a": 1, "b": {"c": "hello", "d": [10, 20]}}, root_name="data")
        '<data><a>1</a><b><c>hello</c><d>10</d><d>20</d></b></data>'
    """

    def _convert(value: Any, tag_name: str) -> str:
        # If value is a dict, recursively convert its keys
        if isinstance(value, dict):
            inner = "".join(_convert(v, k) for k, v in value.items())
            return f"<{tag_name}>{inner}</{tag_name}>"
        # If value is a list, repeat the same tag for each element
        elif isinstance(value, list):
            return "".join(_convert(item, tag_name) for item in value)
        # If value is a primitive, convert to string and place inside tag
        else:
            text = "" if value is None else str(value)
            # Escape special XML characters if needed (minimal)
            text = (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )
            return f"<{tag_name}>{text}</{tag_name}>"

    # If top-level obj is not a dict, wrap it in one
    if not isinstance(obj, dict):
        obj = {root_name: obj}

    inner_xml = "".join(_convert(v, k) for k, v in obj.items())
    return f"<{root_name}>{inner_xml}</{root_name}>"


class XMLParser:
    def __init__(self, xml_string: str):
        self.xml_string = xml_string.strip()
        self.index = 0

    def parse(self) -> dict[str, Any]:
        """Parse the XML string and return the root element as a dictionary."""
        return self._parse_element()

    def _parse_element(self) -> dict[str, Any]:
        """Parse a single XML element and its children."""
        self._skip_whitespace()
        if self.xml_string[self.index] != "<":
            raise ValueError(
                f"Expected '<', found '{self.xml_string[self.index]}'"
            )

        tag, attributes = self._parse_opening_tag()
        children: dict[str, str | list | dict] = {}
        text = ""

        while self.index < len(self.xml_string):
            self._skip_whitespace()
            if self.xml_string.startswith("</", self.index):
                closing_tag = self._parse_closing_tag()
                if closing_tag != tag:
                    raise ValueError(
                        f"Mismatched tags: '{tag}' and '{closing_tag}'"
                    )
                break
            elif self.xml_string.startswith("<", self.index):
                child = self._parse_element()
                child_tag, child_data = next(iter(child.items()))
                if child_tag in children:
                    if not isinstance(children[child_tag], list):
                        children[child_tag] = [children[child_tag]]
                    children[child_tag].append(child_data)
                else:
                    children[child_tag] = child_data
            else:
                text += self._parse_text()

        result: dict[str, Any] = {}
        if attributes:
            result["@attributes"] = attributes
        if children:
            result.update(children)
        elif text.strip():
            result = text.strip()

        return {tag: result}

    def _parse_opening_tag(self) -> tuple[str, dict[str, str]]:
        """Parse an opening XML tag and its attributes."""
        match = re.match(
            r'<(\w+)((?:\s+\w+="[^"]*")*)\s*/?>',
            self.xml_string[self.index :],  # noqa
        )
        if not match:
            raise ValueError("Invalid opening tag")
        self.index += match.end()
        tag = match.group(1)
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2)))
        return tag, attributes

    def _parse_closing_tag(self) -> str:
        """Parse a closing XML tag."""
        match = re.match(r"</(\w+)>", self.xml_string[self.index :])  # noqa
        if not match:
            raise ValueError("Invalid closing tag")
        self.index += match.end()
        return match.group(1)

    def _parse_text(self) -> str:
        """Parse text content between XML tags."""
        start = self.index
        while (
            self.index < len(self.xml_string)
            and self.xml_string[self.index] != "<"
        ):
            self.index += 1
        return self.xml_string[start : self.index]  # noqa

    def _skip_whitespace(self) -> None:
        """Skip any whitespace characters at the current parsing position."""
        p_ = len(self.xml_string[self.index :])  # noqa
        m_ = len(self.xml_string[self.index :].lstrip())  # noqa

        self.index += p_ - m_


def xml_to_dict(
    xml_string: str,
    /,
    suppress=False,
    remove_root: bool = True,
    root_tag: str = None,
) -> dict[str, Any]:
    """
    Parse an XML string into a nested dictionary structure.

    This function converts an XML string into a dictionary where:
    - Element tags become dictionary keys
    - Text content is assigned directly to the tag key if there are no children
    - Attributes are stored in a '@attributes' key
    - Multiple child elements with the same tag are stored as lists

    Args:
        xml_string: The XML string to parse.

    Returns:
        A dictionary representation of the XML structure.

    Raises:
        ValueError: If the XML is malformed or parsing fails.
    """
    try:
        a = XMLParser(xml_string).parse()
        if remove_root and (root_tag or "root") in a:
            a = a[root_tag or "root"]
        return a
    except ValueError as e:
        if not suppress:
            raise e


def dict_to_xml(data: dict, /, root_tag: str = "root") -> str:
    root = ET.Element(root_tag)

    def convert(dict_obj: dict, parent: Any) -> None:
        for key, val in dict_obj.items():
            if isinstance(val, dict):
                element = ET.SubElement(parent, key)
                convert(dict_obj=val, parent=element)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(object=val)

    convert(dict_obj=data, parent=root)
    return ET.tostring(root, encoding="unicode")
