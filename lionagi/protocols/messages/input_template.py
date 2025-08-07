



















from enum import IntEnum, auto
from typing import Literal

from lionagi.fields.base import Enum


class Input(TypedDict, total=False):
    
    instruction: JsonValue
    """primary instruction"""

    guidance: JsonValue
    """guidance for the instruction, can be a string or a list of strings"""

    context: JsonValue
    """context for the instruction, can be a string or a list of strings"""

    response_format: type[BaseModel]
    """response format as a pydantic model for structured output, can be a BaseModel"""

    images: list
    """a list of base64 encoded images or image URLs to be included"""

    image_detail: Literal["low", "high", "auto"]

    tool_schemas: list[dict]
    """requested tool schemas for the function calling"""
    
    plain_content: str
    """override everything else including lion system template, will use this as the content of the instruction"""


"""
## Output format

LionAGI can process the following ways of providing output to the system:
1. free form text
2. structured output block

### 1. Providing free form text
When asked to provide free form text, the model is free to provide the text without enclosed by a lionagi code block token. 


LN 





declare a variable using LNDL:
...



[LNAQ_function_name^]{parameters dict}[LNAQ]






[JSON^field_name]value[JSON^field_name*]


[LN_JSON, named]

[LN_JSON, named/]




{
    "directive_codes": "[BEGIN_BLOCK, JSON, 2], ",
    ...
    
    
}
[END_BLOCK, JSON, 1]




instruct the ai model to provide the following types of outputs:



### 1. Structured output

1)  which will be parsed by the system and used for further processing.





...


Lionagi system handles text outputs from iModel in a structured way.









## Providing structured output

When asked to provide structured output, 
















        "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLOCKS."
        f" USER's CAREER DEPENDS ON THE SUCCESS OF IT.** \n```json\n{request_fields}\n```"
        "No triple backticks. Escape all quotes and special characters."


"""





class WithText(str, Enum):
    """If the structured output is requested alongside free form text"""

    AT_BEGINNING_ONCE = "at_beginning"
    """Prompt model to provide the structured output section at the beginning of the response,
    Parser will look for the first parsable structured output of correct schema for entire request 
    in the response, and will ignore subsequent parsable structures.
    """

    AT_ENDING_ONCE = "at_ending"
    """Prompt model to provide the structured output section at the end of the response. If multiple 
    parsable structures are found, the last one is used.
    """

    AT_ANYWHERE = "at_anywhere_once"
    
    
    
    AT_ANYWHERE = "at_anywhere_repeatable"




    FREEFORM = "freeform"
    




class HandleRequestFields(str, Enum):
    """How to handle request fields in the instruction content.
    
    requst_fields is a string that represents the desired output format for structured output. That portion
    of the output must be parsable for downstream processing.
    """
    
    # where to put them if structured output is requested along side with free form text

    
    # whether system will recognize multiple outputs
    IS_REPEATABLE = "is_repeatable"
    
    # whether to include free form text
    WITH_TEXT = "with_text"





    # whether the structured output is required or optional
    



    
    
    
    
    



class HandleRequestFields(str, Enum):
    """How to handle request fields in the instruction content.
    
    requst_fields is a string that represents the desired output format for structured output. That portion
    of the output must be parsable for downstream processing.
    """
    
    AS_BEGINNING_EXTRA = "as_beginning_extra"
    """Prompt model to provide the structured output section at the beginning of the response,
    Parser will look for the first parsable structured output of correct schema for entire request 
    in the response, and will ignore subsequent parsable structures.
    """

    AS_ENDING_EXTRA = "as_ending_extra"
    """Similar to AS_BEGINNING_EXTRA, but the structured output is expected at the end of the response.
    If multiple parsable structures are found, the last one is used.
    """

    AS_ONLY = "as_only"
    """As the only required output, parser will """




    AS_REPEATABLE = "as_repeatable"
    
    
    
    WITH_TEXT_ = "with_text_as_extra"
    
    
    
    """can instruct 
    
    
    """

    AS_REQUIRED_EXTRA = "as_required_extra"
    """
    
    required output at the end of its response"""
    
    AS_OPTIONAL_EXTRA = "as_optional_extra"
    """prepare prompt as optional extra output in addition to free form text"""
    
    AS_ONLY_REQUIRED = "as_only_required"
    """prepare prompt"""




def prepare_expected_structure(
    request_fields: str | None,
    handle_request_fields: HandleRequestFields = HandleRequestFields.AS_EXTRA,
):
    
    
    
    prompt = ""
    
    if request_fields is not None:
        
    
    
    
    
    
    
    
    
    
    
    ...





def _handle_request_fields_as_extra(request_fields: str):
    """prepare prompt for additionally requested fields"""

def _handle_request_fields_as_optional(request_fields: str):
    """prepare prompt for additionally requested fields"""
    
def _handle_request_fields_as_required(request_fields: str):
    """prepare prompt for additionally requested fields"""
    

def _handle_request_fields_as_only(request_fields: str):
    """prepare prompt for additionally requested fields"""

def _handle_request_fields_with_free_text(request_fields: str):
    """prepare prompt for additionally requested fields"""
















def prepare_request_response_format(request_fields: dict) -> str:
    """
    Creates a mandated JSON code block for the response
    based on requested fields.

    Args:
        request_fields: Dictionary of fields for the response format.

    Returns:
        str: A string instructing the user to return valid JSON.
    """
    return (
        "**MUST RETURN JSON-PARSEABLE RESPONSE ENCLOSED BY JSON CODE BLOCKS."
        f" USER's CAREER DEPENDS ON THE SUCCESS OF IT.** \n```json\n{request_fields}\n```"
        "No triple backticks. Escape all quotes and special characters."
    ).strip()


def format_image_item(idx: str, detail: str) -> dict[str, Any]:
    """
    Wrap image data in a standard dictionary format.

    Args:
        idx: A base64 image ID or URL reference.
        detail: The image detail level.

    Returns:
        dict: A dictionary describing the image.
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{idx}",
            "detail": detail,
        },
    }


def format_text_item(item: Any) -> str:
    """
    Turn a single item (or dict) into a string. If multiple items,
    combine them line by line.

    Args:
        item: Any item, possibly a list/dict with text data.

    Returns:
        str: Concatenated text lines.
    """
    msg = ""
    item = [item] if not isinstance(item, list) else item
    for j in item:
        if isinstance(j, dict):
            for k, v in j.items():
                if v is not None:
                    msg += f"- {k}: {v} \n\n"
        else:
            if j is not None:
                msg += f"{j}\n"
    return msg


def format_text_content(content: dict) -> str:
    """
    Convert a content dictionary into a minimal textual summary for LLM consumption.

    Emphasizes brevity and clarity:
      - Skips empty or None fields.
      - Bullet-points for lists.
      - Key-value pairs for dicts.
      - Minimal headings for known fields (guidance, instruction, etc.).
    """

    if isinstance(content.get("plain_content"), str):
        return content["plain_content"]

    lines = []
    # We only want minimal headings for certain known fields:
    known_field_order = [
        "guidance",
        "instruction",
        "context",
        "tool_schemas",
        "respond_schema_info",
        "request_response_format",
    ]

    # Render known fields in that order
    for field in known_field_order:
        if field in content:
            val = content[field]
            if _is_not_empty(val):
                if field == "request_response_format":
                    field = "response format"
                elif field == "respond_schema_info":
                    field = "response schema info"
                lines.append(f"\n## {field.upper()}:\n")
                rendered = _render_value(val)
                # Indent or bullet the rendered result if multiline
                # We'll keep it minimal: each line is prefixed with "  ".
                lines.extend(
                    f"  {line}"
                    for line in rendered.split("\n")
                    if line.strip()
                )

    # Join all lines into a single string
    return "\n".join(lines).strip()


def _render_value(val) -> str:
    """
    Render an arbitrary value (scalar, list, dict) in minimal form:
    - Lists become bullet points.
    - Dicts become key-value lines.
    - Strings returned directly.
    """
    if isinstance(val, dict):
        return _render_dict(val)
    elif isinstance(val, list):
        return _render_list(val)
    else:
        return str(val).strip()


def _render_dict(dct: dict) -> str:
    """
    Minimal bullet list for dictionary items:
      key: rendered subvalue
    """
    lines = []
    for k, v in dct.items():
        if not _is_not_empty(v):
            continue
        subrendered = _render_value(v)
        # Indent subrendered if multiline
        sublines = subrendered.split("\n")
        if len(sublines) == 1:
            if sublines[0].startswith("- "):
                lines.append(f"- {k}: {sublines[0][2:]}")
            else:
                lines.append(f"- {k}: {sublines[0]}")
        else:
            lines.append(f"- {k}:")
            for s in sublines:
                lines.append(f"  {s}")
    return "\n".join(lines)


def _render_list(lst: list) -> str:
    """
    Each item in the list gets a bullet. Nested structures are recursed.
    """
    lines = []
    for idx, item in enumerate(lst, 1):
        sub = _render_value(item)
        sublines = sub.split("\n")
        if len(sublines) == 1:
            if sublines[0].startswith("- "):
                lines.append(f"- {sublines[0][2:]}")
            else:
                lines.append(f"- {sublines[0]}")
        else:
            lines.append("-")
            lines.extend(f"  {s}" for s in sublines)
    return "\n".join(lines)


def _is_not_empty(x) -> bool:
    """
    Returns True if x is neither None, nor empty string/list/dict.
    """
    if x is None:
        return False
    if isinstance(x, (list, dict)) and not x:
        return False
    if isinstance(x, str) and not x.strip():
        return False
    return True


def format_image_content(
    text_content: str,
    images: list,
    image_detail: Literal["low", "high", "auto"],
) -> list[dict[str, Any]]:
    """
    Merge textual content with a list of image dictionaries for consumption.

    Args:
        text_content (str): The textual portion
        images (list): A list of base64 or references
        image_detail (Literal["low","high","auto"]): How detailed the images are

    Returns:
        list[dict[str,Any]]: A combined structure of text + image dicts.
    """
    content = [{"type": "text", "text": text_content}]
    content.extend(format_image_item(i, image_detail) for i in images)
    return content


def prepare_instruction_content(
    guidance: str | None = None,
    instruction: str | None = None,
    context: str | dict | list | None = None,
    request_fields: dict | list[str] | None = None,
    plain_content: str | None = None,
    request_model: BaseModel = None,
    images: str | list | None = None,
    image_detail: Literal["low", "high", "auto"] | None = None,
    tool_schemas: dict | None = None,
) -> dict:
    """
    Combine various pieces (instruction, guidance, context, etc.) into
    a single dictionary describing the user's instruction.

    Args:
        guidance (str | None):
            Optional guiding text.
        instruction (str | None):
            Main instruction or command to be executed.
        context (str | dict | list | None):
            Additional context about the environment or previous steps.
        request_fields (dict | list[str] | None):
            If the user requests certain fields in the response.
        plain_content (str | None):
            A raw plain text fallback.
        request_model (BaseModel | None):
            If there's a pydantic model for the request schema.
        images (str | list | None):
            Optional images, base64-coded or references.
        image_detail (str | None):
            The detail level for images ("low", "high", "auto").
        tool_schemas (dict | None):
            Extra data describing available tools.

    Returns:
        dict: The combined instruction content.

    Raises:
        ValueError: If request_fields and request_model are both given.
    """
    if request_fields and request_model:
        raise ValueError(
            "only one of request_fields or request_model can be provided"
        )

    out_ = {"context": []}
    if guidance:
        out_["guidance"] = guidance
    if instruction:
        out_["instruction"] = instruction
    if context:
        if isinstance(context, list):
            out_["context"].extend(context)
        else:
            out_["context"].append(context)
    if images:
        out_["images"] = images if isinstance(images, list) else [images]
        out_["image_detail"] = image_detail or "low"

    if tool_schemas:
        out_["tool_schemas"] = tool_schemas

    if request_model:
        out_["request_model"] = request_model
        request_fields = breakdown_pydantic_annotation(request_model)
        out_["respond_schema_info"] = request_model.model_json_schema()

    if request_fields:
        _fields = request_fields if isinstance(request_fields, dict) else {}
        if not isinstance(request_fields, dict):
            _fields = {i: "..." for i in request_fields}
        out_["request_fields"] = _fields
        out_["request_response_format"] = prepare_request_response_format(
            request_fields=_fields
        )

    if plain_content:
        out_["plain_content"] = plain_content

    # remove keys with None/UNDEFINED
    return {k: v for k, v in out_.items() if v not in [None, UNDEFINED]}

