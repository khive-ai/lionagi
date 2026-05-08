# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from lionagi.ln.types import Operable

from ._parse_function_call import parse_function_call

from .errors import MissingFieldError, MissingOutBlockError, TypeMismatchError
from .parser import parse_value
from .types import ActionCall, LactMetadata, LNDLOutput, LvarMetadata, RLvarMetadata

NOTE_NAMESPACE = "note"


def _is_note_ref(ref: str) -> bool:
    """An OUT{} ref like 'note.draft' addresses the scratchpad."""
    return ref.startswith(f"{NOTE_NAMESPACE}.") and len(ref) > len(NOTE_NAMESPACE) + 1


def _read_note(ref: str, scratchpad: Any) -> str:
    """Resolve a 'note.X' (or 'note.X.Y') ref to its scratchpad value."""
    if scratchpad is None:
        raise ValueError(f"Scratchpad reference '{ref}' but no scratchpad available in this round")
    keys = ref.split(".")[1:]  # drop "note"
    try:
        value = scratchpad[keys[0]] if len(keys) == 1 else scratchpad[tuple(keys)]
    except (KeyError, IndexError, TypeError):
        available = list(scratchpad) if hasattr(scratchpad, "__iter__") else []
        raise ValueError(
            f"Scratchpad miss: '{ref}' not committed in any prior round. "
            f"Available keys: {available}"
        ) from None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        from lionagi.libs.schema.minimal_yaml import minimal_yaml

        return minimal_yaml(value)
    return str(value)


def _normalize_note_lvars(
    lvars: dict[str, LvarMetadata | RLvarMetadata],
) -> dict[str, LvarMetadata | RLvarMetadata]:
    """Convert lvars in the 'note' namespace to RLvarMetadata.

    Round-local handle stays the same (alias). Removing the typed-Pydantic
    metadata lets the existing positional-assignment path treat them as
    plain values when used inside OUT{} arrays.
    """
    out: dict[str, LvarMetadata | RLvarMetadata] = {}
    for alias, meta in lvars.items():
        if isinstance(meta, LvarMetadata) and meta.model == NOTE_NAMESPACE:
            out[alias] = RLvarMetadata(local_name=meta.local_name, value=meta.value)
        else:
            out[alias] = meta
    return out


def resolve_references_prefixed(
    out_fields: dict[str, list[str] | str],
    lvars: dict[str, LvarMetadata | RLvarMetadata],
    lacts: dict[str, LactMetadata],
    operable: Operable,
    scratchpad: Any | None = None,
) -> LNDLOutput:
    lvars = _normalize_note_lvars(lvars)
    lvar_names = set(lvars.keys())
    lact_names = set(lacts.keys())
    collisions = lvar_names & lact_names
    if collisions:
        raise ValueError(
            f"Name collision detected: {collisions} used in both <lvar> and <lact> declarations"
        )

    operable.check_allowed(*out_fields.keys())

    for spec in operable.get_specs():
        is_required = spec.get("required", True)
        if is_required and spec.name not in out_fields:
            raise MissingFieldError(f"Required field '{spec.name}' missing from OUT{{}}")

    validated_fields = {}
    parsed_actions: dict[str, ActionCall] = {}
    errors: list[Exception] = []

    for field_name, value in out_fields.items():
        try:
            spec = operable.get(field_name)
            if spec is None:
                raise ValueError(
                    f"OUT{{}} field '{field_name}' has no corresponding Spec in Operable"
                )

            target_type = spec.base_type
            is_scalar = target_type in (float, str, int, bool)

            if not is_scalar and not (
                isinstance(target_type, type) and issubclass(target_type, BaseModel)
            ):
                import typing

                origin = getattr(target_type, "__origin__", None)
                if origin is typing.Literal or target_type is type(None):
                    is_scalar = True
                    target_type = str

            if is_scalar:
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError(
                            f"Scalar field '{field_name}' cannot use multiple variables, got {value}"
                        )
                    var_name = value[0]

                    if _is_note_ref(var_name):
                        note_value = _read_note(var_name, scratchpad)
                        parsed_value = parse_value(note_value)
                        try:
                            validated_fields[field_name] = target_type(parsed_value)
                        except (ValueError, TypeError):
                            validated_fields[field_name] = parsed_value
                        continue

                    if var_name in lacts:
                        lact_meta = lacts[var_name]
                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call in action '{var_name}' for scalar field '{field_name}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        action_call = ActionCall(
                            name=var_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[var_name] = action_call
                        validated_fields[field_name] = action_call
                        continue

                    if var_name not in lvars:
                        parsed_value = parse_value(var_name)
                        try:
                            validated_fields[field_name] = target_type(parsed_value)
                        except (ValueError, TypeError):
                            validated_fields[field_name] = parsed_value
                        continue

                    lvar_meta = lvars[var_name]
                    parsed_value = (
                        parse_value(lvar_meta.value)
                        if isinstance(lvar_meta.value, str)
                        else lvar_meta.value
                    )
                else:
                    parsed_value = parse_value(value) if isinstance(value, str) else value

                try:
                    validated_value = target_type(parsed_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Failed to convert value for field '{field_name}' to {target_type.__name__}: {e}"
                    ) from e

                validated_fields[field_name] = validated_value

            else:
                if not isinstance(value, list):
                    raise ValueError(
                        f"BaseModel field '{field_name}' requires array syntax, got literal: {value}"
                    )

                var_list = value

                if not isinstance(target_type, type) or not issubclass(target_type, BaseModel):
                    raise TypeError(
                        f"Spec base_type for '{field_name}' must be BaseModel or scalar, got {target_type}"
                    )

                if len(var_list) == 1 and var_list[0] in lacts:
                    action_name = var_list[0]
                    lact_meta = lacts[action_name]

                    if lact_meta.model is None:
                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call in direct action '{action_name}' for field '{field_name}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        action_call = ActionCall(
                            name=action_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[action_name] = action_call
                        validated_fields[field_name] = action_call
                        continue

                kwargs = {}
                for var_name in var_list:
                    if _is_note_ref(var_name):
                        note_value = _read_note(var_name, scratchpad)
                        filled = set(kwargs.keys())
                        model_fields = list(target_type.model_fields.keys())
                        unfilled = [f for f in model_fields if f not in filled]
                        if unfilled:
                            kwargs[unfilled[0]] = parse_value(note_value)
                        continue

                    if var_name in lacts:
                        lact_meta = lacts[var_name]

                        if lact_meta.model is None or lact_meta.field is None:
                            raise ValueError(
                                f"Direct action '{var_name}' cannot be mixed with lvars in BaseModel field '{field_name}'. "
                                f"Use namespaced syntax: <lact {target_type.__name__}.fieldname {var_name}>...</lact>"
                            )

                        if lact_meta.model.lower() != target_type.__name__.lower():
                            raise TypeMismatchError(
                                f"Action '{var_name}' is for model '{lact_meta.model}', "
                                f"but field '{field_name}' expects '{target_type.__name__}'"
                            )

                        try:
                            parsed_call = parse_function_call(lact_meta.call)
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid function call in action '{var_name}' for field '{lact_meta.model}.{lact_meta.field}':\n"
                                f"  Action call: {lact_meta.call}\n"
                                f"  Parse error: {e}"
                            ) from e

                        action_call = ActionCall(
                            name=var_name,
                            function=parsed_call["tool"],
                            arguments=parsed_call["arguments"],
                            raw_call=lact_meta.call,
                        )
                        parsed_actions[var_name] = action_call
                        kwargs[lact_meta.field] = action_call
                        continue

                    if var_name not in lvars:
                        # Treat as inline literal — assign to next unfilled field
                        filled = set(kwargs.keys())
                        model_fields = list(target_type.model_fields.keys())
                        unfilled = [f for f in model_fields if f not in filled]
                        if unfilled:
                            kwargs[unfilled[0]] = parse_value(var_name)
                        continue

                    lvar_meta = lvars[var_name]

                    if isinstance(lvar_meta, RLvarMetadata):
                        filled = set(kwargs.keys())
                        model_fields = list(target_type.model_fields.keys())
                        unfilled = [f for f in model_fields if f not in filled]
                        if unfilled:
                            kwargs[unfilled[0]] = (
                                parse_value(lvar_meta.value)
                                if isinstance(lvar_meta.value, str)
                                else lvar_meta.value
                            )
                        continue

                    if lvar_meta.model.lower() != target_type.__name__.lower():
                        raise TypeMismatchError(
                            f"Variable '{var_name}' is for model '{lvar_meta.model}', "
                            f"but field '{field_name}' expects '{target_type.__name__}'"
                        )

                    kwargs[lvar_meta.field] = (
                        parse_value(lvar_meta.value)
                        if isinstance(lvar_meta.value, str)
                        else lvar_meta.value
                    )

                has_actions = any(isinstance(v, ActionCall) for v in kwargs.values())
                try:
                    if has_actions:
                        instance = target_type.model_construct(**kwargs)
                    else:
                        instance = target_type(**kwargs)
                except PydanticValidationError as e:
                    raise ValueError(
                        f"Failed to construct {target_type.__name__} for field '{field_name}': {e}"
                    ) from e

                validators = spec.get("validator")
                if validators:
                    validators = validators if isinstance(validators, list) else [validators]
                    for validator in validators:
                        if hasattr(validator, "invoke"):
                            instance = validator.invoke(field_name, instance, target_type)
                        else:
                            instance = validator(instance)

                validated_fields[field_name] = instance

        except Exception as e:
            errors.append(e)

    if errors:
        raise ExceptionGroup("LNDL validation failed", errors)

    return LNDLOutput(
        fields=validated_fields,
        lvars=lvars,
        lacts=lacts,
        actions=parsed_actions,
        raw_out_block=str(out_fields),
    )


def parse_lndl(
    response: str,
    operable: Operable,
    scratchpad: Any | None = None,
) -> LNDLOutput:
    from .ast import Lvar
    from .lexer import Lexer
    from .parser import Parser

    lexer = Lexer(response)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=response)
    program = parser.parse()

    lvars_prefixed: dict[str, LvarMetadata | RLvarMetadata] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars_prefixed[lvar.alias] = LvarMetadata(
                model=lvar.model,
                field=lvar.field,
                local_name=lvar.alias,
                value=lvar.content,
            )
        else:
            lvars_prefixed[lvar.alias] = RLvarMetadata(
                local_name=lvar.alias,
                value=lvar.content,
            )

    lacts_prefixed: dict[str, LactMetadata] = {}
    for lact in program.lacts:
        lacts_prefixed[lact.alias] = LactMetadata(
            model=lact.model,
            field=lact.field,
            local_name=lact.alias,
            call=lact.call,
        )

    if not program.out_block:
        raise MissingOutBlockError("No OUT{} block found in response")

    return resolve_references_prefixed(
        program.out_block.fields, lvars_prefixed, lacts_prefixed, operable, scratchpad=scratchpad
    )
