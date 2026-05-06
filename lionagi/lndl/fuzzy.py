# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL Fuzzy Parser — typo-tolerant parsing using rapidfuzz backend."""

import logging
import re

from lionagi.ln.fuzzy import SimilarityAlgo, string_similarity
from lionagi.ln.types import Operable, Spec

from .errors import AmbiguousMatchError, MissingFieldError, MissingOutBlockError
from .resolver import resolve_references_prefixed
from .types import LactMetadata, LNDLOutput, LvarMetadata, RLvarMetadata

__all__ = ("normalize_lndl_text", "parse_lndl_fuzzy")

logger = logging.getLogger(__name__)

_NOTE_NS = "note"
_XML_ATTR_RE = re.compile(r'\b\w+=["\'][^"\']*["\']')


def normalize_lndl_text(text: str) -> str:
    """Normalize model-invented syntax before lexing.

    Handles:
    - Curly-brace tags: {lact X}fn(){/lact} or {lact X}fn()</lact> → <lact X>fn()</lact>
    - XML attributes: <lact name="X" type="Y"> → <lact X>
    - Capitalized Note namespace: <lvar Note.X> → <lvar note.X>
    """
    text = re.sub(r"\{(lvar|lact)(\s+[^}]*)\}", r"<\1\2>", text)
    text = re.sub(r"\{/(lvar|lact)\}", r"</\1>", text)

    def _clean_tag(m: re.Match) -> str:
        tag = m.group(1)
        body = m.group(2)

        attrs = dict(re.findall(r'(\w+)=["\']([^"\']*)["\']', body))
        cleaned = _XML_ATTR_RE.sub("", body).strip()

        parts = cleaned.split() if cleaned else []
        name_val = attrs.get("name", "")
        if name_val and name_val not in " ".join(parts):
            parts.append(name_val)

        tag_body = " ".join(parts)
        return f"<{tag} {tag_body}>" if tag_body else f"<{tag}>"

    text = re.sub(r"<(lvar|lact)\s+((?:[^>])*?)>", _clean_tag, text)
    text = re.sub(r"<(lvar|lact)\s+Note\.", r"<\1 note.", text)
    return text


def _correct_name(
    target: str,
    candidates: list[str],
    threshold: float,
    context: str = "name",
) -> str:
    if target in candidates:
        return target

    if threshold >= 1.0:
        raise MissingFieldError(
            f"{context.capitalize()} '{target}' not found. "
            f"Available: {candidates} (strict mode: exact match required)"
        )

    result = string_similarity(
        word=target,
        correct_words=candidates,
        algorithm=SimilarityAlgo.JARO_WINKLER,
        threshold=threshold,
        return_most_similar=False,
    )

    if not result:
        raise MissingFieldError(
            f"{context.capitalize()} '{target}' not found above threshold {threshold}. "
            f"Available: {candidates}"
        )

    algo_func = SimilarityAlgo.jaro_winkler_similarity
    scores = {candidate: algo_func(target, candidate) for candidate in result}

    max_score = max(scores.values())
    ties = [k for k, v in scores.items() if abs(v - max_score) < 0.05]

    if len(ties) > 1:
        scores_str = ", ".join(f"'{k}': {scores[k]:.3f}" for k in ties)
        raise AmbiguousMatchError(
            f"Ambiguous match for {context} '{target}': [{scores_str}]. "
            f"Multiple candidates scored within 0.05. Be more specific."
        )

    match = max(scores.items(), key=lambda kv: kv[1])[0]

    if match != target:
        logger.debug(f"Fuzzy corrected {context}: '{target}' → '{match}'")

    return match


def parse_lndl_fuzzy(
    response: str,
    operable: Operable,
    /,
    *,
    capabilities: list[str] | None = None,
    threshold: float = 0.85,
    threshold_field: float | None = None,
    threshold_lvar: float | None = None,
    threshold_model: float | None = None,
    threshold_spec: float | None = None,
    scratchpad: object | None = None,
) -> LNDLOutput:
    threshold_field = threshold_field if threshold_field is not None else threshold
    threshold_lvar = threshold_lvar if threshold_lvar is not None else threshold
    threshold_model = (
        threshold_model if threshold_model is not None else max(threshold, 0.90)
    )
    threshold_spec = threshold_spec if threshold_spec is not None else threshold

    if capabilities is not None:
        operable.check_allowed(*capabilities)
        active_specs = operable.get_specs(include=set(capabilities))
        active_spec_names = set(capabilities)
        filtered_operable = Operable(active_specs, name=operable.name)
    else:
        active_specs = operable.get_specs()
        active_spec_names = operable.allowed()
        filtered_operable = operable

    from .ast import Lvar
    from .lexer import Lexer
    from .parser import Parser

    response = normalize_lndl_text(response)
    lexer = Lexer(response)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=response)
    program = parser.parse()

    lvars_raw: dict[str, LvarMetadata | RLvarMetadata] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars_raw[lvar.alias] = LvarMetadata(
                model=lvar.model,
                field=lvar.field,
                local_name=lvar.alias,
                value=lvar.content,
            )
        else:
            lvars_raw[lvar.alias] = RLvarMetadata(
                local_name=lvar.alias,
                value=lvar.content,
            )

    lacts_raw: dict[str, LactMetadata] = {}
    for lact in program.lacts:
        lacts_raw[lact.alias] = LactMetadata(
            model=lact.model,
            field=lact.field,
            local_name=lact.alias,
            call=lact.call,
        )

    if not program.out_block:
        raise MissingOutBlockError("No OUT{} block found in response")

    out_fields_raw = program.out_block.fields

    spec_map: dict[str, tuple] = {}
    field_based_specs: dict[str, Spec] = {}

    for spec in active_specs:
        base_type = spec.base_type
        is_pydantic_model = hasattr(base_type, "model_fields")
        if is_pydantic_model:
            spec_map[base_type.__name__] = (spec, True)
            if spec.name and spec.name not in spec_map:
                spec_map[spec.name] = (spec, True)
        else:
            if spec.name:
                field_based_specs[spec.name] = spec
            if operable.name and operable.name not in spec_map:
                spec_map[operable.name] = (spec, False)
                response_name = f"{operable.name}Response"
                if response_name not in spec_map:
                    spec_map[response_name] = (spec, False)

    expected_models = set(spec_map.keys())

    if threshold >= 1.0:
        for lvar in lvars_raw.values():
            if isinstance(lvar, RLvarMetadata):
                continue
            if lvar.model == _NOTE_NS:
                continue
            if lvar.model not in expected_models:
                raise MissingFieldError(
                    f"Model '{lvar.model}' not found. Available: {list(expected_models)} (strict mode)"
                )

        for lvar in lvars_raw.values():
            if isinstance(lvar, RLvarMetadata):
                continue
            if lvar.model == _NOTE_NS:
                continue
            spec, is_model_based = spec_map[lvar.model]
            if is_model_based:
                expected_fields = list(spec.base_type.model_fields.keys())
            else:
                expected_fields = list(active_spec_names)
            if lvar.field not in expected_fields:
                raise MissingFieldError(
                    f"Field '{lvar.field}' not found in model {lvar.model}. Available: {expected_fields} (strict mode)"
                )

        for lact in lacts_raw.values():
            if lact.model and lact.model != _NOTE_NS:
                if lact.model not in expected_models:
                    raise MissingFieldError(
                        f"Action model '{lact.model}' not found. Available: {list(expected_models)} (strict mode)"
                    )
                spec, is_model_based = spec_map[lact.model]
                if is_model_based:
                    expected_fields = list(spec.base_type.model_fields.keys())
                else:
                    expected_fields = list(active_spec_names)
                if lact.field not in expected_fields:
                    raise MissingFieldError(
                        f"Action field '{lact.field}' not found in model {lact.model}. Available: {expected_fields} (strict mode)"
                    )

        for spec_name in out_fields_raw:
            if spec_name not in list(active_spec_names):
                raise MissingFieldError(
                    f"Spec '{spec_name}' not found. Available: {list(active_spec_names)} (strict mode)"
                )

        return resolve_references_prefixed(
            out_fields_raw,
            lvars_raw,
            lacts_raw,
            filtered_operable,
            scratchpad=scratchpad,
        )

    raw_model_names = {
        lvar.model
        for lvar in lvars_raw.values()
        if isinstance(lvar, LvarMetadata) and lvar.model != _NOTE_NS
    }
    for lact in lacts_raw.values():
        if lact.model and lact.model != _NOTE_NS:
            raw_model_names.add(lact.model)

    raw_field_names_by_model: dict[str, set[str]] = {}
    for lvar in lvars_raw.values():
        if isinstance(lvar, RLvarMetadata):
            continue
        if lvar.model == _NOTE_NS:
            continue
        if lvar.model not in raw_field_names_by_model:
            raw_field_names_by_model[lvar.model] = set()
        raw_field_names_by_model[lvar.model].add(lvar.field)
    # Also collect field names from lacts
    for lact in lacts_raw.values():
        if not lact.model or lact.model == _NOTE_NS:
            continue
        if lact.model not in raw_field_names_by_model:
            raw_field_names_by_model[lact.model] = set()
        if lact.field:
            raw_field_names_by_model[lact.model].add(lact.field)

    model_corrections: dict[str, str] = {}
    for raw_model in raw_model_names:
        model_corrections[raw_model] = _correct_name(
            raw_model, list(expected_models), threshold_model, "model"
        )

    field_corrections: dict[tuple[str, str], str] = {}
    for raw_model, raw_fields in raw_field_names_by_model.items():
        corrected_model = model_corrections[raw_model]
        spec, is_model_based = spec_map[corrected_model]
        if is_model_based:
            expected_fields = list(spec.base_type.model_fields.keys())
        else:
            expected_fields = list(active_spec_names)
        for raw_field in raw_fields:
            field_corrections[(raw_model, raw_field)] = _correct_name(
                raw_field,
                expected_fields,
                threshold_field,
                f"field (model {corrected_model})",
            )

    lvars_corrected: dict[str, LvarMetadata | RLvarMetadata] = {}
    for local_name, lvar in lvars_raw.items():
        if isinstance(lvar, RLvarMetadata) or lvar.model == _NOTE_NS:
            lvars_corrected[local_name] = lvar
        else:
            lvars_corrected[local_name] = LvarMetadata(
                model=model_corrections.get(lvar.model, lvar.model),
                field=field_corrections.get((lvar.model, lvar.field), lvar.field),
                local_name=lvar.local_name,
                value=lvar.value,
            )

    lacts_corrected: dict[str, LactMetadata] = {}
    for local_name, lact in lacts_raw.items():
        if lact.model and lact.model != _NOTE_NS:
            lacts_corrected[local_name] = LactMetadata(
                model=model_corrections.get(lact.model, lact.model),
                field=field_corrections.get((lact.model, lact.field), lact.field),
                local_name=lact.local_name,
                call=lact.call,
            )
        else:
            lacts_corrected[local_name] = lact

    expected_spec_names = list(active_spec_names)
    out_fields_corrected: dict[str, list[str] | str] = {}
    for raw_spec_name, value in out_fields_raw.items():
        out_fields_corrected[
            _correct_name(raw_spec_name, expected_spec_names, threshold_spec, "spec")
        ] = value

    available_var_or_action_names = list(lvars_corrected.keys()) + list(
        lacts_corrected.keys()
    )
    out_fields_final: dict[str, list[str] | str] = {}
    for spec_name, value in out_fields_corrected.items():
        if isinstance(value, list):
            corrected_refs = []
            for raw_ref in value:
                # note.X refs pass through to the resolver — they read from scratchpad
                if raw_ref.startswith(f"{_NOTE_NS}."):
                    corrected_refs.append(raw_ref)
                else:
                    corrected_refs.append(
                        _correct_name(
                            raw_ref,
                            available_var_or_action_names,
                            threshold_lvar,
                            "variable or action reference",
                        )
                    )
            out_fields_final[spec_name] = corrected_refs
        else:
            out_fields_final[spec_name] = value

    return resolve_references_prefixed(
        out_fields_final,
        lvars_corrected,
        lacts_corrected,
        filtered_operable,
        scratchpad=scratchpad,
    )
