"""Session guard functions built on the core capability policy.

These are not IPU invariants. IPU invariants are Runner-local contracts over
``Principal + OpNode + result``. Session guards validate session registry state
or adapt human-facing resource names to core capability rights before an
operation reaches a side effect.
"""

from lionagi._errors import AccessError, ConfigurationError, ExistsError, NotFoundError
from lionagi.core.policy import covers, policy_check

__all__ = (
    "branch_name_must_be_unique",
    "capabilities_are_granted",
    "capability_must_include",
    "capabilities_must_be_granted",
    "genai_model_must_be_configured",
    "resource_is_accessible",
    "resource_must_be_accessible",
    "resource_must_exist",
    "right_is_granted",
    "right_must_be_granted",
    "scope_is_accessible",
    "scope_in_resources",
    "scope_to_right",
    "scope_must_be_accessible",
)


def right_is_granted(branch, required: str) -> bool:
    """Return whether branch.principal grants the required core right."""
    return policy_check(branch.principal, None, override_reqs={required})


def right_must_be_granted(branch, required: str) -> None:
    """Raise AccessError unless branch.principal grants the required core right."""
    if right_is_granted(branch, required):
        return
    raise AccessError(
        f"Branch '{branch.name}' missing capability: {required}",
        details={
            "requested": required,
            "available": sorted(branch.principal.rights()),
        },
    )


def scope_to_right(scope: str) -> str:
    if scope in {"*", "*:*"}:
        return "service.call"
    if scope.startswith("service.call"):
        return scope
    return f"service.call:{scope}"


def scope_is_accessible(branch, scope: str) -> bool:
    if right_is_granted(branch, scope_to_right(scope)):
        return True
    if ":" not in scope:
        return right_is_granted(branch, scope_to_right(f"{scope}:*"))
    return False


def scope_in_resources(scope: str, resources: set[str]) -> bool:
    """Additive-only wildcard grant check: exact, ``*:*``, or ``svc:*`` satisfies any ``svc:op``."""
    required = scope_to_right(scope)
    granted = {scope_to_right(resource) for resource in resources}
    return any(covers(have, required) for have in granted)


def scope_must_be_accessible(branch, scope: str) -> None:
    if scope_is_accessible(branch, scope):
        return
    raise AccessError(
        f"Branch '{branch.name}' denied scope '{scope}'",
        details={
            "scope": scope,
            "granted": sorted(branch.resources),
        },
    )


def resource_must_exist(session, name: str):
    """Validate resource exists in session. Raise NotFoundError if not."""
    if not session.resources.has(name):
        raise NotFoundError(
            f"Service '{name}' not found in session services",
            details={"available": session.resources.list_names()},
        )


def resource_is_accessible(branch, name: str) -> bool:
    return right_is_granted(branch, scope_to_right(name)) or right_is_granted(
        branch,
        scope_to_right(f"{name}:*"),
    )


def resource_must_be_accessible(branch, name: str) -> None:
    """Bare resource name maps to ``service.call:{name}``; scoped grants like ``service.call:{name}:*`` also satisfy."""
    if resource_is_accessible(branch, name):
        return
    raise AccessError(
        f"Branch '{branch.name}' has no access to resource '{name}'",
        details={
            "branch": branch.name,
            "resource": name,
            "available": sorted(branch.principal.rights()),
        },
    )


def capability_must_include(branch, capability: str) -> None:
    """Validate branch principal includes capability. Raise AccessError if not."""
    right_must_be_granted(branch, capability)


def capabilities_are_granted(branch, capabilities: set[str]) -> bool:
    return all(right_is_granted(branch, cap) for cap in capabilities)


def capabilities_must_be_granted(branch, capabilities: set[str]) -> None:
    """Validate branch has all capabilities. Raise AccessError listing missing."""
    missing = {cap for cap in capabilities if not right_is_granted(branch, cap)}
    if missing:
        raise AccessError(
            f"Branch '{branch.name}' missing capabilities: {missing}",
            details={
                "requested": sorted(capabilities),
                "available": sorted(branch.principal.rights()),
            },
        )


def branch_name_must_be_unique(session, name: str) -> None:
    try:
        session.communications.get_progression(name)
        raise ExistsError(f"Branch with name '{name}' already exists")
    except (KeyError, NotFoundError):
        pass


def genai_model_must_be_configured(session) -> None:
    if session.default_gen_model is None:
        raise ConfigurationError(
            "Session has no default_gen_model configured",
            details={"session_id": str(session.id)},
        )
