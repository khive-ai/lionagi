import pytest

from lionagi.session.session import Session


def _required_model_right(params, **_):
    return {f"service.call:{params['model']}"}


async def _handler(params, ctx):
    return {"value": params["value"]}


@pytest.mark.asyncio
async def test_operation_dynamic_rights_run_through_runner_policy():
    session = Session()
    session.register_operation(
        "dynamic_model",
        _handler,
        required_rights=_required_model_right,
    )
    denied = session.create_branch(name="denied")
    allowed = session.create_branch(name="allowed", resources={"model-a"})

    failed = await session.conduct(
        "dynamic_model",
        branch=denied,
        params={"model": "model-a", "value": 1},
    )
    succeeded = await session.conduct(
        "dynamic_model",
        branch=allowed,
        params={"model": "model-a", "value": 2},
    )

    assert failed.execution.error is not None
    assert "Policy denied" in str(failed.execution.error)
    assert succeeded.execution.error is None
    assert succeeded.response == {"value": 2}
