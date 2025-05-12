import asyncio
import logging

from pydantic import Field

from lionagi.traits import Identifiable, Invokable, Temporal
from lionagi.traits.types import ExecutionStatus
from lionagi.traits.utils import validate_model_to_dict

from .endpoint import Endpoint

logger = logging.getLogger(__name__)


class APICalling(Identifiable, Temporal, Invokable):
    """Represents an API call event, storing payload, headers, and endpoint info.

    This class extends `Event` and provides methods to invoke or stream the
    request asynchronously.
    """

    endpoint: Endpoint = Field(exclude=True)
    cache_control: bool = Field(default=False, exclude=True)
    headers: dict | None = Field(None, exclude=True)

    async def invoke(self) -> None:
        """Invokes the API call, updating the execution state with results.

        Raises:
            Exception: If any error occurs, the status is set to FAILED and
                the error is logged.
        """
        start = asyncio.get_event_loop().time()
        response = None
        e1 = None

        try:
            # Use the endpoint as a context manager
            response = await self.endpoint.call(
                payload=self.request,
                headers=self.headers,
                cache_control=self.cache_control,
            )

        except asyncio.CancelledError as ce:
            e1 = ce
            logger.warning("invoke() canceled by external request.")
            raise
        except Exception as ex:
            e1 = ex

        finally:
            self.execution.duration = asyncio.get_event_loop().time() - start
            if response is None and e1 is not None:
                self.execution.error = str(e1)
                self.execution.status = ExecutionStatus.FAILED
                logger.error(
                    msg=f"API call to {self.endpoint.config.base_url}/{self.endpoint.config.endpoint} failed: {e1}"
                )
            else:
                self.response_obj = response
                self.execution.response = validate_model_to_dict(response)
                self.execution.status = ExecutionStatus.COMPLETED
            self.update_timestamp()
