from typing import Any, Callable

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


@pytest.mark.parametrize(
    "operation, call",
    [
        pytest.param(
            "invoke_model_with_response_stream",
            lambda client: client.invoke_model_with_response_stream(modelId=_MODEL_ID, body=b"{}"),
            id="invoke_model_with_response_stream",
        ),
        pytest.param(
            "converse_stream",
            lambda client: client.converse_stream(
                modelId=_MODEL_ID, messages=[{"role": "user", "content": [{"text": "hi"}]}]
            ),
            id="converse_stream",
        ),
    ],
)
def test_sync_stream_call_error_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    operation: str,
    call: Callable[[Any], Any],
) -> None:
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )

    with Stubber(client) as stubber:
        stubber.add_client_error(
            operation, service_error_code="ThrottlingException", http_status_code=429
        )
        with pytest.raises(ClientError):
            call(client)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].events[0].name == "exception"
