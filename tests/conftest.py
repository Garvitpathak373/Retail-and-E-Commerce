"""Pytest shared test fixtures and environment isolation."""

import pytest
from llm_advisor import circuit_breaker, clear_query_cache


@pytest.fixture(autouse=True)
def reset_advisor_global_state():
    """Ensure every test executes with an isolated, clean circuit breaker and cache state."""
    circuit_breaker.reset()
    clear_query_cache()
    yield
    circuit_breaker.reset()
    clear_query_cache()
