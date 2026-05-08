"""Shared pytest fixtures for lnclite integration tests."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from openai_embeddings_model import ModelSettings

from lnclite import DocumentCreate, Lnclite, get_openai_embeddings_model

NEGATIVE_TESTING_DOC = """# Negative Testing and Error Path Validation


Negative testing validates that the system behaves correctly when
given invalid inputs, unexpected conditions, or malicious attempts.
Financial systems must reject invalid operations gracefully.

**Invalid Input:** We test with malformed data, wrong types, out-of-range
values, and boundary violations. The system must reject invalid input
with clear error messages and without corruption or crash.

**Exception Handling:** We simulate failures: database unavailable,
network timeout, external service down. The system should handle
these without data loss, and with appropriate user feedback and
logging. Recovery procedures should be testable.

**Security Negative Tests:** We test authentication failures, expired
sessions, and authorization bypass attempts. The system must deny
unauthorized access and log attempts.
"""


CI_AUTOMATION_DOC = """# CI Integration for Test Automation


Automated tests run as part of our CI pipeline. Every commit triggers
a subset of tests; full regression runs on merge to main. CI integration
ensures fast feedback and prevents regressions from reaching production.

**Pipeline Stages:** Unit tests run first and fastest. Integration tests
follow. API and UI tests may run in parallel or on a separate stage.
Deployment to staging triggers end-to-end smoke tests.

**Failure Handling:** A failed test blocks the pipeline. Developers
receive notifications with failure details and logs. Flaky tests are
quarantined and fixed; we do not allow known flaky tests to block
merges repeatedly.

**Reporting:** CI produces test reports (JUnit, Allure, or similar)
that are archived and linked to builds. We track trends: pass rates,
execution time, and failure categories over time.
"""


UAT_PROCESS_DOC = """# User Acceptance Testing (UAT) Process


UAT validates that the system meets business requirements from the
user's perspective. Business stakeholders execute scenarios in a
staging environment before production release.

**UAT Planning:** UAT scenarios are derived from acceptance criteria
and user stories. Scenarios represent real-world workflows: happy
paths and key exception flows. We schedule UAT with sufficient
time for execution and defect resolution.

**Environment and Data:** UAT runs in a staging environment that
mirrors production as closely as possible. Test data is prepared
to support the scenarios. Access and credentials are provided
to UAT participants.

**Sign-Off:** UAT completion requires all scenarios passed or
waived. Business sign-off is recorded. Defects found in UAT
are triaged; critical issues block release until fixed.
"""


@pytest.fixture
def tempfile_dir(tmp_path):
    return tmp_path


@pytest.fixture
def lnclite_client(tempfile_dir):
    return Lnclite(
        lancedb_path=tempfile_dir / "lancedb",
        openai_embeddings_model=get_openai_embeddings_model(
            openai_client=AsyncOpenAI(),
        ),
        model_settings=ModelSettings(dimensions=1536),
        token_secret_key="test-secret",
    )


@pytest.fixture
def bulk_lnclite_client(tmp_path):
    return Lnclite(
        lancedb_path=tmp_path / "bulk.lance",
        openai_embeddings_model=SimpleNamespace(
            _max_input_tokens=8192,
            model="test-embeddings",
        ),
        model_settings=ModelSettings(dimensions=4),
        token_secret_key="test-secret",
    )


@pytest_asyncio.fixture
async def seeded_lnclite(lnclite_client):
    await lnclite_client.documents.batch_create(
        [
            DocumentCreate(
                content=NEGATIVE_TESTING_DOC,
                tags=["qa", "negative", "security"],
            ),
            DocumentCreate(
                content=CI_AUTOMATION_DOC,
                tags=["qa", "automation", "ci"],
            ),
            DocumentCreate(
                content=UAT_PROCESS_DOC,
                tags=["qa", "manual", "release"],
            ),
        ]
    )
    return lnclite_client
