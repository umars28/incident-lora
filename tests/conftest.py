from pathlib import Path

import pytest

from incident_lora.scenarios import load_actions, load_scenarios

FIXTURES = Path(__file__).parent / "fixtures" / "incidents"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURES


@pytest.fixture
def scenarios():
    return load_scenarios(FIXTURES)


@pytest.fixture
def actions():
    return load_actions(FIXTURES)
