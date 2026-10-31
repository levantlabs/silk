"""Tests for silk.constraints.envelope.ConstraintEnvelope."""
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError


def test_envelope_has_seven_named_fields():
    from silk.constraints.envelope import ConstraintEnvelope

    expected = {
        "id", "source", "confidence", "severity",
        "created_at", "rationale", "prompt_hash",
    }
    assert expected.issubset(set(ConstraintEnvelope.model_fields))


def test_envelope_default_values_are_sensible():
    from silk.constraints.envelope import ConstraintEnvelope

    env = ConstraintEnvelope()
    assert isinstance(env.id, str) and len(env.id) >= 16
    assert env.source == "human"
    assert env.confidence == 1.0
    assert env.severity == "hard"
    assert env.rationale is None
    assert env.prompt_hash is None
    assert env.created_at.tzinfo is not None
    # within the last 60s
    delta = datetime.now(timezone.utc) - env.created_at
    assert delta < timedelta(seconds=60)


def test_two_envelopes_get_distinct_ids():
    from silk.constraints.envelope import ConstraintEnvelope

    a, b = ConstraintEnvelope(), ConstraintEnvelope()
    assert a.id != b.id


def test_confidence_must_be_in_unit_interval():
    from silk.constraints.envelope import ConstraintEnvelope

    ConstraintEnvelope(confidence=0.0)
    ConstraintEnvelope(confidence=1.0)
    ConstraintEnvelope(confidence=0.5)
    with pytest.raises(ValidationError):
        ConstraintEnvelope(confidence=-0.01)
    with pytest.raises(ValidationError):
        ConstraintEnvelope(confidence=1.01)


def test_severity_restricted_to_hard_or_soft():
    from silk.constraints.envelope import ConstraintEnvelope

    ConstraintEnvelope(severity="hard")
    ConstraintEnvelope(severity="soft")
    with pytest.raises(ValidationError):
        ConstraintEnvelope(severity="medium")


def test_source_restricted_vocabulary():
    from silk.constraints.envelope import ConstraintEnvelope

    for s in ("motif", "llm", "human", "auto"):
        ConstraintEnvelope(source=s)
    with pytest.raises(ValidationError):
        ConstraintEnvelope(source="oracle")


def test_extra_fields_forbidden():
    from silk.constraints.envelope import ConstraintEnvelope

    with pytest.raises(ValidationError):
        ConstraintEnvelope(unknown_field="x")


def test_envelope_json_round_trip():
    from silk.constraints.envelope import ConstraintEnvelope

    env = ConstraintEnvelope(
        source="llm",
        confidence=0.7,
        severity="soft",
        rationale="LLM thinks these match",
        prompt_hash="deadbeef" * 8,
    )
    data = env.model_dump(mode="json")
    rebuilt = ConstraintEnvelope.model_validate(data)
    assert rebuilt == env


def test_assignment_is_validated():
    from silk.constraints.envelope import ConstraintEnvelope

    env = ConstraintEnvelope()
    with pytest.raises(ValidationError):
        env.confidence = 2.0
