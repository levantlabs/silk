"""Provenance / audit envelope mixed into every concrete constraint.

The envelope answers: who emitted this constraint, when, with what
confidence, against what prompt. These fields are what makes the
schema tape-out grade — every layout decision must be replayable from
its inputs years later. See `silk/constraints/DESIGN.md`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


#: Who emitted a constraint: a layout motif, a language model, a person, or an
#: automatic pass.
Source = Literal["motif", "llm", "human", "auto"]
Severity = Literal["hard", "soft"]


def _new_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConstraintEnvelope(BaseModel):
    """Audit fields shared by every constraint.

    Concrete constraint types in ``silk.constraints.types`` inherit this
    class so the envelope is part of the constraint object itself, not a
    separate wrapper. Models forbid unknown fields and re-validate on
    assignment. JSON objects remain flat because the discriminator
    field (``constraint``) and the envelope fields are both flat keys
    on the same JSON object.

    Attributes
    ----------
    id : str
        Unique identifier (a hex UUID by default).
    source : str
        Who emitted the constraint: ``"motif"``, ``"llm"``, ``"human"`` or
        ``"auto"``. Defaults to ``"human"``.
    confidence : float
        Confidence of the emitter, from 0.0 to 1.0 (dimensionless). Defaults
        to 1.0.
    severity : str
        ``"hard"`` or ``"soft"``. Defaults to ``"hard"``; concrete types may
        override the default.
    created_at : datetime.datetime
        When the constraint was created (UTC by default).
    rationale : str, optional
        Why the constraint was emitted.
    prompt_hash : str, optional
        Hash of the prompt that produced it, when a model emitted it.

    Raises
    ------
    ValueError
        On construction or assignment, if a field is invalid, for example
        ``confidence`` outside 0 to 1 or an unknown field name (pydantic's
        ``ValidationError``, a ``ValueError`` subclass).

    Examples
    --------
    >>> from silk.constraints import ConstraintEnvelope
    >>> ConstraintEnvelope(source='llm', confidence=0.8).severity
    'hard'
    """

    id: str = Field(default_factory=_new_id)
    source: Source = "human"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    severity: Severity = "hard"
    created_at: datetime = Field(default_factory=_utc_now)
    rationale: Optional[str] = None
    prompt_hash: Optional[str] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
