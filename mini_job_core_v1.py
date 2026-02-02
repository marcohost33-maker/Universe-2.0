"""
MINI JOB APP – DETERMINISTIC CORE v1.1
Single-File Reference Implementation

Eigenschaften:
- deterministisch (replay-fähig)
- auditierbar (Spec-Hash, Override-Versionen)
- canary-ready (hash-based routing)
- EU AI Act kompatibel (Human Oversight, Logging, Explainability)

Änderungen v1.1:
- Echte Immutability durch tuple statt List
- Thread-safe MetricsCollector
- Input-Validierung
- Spec-Validierung (Gewichte-Summe)
- Strukturiertes Logging
- Typ-sichere Enums
- Verbesserte Edge-Case-Behandlung
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("mini_job_core")


# ============================================================
# CONSTANTS
# ============================================================

TOTAL_RULES = 5
WEIGHT_SUM_TOLERANCE = 0.001
BUCKET_MAX = 0xFFFFFFFF


# ============================================================
# ENUMS
# ============================================================

class DecisionType(str, Enum):
    """Typ-sicheres Decision-Enum."""
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"


class RouteType(str, Enum):
    """Routing-Entscheidung für Audit-Trail."""
    STABLE = "stable"
    CANARY = "canary"
    STABLE_FORCED = "stable_forced"


# ============================================================
# UTILITIES
# ============================================================

def sha256(text: str) -> str:
    """Berechnet SHA-256 Hash eines Strings."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_split(key: str, traffic: float) -> bool:
    """
    Deterministischer Canary-Split.

    Args:
        key: Eindeutiger Schlüssel für konsistentes Routing
        traffic: Anteil des Traffics für Canary (0.0-1.0)

    Returns:
        True wenn Canary, False wenn Stable

    Raises:
        ValueError: Bei ungültigem traffic-Wert
    """
    if not 0.0 <= traffic <= 1.0:
        raise ValueError(f"traffic muss zwischen 0.0 und 1.0 liegen, war: {traffic}")

    if traffic == 0.0:
        return False
    if traffic == 1.0:
        return True

    h = hashlib.sha256(key.encode()).hexdigest()
    bucket = int(h[:8], 16) / BUCKET_MAX
    return bucket < traffic


# ============================================================
# DOMAIN MODELS
# ============================================================

@dataclass(frozen=True)
class Applicant:
    """
    Bewerber-Entität (immutable).

    Verwendet tuple statt List für echte Immutability.
    """
    id: str
    skills: tuple[str, ...]
    location_km: float
    availability: bool
    experience_years: int
    companion_required: bool

    def __post_init__(self) -> None:
        """Validiert Eingabewerte."""
        if not self.id:
            raise ValueError("Applicant.id darf nicht leer sein")
        if self.location_km < 0:
            raise ValueError(f"location_km muss >= 0 sein, war: {self.location_km}")
        if self.experience_years < 0:
            raise ValueError(f"experience_years muss >= 0 sein, war: {self.experience_years}")


@dataclass(frozen=True)
class Job:
    """
    Job-Entität (immutable).

    Verwendet tuple statt List für echte Immutability.
    """
    id: str
    required_skills: tuple[str, ...]
    max_distance_km: float
    requires_companion: bool

    def __post_init__(self) -> None:
        """Validiert Eingabewerte."""
        if not self.id:
            raise ValueError("Job.id darf nicht leer sein")
        if self.max_distance_km < 0:
            raise ValueError(f"max_distance_km muss >= 0 sein, war: {self.max_distance_km}")


@dataclass(frozen=True)
class Decision:
    """
    Entscheidungsergebnis (immutable).

    Enthält alle Informationen für Audit und Explainability.
    """
    decision: DecisionType
    score: float
    confidence: float
    reasons: tuple[str, ...]
    explanation: str
    spec_hash: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert Decision für Logging/API."""
        return {
            "decision": self.decision.value,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "explanation": self.explanation,
            "spec_hash": self.spec_hash,
            "timestamp": self.timestamp
        }


# ============================================================
# RULE SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class RuleSpec:
    """
    Validierte Regel-Spezifikation.

    Stellt sicher, dass Gewichte summiert ~1.0 ergeben.
    """
    threshold: float
    min_experience: int
    weight_skills: float
    weight_distance: float
    weight_time: float
    weight_companion: float
    weight_experience: float

    def __post_init__(self) -> None:
        """Validiert Spec-Werte."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold muss zwischen 0.0 und 1.0 liegen, war: {self.threshold}")
        if self.min_experience < 0:
            raise ValueError(f"min_experience muss >= 0 sein, war: {self.min_experience}")

        weight_sum = (
            self.weight_skills +
            self.weight_distance +
            self.weight_time +
            self.weight_companion +
            self.weight_experience
        )
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Gewichte müssen ~1.0 summieren, war: {weight_sum:.4f}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleSpec:
        """Erstellt RuleSpec aus Dictionary."""
        weights = data.get("weights", {})
        return cls(
            threshold=data["threshold"],
            min_experience=data["min_experience"],
            weight_skills=weights.get("skills", 0.0),
            weight_distance=weights.get("distance", 0.0),
            weight_time=weights.get("time", 0.0),
            weight_companion=weights.get("companion", 0.0),
            weight_experience=weights.get("experience", 0.0)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert RuleSpec für Hashing."""
        return {
            "threshold": self.threshold,
            "min_experience": self.min_experience,
            "weights": {
                "skills": self.weight_skills,
                "distance": self.weight_distance,
                "time": self.weight_time,
                "companion": self.weight_companion,
                "experience": self.weight_experience
            }
        }

    @property
    def spec_hash(self) -> str:
        """Berechnet deterministischen Hash der Spezifikation."""
        return sha256(json.dumps(self.to_dict(), sort_keys=True))


# ============================================================
# RULE ENGINE (SPEC-FIRST)
# ============================================================

class RuleEngine:
    """
    Deterministische Regel-Engine.

    KEINE impliziten Gewichte, KEIN Random.
    Alle Entscheidungen sind vollständig nachvollziehbar.
    """

    def __init__(self, spec: RuleSpec) -> None:
        self.spec = spec
        logger.info(
            "RuleEngine initialisiert",
            extra={"spec_hash": spec.spec_hash}
        )

    @property
    def spec_hash(self) -> str:
        """Gibt den Spec-Hash zurück."""
        return self.spec.spec_hash

    def run(self, applicant: Applicant, job: Job) -> Decision:
        """
        Führt deterministische Regel-Evaluation durch.

        Args:
            applicant: Bewerber-Daten
            job: Job-Anforderungen

        Returns:
            Decision mit Score, Confidence und Begründungen
        """
        reasons: list[str] = []
        score = 0.0
        matched = 0

        # Regel 1: Skills
        if set(job.required_skills).issubset(set(applicant.skills)):
            score += self.spec.weight_skills
            matched += 1
            reasons.append("Skills match")

        # Regel 2: Distanz
        if applicant.location_km <= job.max_distance_km:
            score += self.spec.weight_distance
            matched += 1
            reasons.append("Distance acceptable")

        # Regel 3: Verfügbarkeit
        if applicant.availability:
            score += self.spec.weight_time
            matched += 1
            reasons.append("Availability confirmed")

        # Regel 4: Begleitung
        if job.requires_companion == applicant.companion_required:
            score += self.spec.weight_companion
            matched += 1
            reasons.append("Companion requirement fulfilled")

        # Regel 5: Erfahrung
        if applicant.experience_years >= self.spec.min_experience:
            score += self.spec.weight_experience
            matched += 1
            reasons.append("Experience sufficient")

        confidence = round(matched / TOTAL_RULES, 3)
        decision_type = (
            DecisionType.MATCH
            if score >= self.spec.threshold
            else DecisionType.NO_MATCH
        )

        decision = Decision(
            decision=decision_type,
            score=round(score, 3),
            confidence=confidence,
            reasons=tuple(reasons),
            explanation=f"Rule-based deterministic decision ({matched}/{TOTAL_RULES} rules matched)",
            spec_hash=self.spec_hash
        )

        logger.debug(
            "Decision computed",
            extra={
                "applicant_id": applicant.id,
                "job_id": job.id,
                "decision": decision_type.value,
                "score": score
            }
        )

        return decision


# ============================================================
# ADMIN OVERRIDE (HUMAN OVERSIGHT)
# ============================================================

@dataclass
class OverrideState:
    """Aktueller Override-Zustand."""
    version: int = 0
    force_fallback: bool = False
    disable_canary: bool = False
    emergency_stop: bool = False
    timestamp: float | None = None
    operator: str | None = None
    reason: str | None = None


class AdminOverride:
    """
    Human-in-the-loop Override.

    Jede Änderung ist versioniert und auditierbar.
    Thread-safe durch Lock.
    """

    VALID_FLAGS = {"force_fallback", "disable_canary", "emergency_stop"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = OverrideState()
        self._history: list[dict[str, Any]] = []
        logger.info("AdminOverride initialisiert")

    def set(self, flag: str, value: bool, operator: str, reason: str) -> int:
        """
        Setzt ein Override-Flag.

        Args:
            flag: Name des Flags (force_fallback, disable_canary, emergency_stop)
            value: Neuer Wert
            operator: ID des Operators
            reason: Begründung für Audit

        Returns:
            Neue Versionsnummer

        Raises:
            ValueError: Bei ungültigem Flag-Namen
        """
        if flag not in self.VALID_FLAGS:
            raise ValueError(
                f"Ungültiges Flag: {flag}. Erlaubt: {self.VALID_FLAGS}"
            )

        with self._lock:
            self._state.version += 1
            setattr(self._state, flag, value)
            self._state.timestamp = time.time()
            self._state.operator = operator
            self._state.reason = reason

            history_entry = {
                "version": self._state.version,
                "flag": flag,
                "value": value,
                "timestamp": self._state.timestamp,
                "operator": operator,
                "reason": reason
            }
            self._history.append(history_entry)

            logger.warning(
                f"Override gesetzt: {flag}={value}",
                extra=history_entry
            )

            return self._state.version

    def is_active(self, flag: str) -> bool:
        """Prüft ob ein Flag aktiv ist (thread-safe)."""
        with self._lock:
            return bool(getattr(self._state, flag, False))

    def get_state(self) -> dict[str, Any]:
        """Gibt aktuellen Zustand zurück (thread-safe, Kopie)."""
        with self._lock:
            return {
                "version": self._state.version,
                "force_fallback": self._state.force_fallback,
                "disable_canary": self._state.disable_canary,
                "emergency_stop": self._state.emergency_stop,
                "timestamp": self._state.timestamp,
                "operator": self._state.operator,
                "reason": self._state.reason
            }

    def get_history(self) -> list[dict[str, Any]]:
        """Gibt Änderungshistorie zurück (thread-safe, Kopie)."""
        with self._lock:
            return list(self._history)


# ============================================================
# METRICS (THREAD-SAFE)
# ============================================================

class MetricsCollector:
    """
    Thread-safe Metrics Collector.

    Sammelt strukturierte Metriken für Monitoring und Audit.
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._max_records = max_records
        self._dropped_count = 0

    def record(self, name: str, value: Any, meta: dict[str, Any] | None = None) -> None:
        """
        Zeichnet eine Metrik auf.

        Args:
            name: Metrik-Name
            value: Metrik-Wert
            meta: Zusätzliche Metadaten
        """
        entry = {
            "timestamp": time.time(),
            "metric": name,
            "value": value,
            "meta": meta or {}
        }

        with self._lock:
            if len(self._records) >= self._max_records:
                self._records.pop(0)
                self._dropped_count += 1
            self._records.append(entry)

    def dump(self) -> dict[str, Any]:
        """Gibt alle Metriken zurück (thread-safe, Kopie)."""
        with self._lock:
            return {
                "count": len(self._records),
                "dropped": self._dropped_count,
                "records": list(self._records)
            }

    def clear(self) -> int:
        """Löscht alle Metriken und gibt Anzahl zurück."""
        with self._lock:
            count = len(self._records)
            self._records.clear()
            logger.info(f"MetricsCollector cleared: {count} records")
            return count


# ============================================================
# CANARY RUNNER
# ============================================================

class CanaryRunner:
    """
    Deterministischer Canary-Controller mit Audit-Logging.

    Ermöglicht sicheres A/B-Testing von Regel-Änderungen.
    """

    def __init__(
        self,
        stable: RuleEngine,
        canary: RuleEngine,
        traffic: float = 0.1
    ) -> None:
        if not 0.0 <= traffic <= 1.0:
            raise ValueError(f"traffic muss zwischen 0.0 und 1.0 liegen, war: {traffic}")

        self.stable = stable
        self.canary = canary
        self.traffic = traffic
        self.override = AdminOverride()
        self.metrics = MetricsCollector()

        logger.info(
            "CanaryRunner initialisiert",
            extra={
                "stable_hash": stable.spec_hash,
                "canary_hash": canary.spec_hash,
                "traffic": traffic
            }
        )

    def route(
        self,
        applicant: Applicant,
        job: Job,
        user_id: str | None = None
    ) -> Decision:
        """
        Routet Anfrage zu Stable oder Canary Engine.

        Args:
            applicant: Bewerber-Daten
            job: Job-Anforderungen
            user_id: Optionale User-ID für konsistentes Routing

        Returns:
            Decision vom ausgewählten Engine

        Raises:
            RuntimeError: Bei aktivem Emergency Stop
        """
        if self.override.is_active("emergency_stop"):
            logger.critical("Emergency stop aktiv - Anfrage abgelehnt")
            raise RuntimeError("System stopped by admin override")

        # Routing-Entscheidung
        if self.override.is_active("disable_canary"):
            engine = self.stable
            route_type = RouteType.STABLE_FORCED
        else:
            key = user_id if user_id else f"{applicant.id}:{job.id}"
            use_canary = deterministic_split(key, self.traffic)
            engine = self.canary if use_canary else self.stable
            route_type = RouteType.CANARY if use_canary else RouteType.STABLE

        # Decision berechnen
        decision = engine.run(applicant, job)

        # Metrik aufzeichnen
        self.metrics.record(
            "routing",
            1,
            {
                "route": route_type.value,
                "user_id": user_id,
                "applicant_id": applicant.id,
                "job_id": job.id,
                "spec_hash": engine.spec_hash,
                "decision": decision.decision.value,
                "score": decision.score,
                "confidence": decision.confidence
            }
        )

        return decision

    def get_status(self) -> dict[str, Any]:
        """Gibt aktuellen System-Status zurück."""
        return {
            "stable_spec_hash": self.stable.spec_hash,
            "canary_spec_hash": self.canary.spec_hash,
            "canary_traffic": self.traffic,
            "override_state": self.override.get_state(),
            "metrics_count": self.metrics.dump()["count"]
        }


# ============================================================
# DEFAULT RULE SPEC
# ============================================================

DEFAULT_RULE_SPEC = RuleSpec(
    threshold=0.6,
    min_experience=1,
    weight_skills=0.4,
    weight_distance=0.2,
    weight_time=0.2,
    weight_companion=0.1,
    weight_experience=0.1
)


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_applicant(
    id: str,
    skills: list[str],
    location_km: float,
    availability: bool,
    experience_years: int,
    companion_required: bool
) -> Applicant:
    """Factory für Applicant mit List→Tuple Konvertierung."""
    return Applicant(
        id=id,
        skills=tuple(skills),
        location_km=location_km,
        availability=availability,
        experience_years=experience_years,
        companion_required=companion_required
    )


def create_job(
    id: str,
    required_skills: list[str],
    max_distance_km: float,
    requires_companion: bool
) -> Job:
    """Factory für Job mit List→Tuple Konvertierung."""
    return Job(
        id=id,
        required_skills=tuple(required_skills),
        max_distance_km=max_distance_km,
        requires_companion=requires_companion
    )


# ============================================================
# DEMO EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MINI JOB APP – DETERMINISTIC CORE v1.1 DEMO")
    print("=" * 60)

    # Engines erstellen
    stable_engine = RuleEngine(DEFAULT_RULE_SPEC)
    canary_engine = RuleEngine(DEFAULT_RULE_SPEC)

    runner = CanaryRunner(
        stable=stable_engine,
        canary=canary_engine,
        traffic=0.1
    )

    # Test-Daten
    applicant = create_applicant(
        id="A1",
        skills=["cleaning", "support"],
        location_km=3.0,
        availability=True,
        experience_years=2,
        companion_required=False
    )

    job = create_job(
        id="J1",
        required_skills=["cleaning"],
        max_distance_km=5.0,
        requires_companion=False
    )

    # Decision berechnen
    decision = runner.route(applicant, job, user_id="user-123")

    print("\n--- Decision ---")
    print(json.dumps(decision.to_dict(), indent=2))

    print("\n--- System Status ---")
    print(json.dumps(runner.get_status(), indent=2))

    print("\n--- Metrics ---")
    print(json.dumps(runner.metrics.dump(), indent=2))

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
