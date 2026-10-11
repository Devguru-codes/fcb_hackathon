"""Versioned, serialisable defaults for the deterministic TempoLens decoder."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DecoderConfig:
    """Parameters intentionally expressed in football units where possible.

    These are conservative defaults, not parameters tuned to the five-minute public answer.
    `team_switch_cost` is larger than the dead-ball transition costs because a team hand-over
    needs stronger evidence than the start/end of a stoppage.
    """

    team_switch_cost: float = 3.2
    alive_dead_cost: float = 1.7
    distance_scale_m: float = 2.4
    close_control_m: float = 3.5
    no_player_dead_logit: float = 3.5
    missing_ball_dead_logit: float = -0.35
    tag_team_weight: float = 0.65
    tag_restart_weight: float = 0.45
    min_alive_run_frames: int = 12
    min_dead_run_frames: int = 8


def load_config(path: Optional[str] = None) -> DecoderConfig:
    """Read an optional JSON config, rejecting accidental/unknown keys."""
    if path is None:
        return DecoderConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(DecoderConfig.__dataclass_fields__)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unknown config keys: {}".format(", ".join(sorted(unknown))))
    return DecoderConfig(**payload)


def save_config(config: DecoderConfig, path: str) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
