"""Helpers for feeding recorded human-play games back into training.

Stage 8 stores only AI turns as samples for `human_play` records. These helpers
make that explicit and return `GameStepSample` lists that can be mixed into
future training pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence

from gomoku_ai.alphazero.specs import GameRecord, GameStepSample


def maybe_extract_ai_turn_samples(
    record: GameRecord,
    *,
    record_ai_turn_only: bool = True,
) -> list[GameStepSample]:
    """Return the AI-turn samples stored inside one `human_play` record."""

    if not isinstance(record, GameRecord):
        raise TypeError("record must be a GameRecord instance.")
    if record.source != "human_play":
        raise ValueError("record.source must be 'human_play'.")
    if not record_ai_turn_only:
        raise NotImplementedError("Stage 8 baseline only stores AI-turn-only samples.")
    return list(record.samples)


def extract_human_play_samples(
    records: Sequence[GameRecord],
    *,
    record_ai_turn_only: bool = True,
) -> list[GameStepSample]:
    """Flatten AI-turn samples from multiple recorded human-play games."""

    extracted_samples: list[GameStepSample] = []
    for record in records:
        extracted_samples.extend(
            maybe_extract_ai_turn_samples(record, record_ai_turn_only=record_ai_turn_only)
        )
    return extracted_samples
