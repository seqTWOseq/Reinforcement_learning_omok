"""Human-vs-value-net-Athenan play entrypoint with optional debug output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gomoku_ai.athenan.network import load_athenan_value_net
from gomoku_ai.common.play.human_vs_searcher import (
    HumanVsAthenanResult,
    parse_human_color,
    play_human_vs_searcher_game,
)
from gomoku_ai.athenan.search.valuenet_athenan_inference_searcher import AthenanInferenceSearcher
from gomoku_ai.common.agents import BaseSearcher
from gomoku_ai.env import BLACK, GomokuEnv


def play_human_vs_valuenet_athenan_game(
    *,
    searcher: BaseSearcher,
    human_color: int = BLACK,
    env_factory: Callable[[], GomokuEnv] | None = None,
    human_moves: Sequence[int] | None = None,
    human_move_selector: Callable[[GomokuEnv], int] | None = None,
    debug: bool = False,
    debug_top_k: int = 5,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    use_gui: bool | None = None,
    gui_title: str = "Gomoku vs ValueNet Athenan",
    ai_move_delay_sec: float = 0.15,
    keep_gui_open: bool = True,
) -> HumanVsAthenanResult:
    """Play one game between a human side and the value-net Athenan searcher."""
    return play_human_vs_searcher_game(
        searcher=searcher,
        human_color=human_color,
        env_factory=env_factory,
        human_moves=human_moves,
        human_move_selector=human_move_selector,
        debug=debug,
        debug_top_k=debug_top_k,
        input_fn=input_fn,
        print_fn=print_fn,
        use_gui=use_gui,
        gui_title=gui_title,
        ai_move_delay_sec=ai_move_delay_sec,
        keep_gui_open=keep_gui_open,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for human-vs-Athenan play."""

    parser = argparse.ArgumentParser(
        description="Play Gomoku against Athenan inference search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--human-color", type=str, default="black", help="Human side color: black or white.")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Optional Athenan value-net checkpoint path.")
    parser.add_argument("--max-depth", type=int, default=4, help="Inference max depth.")
    parser.add_argument("--candidate-limit", type=int, default=64, help="Candidate action limit.")
    parser.add_argument("--candidate-radius", type=int, default=2, help="Candidate radius.")
    parser.add_argument("--time-budget-sec", type=float, default=0.0, help="0 means no time budget.")
    parser.add_argument("--no-iterative", action="store_true", help="Disable iterative deepening.")
    parser.add_argument("--ai-move-delay", type=float, default=0.15, help="Delay after each AI move for easier viewing.")
    parser.add_argument("--debug", action="store_true", help="Print root debug info each AI move.")
    parser.add_argument("--debug-top-k", type=int, default=5, help="How many root actions to show in debug output.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for model inference.")
    args = parser.parse_args(argv)

    human_color = parse_human_color(args.human_color)
    checkpoint_path = args.checkpoint_path.strip()
    model = load_athenan_value_net(Path(checkpoint_path), device=args.device) if checkpoint_path else None
    time_budget_sec = None if args.time_budget_sec <= 0.0 else float(args.time_budget_sec)

    searcher = AthenanInferenceSearcher(
        model=model,
        max_depth=args.max_depth,
        candidate_limit=args.candidate_limit,
        candidate_radius=args.candidate_radius,
        iterative_deepening=not args.no_iterative,
        time_budget_sec=time_budget_sec,
        device=args.device,
    )
    play_human_vs_valuenet_athenan_game(
        searcher=searcher,
        human_color=human_color,
        debug=args.debug,
        debug_top_k=args.debug_top_k,
        use_gui=True,
        gui_title="Gomoku vs ValueNet Athenan",
        ai_move_delay_sec=args.ai_move_delay,
    )

if __name__ == "__main__":
    main()
