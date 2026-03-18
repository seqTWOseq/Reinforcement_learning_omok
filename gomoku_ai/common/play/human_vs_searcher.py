"""Shared human-vs-searcher play helpers for Athenan entrypoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class AthenanAIDebugTurn:
    """Debug payload for one AI move."""

    move_index: int
    action: int
    root_value: float
    principal_variation: tuple[int, ...]
    top_actions: tuple[tuple[int, float], ...]
    nodes: int
    depth_reached: int
    forced_tactical: bool


@dataclass(frozen=True)
class HumanVsAthenanResult:
    """Result of one human-vs-Athenan game."""

    winner: int
    move_count: int
    moves: tuple[int, ...]
    human_color: int
    ai_debug_turns: tuple[AthenanAIDebugTurn, ...]


def play_human_vs_searcher_game(
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
    gui_title: str = "Gomoku vs Athenan",
    ai_move_delay_sec: float = 0.15,
    keep_gui_open: bool = True,
) -> HumanVsAthenanResult:
    """Play one game between a human side and a searcher-backed Athenan agent."""

    if human_color not in {BLACK, WHITE}:
        raise ValueError(f"human_color must be BLACK({BLACK}) or WHITE({WHITE}).")
    if debug_top_k <= 0:
        raise ValueError("debug_top_k must be positive.")
    if ai_move_delay_sec < 0.0:
        raise ValueError("ai_move_delay_sec must be non-negative.")

    env = (env_factory or GomokuEnv)()
    env.reset()
    if use_gui is None:
        use_gui = human_moves is None and human_move_selector is None
    human_move_iter = iter(human_moves) if human_moves is not None else None
    moves: list[int] = []
    ai_debug_turns: list[AthenanAIDebugTurn] = []
    gui = None

    if use_gui:
        from gomoku_ai.common.play.tk_gomoku_board import TkGomokuBoard

        gui = TkGomokuBoard(env, title_prefix=gui_title)
        gui.render()

    try:
        while not env.done:
            moving_player = int(env.current_player)
            if moving_player == human_color:
                if gui is not None and human_move_selector is None and human_move_iter is None:
                    action = gui.select_human_action()
                else:
                    action = _resolve_human_action(
                        env,
                        human_move_selector=human_move_selector,
                        human_move_iter=human_move_iter,
                        input_fn=input_fn,
                        print_fn=print_fn,
                    )
            else:
                search_result = searcher.search(env)
                action = int(search_result.best_action)
                if action < 0:
                    raise RuntimeError("Athenan searcher returned best_action < 0 on non-terminal human-play turn.")
                debug_turn = _build_ai_debug_turn(
                    move_index=len(moves),
                    action=action,
                    search_result=search_result,
                    top_k=debug_top_k,
                )
                ai_debug_turns.append(debug_turn)
                if debug:
                    print_fn(_format_ai_debug(debug_turn))

            _assert_legal_action(env, action)
            env.apply_move(action)
            moves.append(int(action))
            if gui is not None:
                gui.render()
                if moving_player != human_color and ai_move_delay_sec > 0.0 and not env.done:
                    time.sleep(ai_move_delay_sec)
    finally:
        if gui is not None and gui.window is not None:
            gui.window.update_idletasks()

    if env.winner is None:
        raise RuntimeError("Game ended without winner information.")
    if debug:
        print_fn(_format_game_result(winner=env.winner, move_count=env.move_count))
    if gui is not None and keep_gui_open:
        gui.keep_open()
    elif gui is not None:
        gui.close()
    return HumanVsAthenanResult(
        winner=int(env.winner),
        move_count=int(env.move_count),
        moves=tuple(moves),
        human_color=human_color,
        ai_debug_turns=tuple(ai_debug_turns),
    )


def parse_human_color(raw: str) -> int:
    """Parse human-side color from CLI input."""

    normalized = raw.strip().lower()
    if normalized in {"black", "b"}:
        return BLACK
    if normalized in {"white", "w"}:
        return WHITE
    raise ValueError("human-color must be one of {'black', 'white'}.")


def _resolve_human_action(
    env: GomokuEnv,
    *,
    human_move_selector: Callable[[GomokuEnv], int] | None,
    human_move_iter: object | None,
    input_fn: Callable[[str], str],
    print_fn: Callable[[str], None],
) -> int:
    if human_move_selector is not None:
        return int(human_move_selector(env))
    if human_move_iter is not None:
        try:
            return int(next(human_move_iter))
        except StopIteration as exc:
            raise RuntimeError("No more scripted human moves are available.") from exc

    while True:
        raw = input_fn("Enter action (int) or row,col: ").strip()
        try:
            action = _parse_human_action(raw, env=env)
        except ValueError as exc:
            print_fn(str(exc))
            continue
        return action


def _parse_human_action(raw: str, *, env: GomokuEnv) -> int:
    if "," in raw:
        parts = [part.strip() for part in raw.split(",", maxsplit=1)]
        if len(parts) != 2:
            raise ValueError("Input must be 'row,col' or one integer action index.")
        row = int(parts[0])
        col = int(parts[1])
        return int(env.coord_to_action(row, col))
    return int(raw)


def _assert_legal_action(env: GomokuEnv, action: int) -> None:
    try:
        env.action_to_coord(action)
    except ValueError as exc:
        raise RuntimeError(f"Action {action} is out of range.") from exc
    if not env.is_legal_action(action):
        raise RuntimeError(f"Action {action} is illegal for current position.")


def _build_ai_debug_turn(
    *,
    move_index: int,
    action: int,
    search_result: SearchResult,
    top_k: int,
) -> AthenanAIDebugTurn:
    top_actions = tuple(
        sorted(
            ((int(candidate_action), float(value)) for candidate_action, value in search_result.action_values.items()),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )
    return AthenanAIDebugTurn(
        move_index=move_index,
        action=action,
        root_value=float(search_result.root_value),
        principal_variation=tuple(int(candidate) for candidate in search_result.principal_variation),
        top_actions=top_actions,
        nodes=int(search_result.nodes),
        depth_reached=int(search_result.depth_reached),
        forced_tactical=bool(search_result.forced_tactical),
    )


def _format_ai_debug(debug_turn: AthenanAIDebugTurn) -> str:
    return (
        f"[AI] move={debug_turn.move_index} action={debug_turn.action} "
        f"root_value={debug_turn.root_value:.4f} depth={debug_turn.depth_reached} "
        f"nodes={debug_turn.nodes} forced={debug_turn.forced_tactical} "
        f"pv={list(debug_turn.principal_variation)} top_actions={list(debug_turn.top_actions)}"
    )


def _format_game_result(*, winner: int, move_count: int) -> str:
    if winner == DRAW:
        winner_text = "draw"
    elif winner == BLACK:
        winner_text = "black"
    elif winner == WHITE:
        winner_text = "white"
    else:
        winner_text = str(winner)
    return f"[Result] winner={winner_text} move_count={move_count}"
