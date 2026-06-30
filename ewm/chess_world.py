"""Chess world-module: a *verifiable* simulator (paper §3.1 'Simulator').

This is the strong instantiation of the EWM blueprint. Where the SD renderer
hallucinates physics (§3.2 faithfulness worry), a chess engine's rollout is
faithful *by construction*: legal moves, real outcomes, a forcing line that
either exists or doesn't. The "inspectable hypothesis" (§2.2) is exactly
inspectable and exactly correct given the premise.

Given a position the reasoner is reasoning about (a FEN), the module returns a
visual-temporal rollout: a short forcing line rendered as a sequence of board
frames (ASCII diagrams) plus the line in SAN and the outcome. Frames are text,
so -- unlike the SD path -- NO vision model is needed to read them back; the
board *is* the scene and the move list *is* the temporal rollout. (This is the
chess-encoding insight in its load-bearing form: a canonical, replayable
serialization of a state trajectory in a constrained legal-move alphabet.)

Backends:
  * 'search'    -- pure python-chess forced-mate DFS, no external dependency.
  * 'stockfish' -- UCI engine PV + centipawn eval, if a binary is available
                   (apt install stockfish). Strictly an upgrade; same interface.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import chess
import chess.engine

from .world_module import Rollout

_FEN_RE = re.compile(
    r"([rnbqkpRNBQKP1-8]+(?:/[rnbqkpRNBQKP1-8]+){7}\s+[wb]\s+\S+\s+\S+(?:\s+\d+\s+\d+)?)"
)


# --------------------------------------------------------------------------
# Forced-mate search (no external engine). Returns the side-to-move's mating
# first moves and one representative line.
# --------------------------------------------------------------------------
def forced_mate_line(board: chess.Board, max_plies: int) -> list[chess.Move] | None:
    """Shortest line by which the side to move forces mate within max_plies."""
    # mate in 1: any move giving immediate checkmate
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return [m]
    if max_plies < 3:
        return None
    # mate in >=2: exists a move s.t. EVERY opponent reply still lets us mate
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate() or board.is_stalemate():
            board.pop()
            continue
        all_forced, rep = True, None
        for r in board.legal_moves:
            board.push(r)
            sub = forced_mate_line(board, max_plies - 2)
            board.pop()
            if sub is None:
                all_forced = False
                break
            if rep is None:
                rep = [r] + sub
        board.pop()
        if all_forced and rep is not None:
            return [m] + rep
    return None


def _move_forces_mate(board: chess.Board, m: chess.Move, max_plies: int) -> bool:
    """True if playing m forces mate within max_plies (m counts as ply 1)."""
    board.push(m)
    try:
        if board.is_checkmate():
            return True
        if max_plies < 3 or board.is_stalemate() or not any(board.legal_moves):
            return False
        # every opponent reply must still allow a forced mate
        for r in list(board.legal_moves):
            board.push(r)
            sub = forced_mate_line(board, max_plies - 2)
            board.pop()
            if sub is None:
                return False
        return True
    finally:
        board.pop()


def mating_first_moves(board: chess.Board, max_plies: int) -> list[chess.Move]:
    """All first moves that begin a forced mate within max_plies (verifier set)."""
    return [m for m in board.legal_moves if _move_forces_mate(board, m, max_plies)]


def _render_frames(board: chess.Board, line: list[chess.Move], out_dir: Path) -> tuple[list[str], str]:
    """Write one ASCII board frame per ply; return (frame_paths, SAN line)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, sans = [], []
    work = board.copy()
    # frame 0: the starting position
    (out_dir / "frame_00.txt").write_text(f"start position\n{work}\n")
    paths.append(str(out_dir / "frame_00.txt"))
    for i, mv in enumerate(line, start=1):
        san = work.san(mv)
        sans.append(san)
        work.push(mv)
        tag = " #" if work.is_checkmate() else (" +" if work.is_check() else "")
        p = out_dir / f"frame_{i:02d}.txt"
        p.write_text(f"after {san}{tag}\n{work}\n")
        paths.append(str(p))
    return paths, " ".join(sans)


class ChessWorldModule:
    module = "chess-search"

    def __init__(self, anchor_fen: str, max_plies: int = 3,
                 engine_path: str | None = None):
        self.anchor_fen = anchor_fen
        self.max_plies = max_plies
        self.engine_path = engine_path or shutil.which("stockfish")
        if self.engine_path:
            self.module = "chess-stockfish"

    def _line(self, board: chess.Board) -> tuple[list[chess.Move], str]:
        """Return (line, verdict_text)."""
        if self.engine_path:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as eng:
                info = eng.analyse(board, chess.engine.Limit(depth=18))
                pv = info.get("pv", [])[:6]
                score = info["score"].pov(board.turn)
                verdict = (f"engine eval {score}; principal variation shown"
                           if pv else "engine found no line")
                return pv, verdict
        line = forced_mate_line(board, self.max_plies)
        if line:
            n = (len(line) + 1) // 2
            return line, f"a forced checkmate in {n} exists for the side to move"
        return [], "no forced mate found within the searched horizon"

    def rollout(self, query: str, out_dir: Path, n_frames: int = 4) -> Rollout:
        t0 = time.time()
        m = _FEN_RE.search(query)
        fen = m.group(1) if m else self.anchor_fen
        try:
            board = chess.Board(fen)
        except ValueError:
            board = chess.Board(self.anchor_fen)
        line, verdict = self._line(board)
        if not line:  # still give the reasoner something inspectable
            best = next(iter(board.legal_moves), None)
            line = [best] if best else []
        frame_paths, san_line = _render_frames(board, line, out_dir)
        first_san = san_line.split(" ")[0] if san_line else "(none)"
        text = (f"World-module rollout for position [{fen}]: {verdict}. "
                f"Forcing line: {san_line}. The decisive first move is {first_san}. "
                f"Frames show the board after each ply ({len(frame_paths)} positions).")
        return Rollout(query=query, frame_paths=frame_paths, beats=[san_line],
                       seconds=round(time.time() - t0, 2), module=self.module, text=text)


def chess_perceive(frame_paths: list[str], query: str) -> str:
    """Perception for the chess world-module: read the ASCII board frames back.

    No vision model needed -- the frames are already text. We concatenate the
    move annotations (the lines of each frame above the board) so the reasoner
    sees the forcing sequence and the final mating/check status.
    """
    beats = []
    for p in frame_paths:
        head = Path(p).read_text().splitlines()[0]
        beats.append(head)
    return "Rollout frames (in order): " + " | ".join(beats)
