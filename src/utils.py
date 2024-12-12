from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PieceInfo:
    color: str
    piece: str

@dataclass
class BoardCorners:
    top_left: Tuple[float, float]
    top_right: Tuple[float, float]
    bottom_right: Tuple[float, float]
    bottom_left: Tuple[float, float]

class ChessboardState:
    def __init__(self):
        self.moves_detected: List[str] = []
        self.positions_per_frame: List[Dict] = []
        self.board_process_data = None
        self.all_chess_positions: set = set()
        self.offset_vectors = None
        self.first_move_black: bool = False 