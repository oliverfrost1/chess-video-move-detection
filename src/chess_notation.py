from typing import Dict, List

class NotationGenerator:
    def __init__(self):
        self.all_chess_positions = set()
        self.first_move_black = False

    def get_primitive_chess_notation(self, piece: str, old_position: str, new_position: str) -> str:
        """
        Generates the chess notation for a move.
        """
        notation = ""

        piece_to_letter = {
            'pawn': '',
            'knight': 'N',
            'bishop': 'B',
            'rook': 'R',
            'queen': 'Q',
            'king': 'K'
        }

        piece_letter = piece_to_letter.get(piece.lower(), '')

        # determine if capture occurred
        is_capture = new_position in self.all_chess_positions

        # remove old_position from all_chess_positions
        self.all_chess_positions.discard(old_position)

        if piece.lower() == 'pawn':
            if is_capture:
                # for pawn captures, use the file of the pawn's starting square
                notation += old_position[0].lower() + 'x' + new_position.lower()
            else:
                notation += new_position.lower()
        else:
            notation += piece_letter
            if is_capture:
                notation += 'x'
            notation += new_position.lower()

        # add new_position to all_chess_positions
        self.all_chess_positions.add(new_position)

        return notation

    def generate_move(self, previous_pieces: Dict, current_pieces: Dict):
        vacated_tiles = set()
        occupied_tiles = set()

        for tile in set(previous_pieces.keys()).union(current_pieces.keys()):
            prev_piece = previous_pieces.get(tile)
            curr_piece = current_pieces.get(tile)

            if prev_piece and not curr_piece:
                vacated_tiles.add(tile)
            elif not prev_piece and curr_piece:
                occupied_tiles.add(tile)
            elif prev_piece and curr_piece:
                if prev_piece != curr_piece:
                    vacated_tiles.add(tile)
                    occupied_tiles.add(tile)

        # try to match vacated and occupied tiles
        if len(vacated_tiles) == 1 and len(occupied_tiles) == 1:
            old_position = vacated_tiles.pop()
            new_position = occupied_tiles.pop()

            if old_position != new_position:
                piece_info = current_pieces.get(new_position)
                piece = piece_info["piece"]
                color = piece_info["color"]
                notation = self.get_primitive_chess_notation(piece, old_position, new_position)
                return notation, color
        else:
            # multiple vacated or occupied tiles, attempt to match based on piece info
            for vacated in vacated_tiles:
                previous_piece_info = previous_pieces.get(vacated)
                for occupied in occupied_tiles:
                    current_piece_info = current_pieces.get(occupied)
                    if previous_piece_info and current_piece_info:
                        if (previous_piece_info["color"] == current_piece_info["color"] and
                            previous_piece_info["piece"] == current_piece_info["piece"]):
                            old_position = vacated
                            new_position = occupied
                            piece = current_piece_info["piece"]
                            color = current_piece_info["color"]
                            notation = self.get_primitive_chess_notation(piece, old_position, new_position)
                            return notation, color
        return None, None

    def format_game_notation(self, moves: List[str], first_move_black: bool) -> str:
        """Format the full game notation."""
        if not moves:
            return ""
            
        if first_move_black:
            notation = f"1... {moves[0]}"
            move_number = 2
            idx = 1
        else:
            notation = f"1. {moves[0]}"
            idx = 1
            if idx < len(moves):
                notation += f" {moves[idx]}"
                idx += 1
            move_number = 2

        while idx < len(moves):
            notation += f" {move_number}."
            notation += f" {moves[idx]}"
            idx += 1
            if idx < len(moves):
                notation += f" {moves[idx]}"
                idx += 1
            move_number += 1

        return notation 