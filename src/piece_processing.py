from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import cv2
import numpy as np
from typing import List, Dict, Tuple

class PieceProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.offset_vectors = []

    def get_proper_piece_info(self, label: int) -> Dict:
        """
        Maps an integer label to its corresponding piece information.
        """
        mapping = {
            0: 'black-bishop', 1: 'black-king', 2: 'black-knight',
            3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
            6: 'white-bishop', 7: 'white-king', 8: 'white-knight',
            9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
        }
        label_name = mapping.get(label, 'unknown-unknown')  
        try:
            color, piece = label_name.split("-")
        except ValueError:
            color, piece = 'unknown', 'unknown'
        return {'color': color, 'piece': piece}

    def position_to_tile(self, assigned_square: Tuple[int, int], rotation: int = 0) -> str:
        """
        Map the assigned square indices to the corresponding chessboard tile.
        """
        if assigned_square is None:
            return None
        col, row = assigned_square
        if rotation == 0:  # A1 is bottom-left
            file = chr(ord('A') + col)
            rank = 8 - row
        elif rotation == 1:  # A8 is bottom-left
            file = chr(ord('H') - col)
            rank = 8 - row
        elif rotation == 2:  # H8 is bottom-left
            file = chr(ord('H') - row)
            rank = 1 + col
        elif rotation == 3:  # H1 is bottom-left
            file = chr(ord('A') + col)
            rank = 1 + row
        else:
            raise ValueError("Rotation must be 0, 1, 2, or 3.")
        return f'{file}{rank}'

    def process_pieces(self, image_rgb: np.ndarray, all_square_corners: List[List[np.ndarray]], rotation: int = 0) -> Tuple[List[Dict], List[Tuple[float, float]]]:
        """
        Processes the chess pieces from the image and returns their positions.
        """
        # extract pieces detection results
        pieces_results = self.model.predict(image_rgb, save=False, retina_masks=True, verbose=False)
        pieces = pieces_results[0]
        boxes = pieces.boxes.xyxy.cpu().numpy()
        labels = pieces.boxes.cls.cpu().numpy().astype(int)
        scores = pieces.boxes.conf.cpu().numpy()

        piece_centers = []

        grid_size = len(all_square_corners)

        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box

            piece_label = labels[i]

            piece_box_polygon = Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ])

            overlaps = []
            for row in range(grid_size):
                for col in range(grid_size):
                    square_corners = all_square_corners[row][col]
                    square_polygon = Polygon(square_corners)

                    # compute the area of the intersection
                    intersection = piece_box_polygon.intersection(square_polygon)
                    overlap_area = intersection.area

                    if overlap_area > 0:
                        overlaps.append({
                            'overlap_area': overlap_area,
                            'square_index': (row, col)
                        })

            # if no overlaps found, proceed to next piece
            if not overlaps:
                print(f"Piece {i} does not overlap with any square.")
                piece_info = {
                    'piece_index': i,
                    'piece_info': self.get_proper_piece_info(piece_label),
                    'confidence': scores[i] if scores is not None and i < len(scores) else None,
                    'pos_x': (xmin + xmax) / 2,
                    'pos_y': (ymin + ymax) / 2,
                    'adjusted_pos_x': None,
                    'adjusted_pos_y': None,
                    'assigned_square': None,
                    'tile': None
                }
                piece_centers.append(piece_info)
                continue

            # overlaps descending order
            overlaps.sort(key=lambda x: x['overlap_area'], reverse=True)

            max_overlap = overlaps[0]
            assigned_square = max_overlap['square_index']

            # if they overlap with more than one square, use second biggest difference for offset vector
            if len(overlaps) > 1:
                second_max_overlap = overlaps[1]

                overlap_diff = max_overlap['overlap_area'] - second_max_overlap['overlap_area']

                if second_max_overlap['overlap_area'] >= 0.05 * max_overlap['overlap_area']:
                    # compute vector from the center of the square with less overlap to the one with more overlap
                    row1, col1 = second_max_overlap['square_index']
                    row2, col2 = max_overlap['square_index']

                    square1_corners = all_square_corners[row1][col1]
                    square2_corners = all_square_corners[row2][col2]

                    center1_x = (square1_corners[0][0] + square1_corners[2][0]) / 2
                    center1_y = (square1_corners[0][1] + square1_corners[2][1]) / 2

                    center2_x = (square2_corners[0][0] + square2_corners[2][0]) / 2
                    center2_y = (square2_corners[0][1] + square2_corners[2][1]) / 2

                    dx = center2_x - center1_x
                    dy = center2_y - center1_y

                    self.offset_vectors.append((dx, dy))

            square_corners = all_square_corners[assigned_square[0]][assigned_square[1]]
            square_top_left = square_corners[0]
            square_bottom_right = square_corners[2]
            adjusted_center_x = (square_top_left[0] + square_bottom_right[0]) / 2
            adjusted_center_y = (square_top_left[1] + square_bottom_right[1]) / 2

            tile = self.position_to_tile(assigned_square, rotation=rotation)

            piece_info = {
                'piece_index': i,
                'piece_info': self.get_proper_piece_info(piece_label),
                'confidence': scores[i] if scores is not None and i < len(scores) else None,
                'pos_x': (xmin + xmax) / 2,
                'pos_y': (ymin + ymax) / 2,
                'adjusted_pos_x': adjusted_center_x,
                'adjusted_pos_y': adjusted_center_y,
                'assigned_square': assigned_square,
                'tile': tile
            }

            piece_centers.append(piece_info)

            cv2.circle(image_rgb, (int(adjusted_center_x), int(adjusted_center_y)), radius=12, color=(0, 0, 255), thickness=-1)
            if tile is not None:
                cv2.putText(image_rgb, tile, (int(adjusted_center_x) + 10, int(adjusted_center_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # compute the average offset vector and change piece location if they are not in correct tile
        if self.offset_vectors:
            avg_dx = sum(v[0] for v in self.offset_vectors) / len(self.offset_vectors)
            avg_dy = sum(v[1] for v in self.offset_vectors) / len(self.offset_vectors)
            print(f"Average offset: dx={avg_dx}, dy={avg_dy}")

            for index, piece in enumerate(piece_centers):
                if piece['adjusted_pos_x'] is not None and piece['adjusted_pos_y'] is not None:
                    # adjust adjustment size to piece type
                    factor = 8
                    if(piece["piece_info"]["piece"] in ["queen", "king"]):
                        factor = 3
                    if(piece["piece_info"]["piece"] in ["rook", "bishop","knight"]):
                        factor = 5
                    piece['adjusted_pos_x'] = piece["pos_x"] + avg_dx/factor
                    piece['adjusted_pos_y'] = piece["pos_y"] + avg_dy/factor

                    # update the assigned square based on the new position
                    adjusted_point = Point(piece['adjusted_pos_x'], piece['adjusted_pos_y'])
                    assigned_square = None
                    for row in range(grid_size):
                        for col in range(grid_size):
                            square_corners = all_square_corners[row][col]
                            square_polygon = Polygon(square_corners)
                            if square_polygon.contains(adjusted_point):
                                assigned_square = (row, col)
                                break
                        if assigned_square is not None:
                            break

                    if assigned_square is not None:
                        piece_centers[index]['assigned_square'] = assigned_square
                        piece_centers[index]['tile'] = self.position_to_tile(assigned_square, rotation=rotation)
                    else:
                        piece_centers[index]['tile'] = None

                    # annotations for error fixes
                    cv2.circle(image_rgb, (int(piece['adjusted_pos_x']), int(piece['adjusted_pos_y'])), radius=8, color=(255, 0, 0), thickness=-1)
                    if piece['tile'] is not None:
                        cv2.putText(image_rgb, piece['tile'], (int(piece['adjusted_pos_x']) + 10, int(piece['adjusted_pos_y']) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for index, piece in enumerate(piece_centers):
            cv2.circle(image_rgb, (int(piece['pos_x']), int(piece['pos_y'])), radius=8, color=(0, 255, 0), thickness=-1)
        
        return piece_centers, self.offset_vectors