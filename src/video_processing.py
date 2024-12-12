from typing import  Dict, Optional
import cv2
import logging
from dataclasses import dataclass
import numpy as np
from shapely.geometry import box, Polygon
from ultralytics import YOLO

from .board_processing import BoardProcessor
from .piece_processing import PieceProcessor
from .chess_notation import NotationGenerator
from .config import INFERENCE_SETTINGS, MODEL_PATHS

logger = logging.getLogger(__name__)

@dataclass
class FrameResult:
    frame_number: int
    pieces: Dict
    is_valid: bool
    hand_detected: bool

class VideoProcessor:
    def __init__(self, video_path: str, board_processor: BoardProcessor, 
                 piece_processor: PieceProcessor, notation_generator: NotationGenerator):
        """
        Initialize video processor with required components.
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.board_processor = board_processor
        self.piece_processor = piece_processor
        self.notation_generator = notation_generator
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_interval = int(self.fps * INFERENCE_SETTINGS['frame_interval_seconds'])
        self.state = {
            'moves_detected': [],
            'positions_per_frame': [],
            'chess_board_process_data': None,
            'all_chess_positions': set(),
            'offset_vectors': None,
            'first_move_black': False,
            'rotations': None,
            'board_polygon': None,
        }

        # Initialize hand detection model
        self.hands_model = YOLO(MODEL_PATHS['hands'])
        self.hands_confidence = INFERENCE_SETTINGS['hands_confidence']

    def annotate_frame_with_board(self, frame, board_corners):
        """
        Annotates the frame with the chessboard corners.
        """
        corner_points = [
            board_corners[0]['top-left'],
            board_corners[0]['top-right'],
            board_corners[0]['bottom-right'],
            board_corners[0]['bottom-left']
        ]
        corner_points = np.array(corner_points, dtype=np.int32)
        cv2.polylines(frame, [corner_points], isClosed=True, color=(0, 255, 0), thickness=2)
        return frame

    def _detect_hands(self, frame: np.ndarray) -> bool:
        hands_results = self.hands_model.predict(frame, save=False, retina_masks=True, verbose=False, conf=self.hands_confidence)
        hands = hands_results[0]
        if hasattr(hands, 'boxes') and hands.boxes is not None:
            hand_boxes = hands.boxes.xyxy.cpu().numpy()
            for box_coords in hand_boxes:
                xmin, ymin, xmax, ymax = box_coords
                hand_polygon = box(xmin, ymin, xmax, ymax)
                if self.state['board_polygon']:
                    intersection_area = self.state['board_polygon'].intersection(hand_polygon).area
                    if intersection_area / self.state['board_polygon'].area >= 0.01:
                        return True
        return False

    def _process_single_frame(self, frame: np.ndarray, frame_number: int) -> Optional[FrameResult]:
        """Process a single frame and return piece positions."""
        try:
            # Detect hands over board
            hand_detected = self._detect_hands(frame)
            if hand_detected:
                logger.info(f"Hand detected in frame {frame_number}")
                return FrameResult(frame_number, {}, False, True)

            # Process pieces only if we have board data
            if not self.state['chess_board_process_data']:
                board_process_data = self.board_processor.process_board(frame)
                if not board_process_data:
                    logger.warning("No chess board detected")
                    return None
                else:
                    self.state['chess_board_process_data'] = board_process_data
                    image_with_board, board_corners, all_square_corners, M, inverse_M = board_process_data
                    # Get board polygon
                    board_corners_points = [
                        board_corners[0]['top-left'],
                        board_corners[0]['top-right'],
                        board_corners[0]['bottom-right'],
                        board_corners[0]['bottom-left']
                    ]
                    board_polygon = Polygon(board_corners_points)
                    self.state['board_polygon'] = board_polygon
                    self.state['M'] = M
                    self.state['inverse_M'] = inverse_M
                    self.all_square_corners = all_square_corners

            # Process pieces
            image_with_board, board_corners, all_square_corners, M, inverse_M = self.state['chess_board_process_data']

            # First determine rotation if not already done
            if self.state['rotations'] is None:
                # Initial piece processing with rotation=0 to determine board orientation
                temp_pieces, _ = self.piece_processor.process_pieces(frame, all_square_corners, rotation=0)
                rotations = self.board_processor.find_rotation(temp_pieces, image_with_board, board_corners)
                self.state['rotations'] = rotations
                logger.info(f"Determined board rotation: {rotations} clockwise 90-degree turns")

            # Now process pieces with the correct rotation
            piece_centers, offset_vectors = self.piece_processor.process_pieces(
                frame, 
                all_square_corners, 
                rotation=self.state['rotations']
            )
            self.state['offset_vectors'] = offset_vectors

            # Convert piece_centers to a dictionary: tile -> piece_info
            current_pieces = {}
            for piece in piece_centers:
                tile = piece['tile']
                if tile is not None:
                    current_pieces[tile] = piece['piece_info']

            # Initialize all_chess_positions from the first frame
            if len(self.state["positions_per_frame"]) == 0:
                self.notation_generator.all_chess_positions = set(current_pieces.keys())
                logger.info("Initialized all_chess_positions with occupied tiles:")
                logger.info(self.notation_generator.all_chess_positions)

            return FrameResult(frame_number, current_pieces, True, False)

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return None

    def process_video(self) -> str:
        """
        Process entire video and return the formatted game notation.
        
        Returns:
            String of moves in algebraic notation
        """
        try:
            frame_number = 0
            previous_pieces = None
            notation_generator = self.notation_generator

            fps = self.fps
            frame_interval = self.frame_interval
            frames_in_n_seconds = int(fps * INFERENCE_SETTINGS['frame_interval_seconds'])
            banned_frames = set()

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_number += 1

                hand_detected = self._detect_hands(frame)
                if hand_detected:
                    # ban frames within one second
                    start_frame = max(1, frame_number - frames_in_n_seconds)
                    end_frame = frame_number + frames_in_n_seconds
                    banned_frames.update(range(start_frame, end_frame + 1))
                    logger.info(f"Banned frames from {start_frame} to {end_frame} due to hand detection on frame {frame_number}")

                if frame_number == 1 or frame_number % frame_interval == 0:
                    if frame_number in banned_frames:
                        logger.info(f"Skipping frame {frame_number} because it's banned.")
                        continue
                    else:
                        logger.info(f"Processing frame {frame_number}")
                        result = self._process_single_frame(frame, frame_number)
                        if not result or not result.is_valid:
                            continue
                        current_pieces = result.pieces
                        if previous_pieces:
                            notation, color = notation_generator.generate_move(previous_pieces, current_pieces)
                            if notation:
                                if len(self.state['moves_detected']) == 0 and color == 'black':
                                    logger.info("Black move first")
                                    self.state['first_move_black'] = True
                                logger.info(f"Detected move: {notation}")
                                self.state['moves_detected'].append(notation)
                        previous_pieces = current_pieces.copy()
                        self.state["positions_per_frame"].append({
                            'frame_number': frame_number,
                            'pieces': current_pieces
                        })

            self.cap.release()
            moves_notation = notation_generator.format_game_notation(self.state['moves_detected'], self.state['first_move_black'])
            return moves_notation

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        finally:
            self.cap.release()