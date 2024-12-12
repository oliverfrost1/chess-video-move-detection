from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, box
from PIL import Image 
import pandas as pd



# INPUTS 
# Insert videos here
video_paths = [
    'video1.mp4',
    'video2.mp4',
    'video3.mp4',
    'video4.mp4',
    'video5.mp4',
    'video6.mp4',
]

# INFERENCE DEFINITIONS 
yoloPieces = YOLO('./models/pieces-model.pt')
yoloBoard = YOLO('./models/board-model.pt')
yoloHands = YOLO('./models/hands-model.pt')

def callBoardInference(image):
  return yoloBoard.predict(image, save=False, retina_masks=True, verbose=False)

def callPiecesInference(image):
  return yoloPieces.predict(image, save=False, retina_masks=True, verbose=False)

def callHandInference(image):
    return yoloHands.predict(image, save=False, retina_masks=True, verbose=False, conf=0.65)

# MAIN PROCESSING FUNCTIONS
def order_points(pts):
    """
    Orders the corner points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

def process_board(image, board_results):
    """
    Processes the chessboard from the image and returns the board information,
    including the corners of each square. Also overlays the mask on the image.
    """
    if image is None:
        print("Error: Image could not be loaded.")
        return None, None, None, None, None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]

    masks = getattr(board_results[0], 'masks', None)
    if masks is not None and hasattr(masks, 'data'):
        masks = masks.data.cpu().numpy()

    all_corners = {}
    all_square_centers = []
    all_square_corners = []
    M = None  # Perspective transformation matrix
    inverse_M = None  # Inverse perspective transformation matrix

    if masks is not None:
        for idx, mask in enumerate(masks):
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            # overlay the mask on the image
            color_mask = np.zeros_like(image_rgb)
            color_mask[mask_resized == 1] = [255, 0, 0] 
            image_rgb = cv2.addWeighted(image_rgb, 1, color_mask, 0.5, 0)

            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"Mask {idx+1}: No contours found.")
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a polygon
            peri = cv2.arcLength(largest_contour, True)
            epsilon = 0.02 * peri  # Adjusted epsilon value
            approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx_corners) == 4:
                # If the approximated contour has 4 points, use them as corners
                contour_points = approx_corners.reshape(-1, 2)
            else:
                # If not, use the minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                contour_points = np.int0(box)
            # order the points consistently: top-left, top-right, bottom-right, bottom-left
            contour_points = order_points(contour_points)
            
            top_left, top_right, bottom_right, bottom_left = contour_points
            
            all_corners[idx] = {
                'top-left': tuple(top_left),
                'top-right': tuple(top_right),
                'bottom-right': tuple(bottom_right),
                'bottom-left': tuple(bottom_left)
            }
            # annotations
            for corner_name, (x, y) in all_corners[idx].items():
                cv2.circle(image_rgb, (int(x), int(y)), radius=8, color=(0, 255, 0), thickness=-1)
                cv2.putText(image_rgb, corner_name, (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"Mask {idx+1}: Detected corners - {all_corners[idx]}")

            # perspective transformation
            pts_src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            side_length = max(
                np.linalg.norm(top_left - top_right),
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(bottom_left - top_left)
            )
            side_length = int(side_length)
            pts_dst = np.array([
                [0, 0],
                [side_length - 1, 0],
                [side_length - 1, side_length - 1],
                [0, side_length - 1]
            ], dtype=np.float32)
            
            # compute transformation matrices
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            inverse_M = cv2.getPerspectiveTransform(pts_dst, pts_src)
            
            # generate grid and square centers and corners in warped view
            num_cells = 8
            grid_size = side_length / num_cells
            for i in range(num_cells):
                for j in range(num_cells):
                    # compute square center in warped image
                    center_warped = ((j + 0.5) * grid_size, (i + 0.5) * grid_size)
                    all_square_centers.append(center_warped)
                    
                    # compute square corners in warped image
                    top_left_warped = (j * grid_size, i * grid_size)
                    top_right_warped = ((j + 1) * grid_size, i * grid_size)
                    bottom_right_warped = ((j + 1) * grid_size, (i + 1) * grid_size)
                    bottom_left_warped = (j * grid_size, (i + 1) * grid_size)
                    square_corners_warped = [top_left_warped, top_right_warped, bottom_right_warped, bottom_left_warped]
                    all_square_corners.append(square_corners_warped)
            
            # transform square centers back to original image
            transformed_square_centers = []
            for center in all_square_centers:
                center_array = np.array([[center]], dtype=np.float32)
                center_transformed = cv2.perspectiveTransform(center_array, inverse_M)[0][0]
                transformed_square_centers.append(center_transformed)
                
                # annotate centers with blue circle
                center_int = tuple(center_transformed.astype(int))
                cv2.circle(image_rgb, center_int, radius=4, color=(0, 0, 255), thickness=-1)
            all_square_centers = transformed_square_centers
            
            # transform squares back to original image dimensions
            transformed_square_corners = []
            for corners in all_square_corners:
                corners_array = np.array([corners], dtype=np.float32)  
                corners_transformed = cv2.perspectiveTransform(corners_array, inverse_M)[0]  
                transformed_square_corners.append(corners_transformed)
                
            all_square_corners = transformed_square_corners
            # draw grid lines on image
            for i in range(num_cells + 1):
                # vertical 
                start_point = (i * grid_size, 0)
                end_point = (i * grid_size, side_length)
                line_pts = np.array([[start_point, end_point]], dtype=np.float32)
                line_transformed = cv2.perspectiveTransform(line_pts, inverse_M)[0]
                cv2.line(image_rgb, tuple(line_transformed[0].astype(int)), tuple(line_transformed[1].astype(int)), (255, 0, 0), 1)
                
                # horizontal 
                start_point = (0, i * grid_size)
                end_point = (side_length, i * grid_size)
                line_pts = np.array([[start_point, end_point]], dtype=np.float32)
                line_transformed = cv2.perspectiveTransform(line_pts, inverse_M)[0]
                cv2.line(image_rgb, tuple(line_transformed[0].astype(int)), tuple(line_transformed[1].astype(int)), (255, 0, 0), 1)
            break  # process only the first valid mask
    else:
        print("No masks found in the board results.")
        return image_rgb, None, None, None, None, None

    # transform all square corners to be a 2D array based on a max length of 8
    number = 8
    new_square_corners = []
    # check if the number of squares is exactly 64 (8x8)
    expected_squares = number * number
    actual_squares = len(all_square_corners)
    if actual_squares != expected_squares:
        print(f"Warning: Expected {expected_squares} squares, but got {actual_squares}.")
    
    # transform the flat list into a 2D list (8x8)
    for i in range(actual_squares):
        row_index = i // number  
        if row_index >= len(new_square_corners):
            new_square_corners.append([])
        # append the current square's corners to the appropriate row
        new_square_corners[row_index].append(all_square_corners[i])
    return image_rgb, all_corners, all_square_centers, new_square_corners, M, inverse_M

def get_warped_coordinates(transform_matrix, original_point):
    # convert the original point to coords
    original_point_homogeneous = np.array([*original_point, 1], dtype='float32')

    warped_point_homogeneous = transform_matrix @ original_point_homogeneous

    # convert back to Cartesian coordinates
    warped_point = warped_point_homogeneous[:2] / warped_point_homogeneous[2]

    return warped_point

def find_rotation(pieces=[], board_rgb=np.array([[[]]]), board_corners={}):
  '''
  Returns how many clockwise 90 degree rotations that should be made to get to correct state.
  That is how many counter-clockwise 90 degree rotations the chess board is rotated.
  '''
  rotations = 0

  # first find the 90 degree orientation. If chess board is rotated 90 degrees 
  # a white square is colored instead. Program finds this by finding an empty square,
  # then deciding whether its white or colored by comparing its brightness
  # to its also emtpy neighboring square. From coordinates of square determine whether
  # it should be white or not.

  # create set of all squares with pieces on them to make it easier to
  # find adjacent squares with no pieces later
  taken_spots = set()
  for piece in pieces:
    taken_spots.add(piece["assigned_square"])

  # find two empty adjacent squares:
  squares = None
  for y in range(0,8):
    for x1, x2 in [(x, x + 1) for x in range(0, 7)]:
      if not ((x1, y) in taken_spots or (x2, y) in taken_spots):
        squares = ((x1, y), (x2, y))
        break
  if not squares:
    for x in range(0, 8):
      for y1, y2 in [[y, y+1] for y in range(0, 7)]:
        if not ((x, y1) in taken_spots or (x, y2) in taken_spots):
          squares = ((x, y1), (x, y2))
          break

  
  gray_image = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY)

  # warp image for a top-down view to divide board into squares easier

  # preparations for build in cv2.warpPerspective
  width = 400  # Width of the chessboard in the warped image
  height = 400  # Height of the chessboard in the warped image
  dst_corners = np.array([ [0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

  # Convert corners to a numpy array for use in cv2.getPerspectiveTransform
  src_corners = np.array([np.array(v) for k, v in board_corners[0].items()], dtype='float32')
    
  # Get the perspective transform matrix
  matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
  warped_image = cv2.warpPerspective(gray_image, matrix, (width, height))

  # find pixels of chosen squares and calc brightness
  square_size = height // 8
  first_square = warped_image[squares[0][0]*square_size:(squares[0][0]+1)*square_size, squares[0][1]*square_size:(squares[0][1]+1)*square_size]
  avg_brightness_first = np.mean(first_square)

  second_square = warped_image[squares[1][0]*square_size:(squares[1][0]+1)*square_size, squares[1][1]*square_size:(squares[1][1]+1)*square_size]
  avg_brightness_second = np.mean(second_square)

  # even sum of coordinates for a square means it should be white
  if ((squares[0][0] + squares[0][1]) % 2) == 0: # sum even so it should be white
    if avg_brightness_first < avg_brightness_second: # first square should be, but is not, white. So it's rotated
      rotations += 1
  else: 
    if avg_brightness_first > avg_brightness_second: # first square is, but should not be, white. So it's rotated
      rotations += 1

  # Now determine 180 degree orientation based on number of pieces on bottom half of y-axis

  # Normally it y-axis coordinates is found in warped_coordinates[1], but if image is rotated, its on the "0"-index.
  axis = 1 if rotations == 0 else 0

  # heuristic: count pieces of each color on bottom half to determine if it belongs to black or white
  b_pieces_bottom_half = 0
  w_pieces_bottom_half = 0
  for piece in pieces:

    warped_coordinates = get_warped_coordinates(matrix, (piece["pos_x"], piece["pos_y"]))
    if warped_coordinates[axis] > height / 2:
      if piece["piece_info"]["color"] == "black":
        b_pieces_bottom_half += 1
      else:
        w_pieces_bottom_half += 1

  if b_pieces_bottom_half > w_pieces_bottom_half:
    rotations += 2

  return rotations

def get_proper_piece_info(label):
    """
    Maps an integer label to its corresponding piece information.
    
    Args:
        label (int): The integer label representing the chess piece.
        
    Returns:
        dict: A dictionary containing the color and piece type.
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

    
def position_to_tile(assigned_square, rotation=0):
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

def process_pieces(image_rgb, pieces_results, all_square_corners,old_offset_vectors, rotation=0,):
    """
    Processes the chess pieces from the image and returns their positions.
    
    Args:
        image_rgb (numpy.ndarray): The RGB image with the board overlay.
        pieces_results: The results from the piece detection model.
        all_square_corners (list): A 2D list containing the corner points of each square.
        rotation (int): The rotation of the chessboard (0, 1, 2, or 3).
        
    Returns:
        list: A list of dictionaries containing piece information and their assigned tiles.
    """
    

    # extract pieces detection results
    pieces = pieces_results[0]
    boxes = pieces.boxes.xyxy.cpu().numpy()
    labels = pieces.boxes.cls.cpu().numpy().astype(int)
    scores = pieces.boxes.conf.cpu().numpy()
    names = pieces.names

    piece_centers = []

    # persistence across frames
    offset_vectors = old_offset_vectors
    if(offset_vectors == None):
        offset_vectors = []

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
                'piece_info': get_proper_piece_info(piece_label),
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

                offset_vectors.append((dx, dy))

        square_corners = all_square_corners[assigned_square[0]][assigned_square[1]]
        square_top_left = square_corners[0]
        square_bottom_right = square_corners[2]
        adjusted_center_x = (square_top_left[0] + square_bottom_right[0]) / 2
        adjusted_center_y = (square_top_left[1] + square_bottom_right[1]) / 2

        tile = position_to_tile(assigned_square, rotation=rotation)

        piece_info = {
            'piece_index': i,
            'piece_info': get_proper_piece_info(piece_label),
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
    if offset_vectors:
        avg_dx = sum(v[0] for v in offset_vectors) / len(offset_vectors)
        avg_dy = sum(v[1] for v in offset_vectors) / len(offset_vectors)
        print(f"Average offset: dx={avg_dx}, dy={avg_dy}")

        for index,piece in enumerate(piece_centers):
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
                    piece_centers[index]['tile'] = position_to_tile(assigned_square, rotation=rotation)
                else:
                    piece_centers[index]['tile'] = None

                # annotations for error fixes
                cv2.circle(image_rgb, (int(piece['adjusted_pos_x']), int(piece['adjusted_pos_y'])), radius=8, color=(255, 0, 0), thickness=-1)
                if piece['tile'] is not None:
                    cv2.putText(image_rgb, piece['tile'], (int(piece['adjusted_pos_x']) + 10, int(piece['adjusted_pos_y']) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for index,piece in enumerate(piece_centers):
        cv2.circle(image_rgb, (int(piece['pos_x']), int(piece['pos_y'])), radius=8, color=(0, 255, 0), thickness=-1)
        
    
    return piece_centers, offset_vectors


# VIDEO PROCESSING
import cv2
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point, box
import pandas as pd

def get_primitive_chess_notation(all_chess_positions, piece, old_position, new_position):
    """
    Generates the chess notation for a move.

    Args:
        all_chess_positions (set): The set of occupied positions before the move.
        piece (str): The piece being moved ('pawn', 'knight', 'bishop', etc.).
        old_position (str): The starting position (e.g., 'e2').
        new_position (str): The ending position (e.g., 'e4').

    Returns:
        str: The move in chess notation.
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
    is_capture = new_position in all_chess_positions

    # remove old_position from all_chess_positions
    all_chess_positions.discard(old_position)
    
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
    all_chess_positions.add(new_position)
    
    return notation

video_data = {
    path: {
        'moves_detected': [],
        'positions_per_frame': [],
        'chess_board_process_data': None,
        'all_chess_positions': set(),
        'offset_vectors': None,
        'first_move_black': False,
    }
    for path in video_paths
}

def annotate_frame_with_board(frame, board_corners):
    """
    Annotates the frame with the chessboard corners.

    Args:
        frame (numpy.ndarray): The video frame to annotate.
        board_corners (list): The list of board corner points.

    Returns:
        numpy.ndarray: The annotated frame.
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

def process_frame(frame, video_key):
    """
    Processes a single frame: processes pieces.

    Args:
        frame (numpy.ndarray): The video frame to process.

    Returns:
        tuple: (piece_centers, image_with_board)
    """
    chess_board_process_data = video_data[video_key]["chess_board_process_data"]
    image_with_board_prev, board_corners, square_centers, all_square_corners, M, inverse_M = chess_board_process_data
    
    image_with_board = annotate_frame_with_board(frame.copy(), board_corners)

    piecesResult = callPiecesInference(frame)
    #piecesResult[0].show()

    # store rotation etc.
    if 'rotations' not in video_data[video_key]:
        piece_centers, offset_vectors = process_pieces(image_with_board, piecesResult, all_square_corners, [], rotation=0)
        rotations = find_rotation(piece_centers, image_with_board, board_corners)
        video_data[video_key]["rotations"] = rotations
        video_data[video_key]["offset_vectors"] =  offset_vectors

    piece_centers, offset_vectors = process_pieces(frame, piecesResult, all_square_corners,video_data[video_key]["offset_vectors"],rotation=video_data[video_key]["rotations"])
    video_data[video_key]["offset_vectors"] =  offset_vectors

    return piece_centers, image_with_board

for video_key in video_paths:
    cap = cv2.VideoCapture(video_key)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()


    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1)  # process every 1 second
    frames_in_1_seconds = int(fps * 1)
    print(f"Video FPS: {fps}")
    print(f"Processing every {frame_interval} frames (every 1 second).")

    # read the first frame to perform chessboard inference
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    # perform chessboard inference once
    chessBoardResult = callBoardInference(frame)
    # process the board to get board information once
    chess_board_process_data = process_board(frame, chessBoardResult)
    video_data[video_key]["chess_board_process_data"] = chess_board_process_data

    image_with_board, board_corners, square_centers, all_square_corners, M, inverse_M = chess_board_process_data

    if board_corners is None:
        print("No board detected.")
        continue  # Skip to next video

    board_corners_points = [
        board_corners[0]['top-left'],
        board_corners[0]['top-right'],
        board_corners[0]['bottom-right'],
        board_corners[0]['bottom-left']
    ]
    board_polygon = Polygon(board_corners_points)
    video_data[video_key]["board_polygon"] = board_polygon

    # reset video to start processing all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    num_frames = 0
    previous_pieces = {}  # dictionary to store previous frame's piece positions
    banned_frames = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # end of video
            break
        num_frames += 1

        #if(num_frames < 14400):
        #    continue
      
        handsResult = callHandInference(frame)
     
        hands = handsResult[0]
        hand_over_board = False

        if hasattr(hands, 'boxes') and hands.boxes is not None:
            hand_boxes = hands.boxes.xyxy.cpu().numpy()
            for box_coords in hand_boxes:
                xmin, ymin, xmax, ymax = box_coords
                hand_polygon = box(xmin, ymin, xmax, ymax)
                
                # check if hand overlaps with board by at least 1%
                intersection_area = board_polygon.intersection(hand_polygon).area
                if intersection_area / board_polygon.area >= 0.01:
                    hand_over_board = True
                    print(f"Hand detected over the board on frame {num_frames}.")
                    break  

        if hand_over_board:
            # ban frames within one second
            start_frame = max(1, num_frames - frames_in_1_seconds)
            end_frame = num_frames + frames_in_1_seconds
            banned_frames.update(range(start_frame, end_frame + 1))
            print(f"Banned frames from {start_frame} to {end_frame} due to hand detection on frame {num_frames}")

        # check if current frame is at the interval and not banned
        if (num_frames == 1 or num_frames % frame_interval == 0):
            if num_frames in banned_frames:
                print(f"Skipping frame {num_frames} because it's banned.")
                continue 
            else:
                print(f"\nProcessing frame {num_frames}")
                piece_centers, image_with_board = process_frame(frame, video_key)

                #plt.imshow(cv2.cvtColor(image_with_board, cv2.COLOR_BGR2RGB))
                #plt.show()

                if piece_centers is None:
                    print(f"Skipping frame {num_frames} due to no pieces detected.")
                    continue  # Skip to next frame

                # convert piece_centers to a dictionary: tile -> piece_info
                current_pieces = {}
                for piece in piece_centers:
                    tile = piece['tile']
                    if tile is not None:
                        current_pieces[tile] = piece['piece_info']

                # initialize all_chess_positions from the first frame
                if len(video_data[video_key]["positions_per_frame"]) == 0:
                    video_data[video_key]["all_chess_positions"] = set(current_pieces.keys())
                    print("Initialized all_chess_positions with occupied tiles:")
                    print(video_data[video_key]["all_chess_positions"])

                #plt.imshow(image_with_board)
                #plt.show()
                
                # detect moves by comparing previous_pieces and current_pieces
                if previous_pieces:
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

                    print("Vacated Tiles:", vacated_tiles)
                    print("Occupied Tiles:", occupied_tiles)

                    # attempt to match vacated tiles to occupied tiles
                    matched_vacated = set()
                    matched_occupied = set()

                    if len(vacated_tiles) == 1 and len(occupied_tiles) == 1:
                        old_position = vacated_tiles.pop()
                        new_position = occupied_tiles.pop()
                    
                        if old_position != new_position:
                            piece_info = current_pieces.get(new_position)
                            piece = piece_info["piece"]
                           
                            notation = get_primitive_chess_notation(
                                video_data[video_key]["all_chess_positions"],
                                piece,
                                old_position,
                                new_position
                            )
                            print("Detected move:", notation)
                            if len(video_data[video_key]["moves_detected"]) == 0 and piece_info["color"] == "black":
                                print("Black move first")
                                video_data[video_key]["first_move_black"] = True
                            video_data[video_key]["moves_detected"].append(notation)
                    else:
                        if len(vacated_tiles)>2:
                            print("Skipping frame because hand is probably undetected over board (3+ pieces missing)")
                            continue;
                        # try to match vacated and occupied tiles based on piece info
                        # Only match one move to frame, so this boolean is needed for edge cases
                        match_found = False
                        for vacated in vacated_tiles:
                            if match_found:
                                break
                            previous_piece_info = previous_pieces.get(vacated)
                            for occupied in occupied_tiles:
                                if match_found:
                                    break
                                current_piece_info = current_pieces.get(occupied)
                                if previous_piece_info and current_piece_info:
                                    if (previous_piece_info["color"] == current_piece_info["color"] and
                                        previous_piece_info["piece"] == current_piece_info["piece"]):
                                        match_found = True
                                        old_position = vacated
                                        new_position = occupied
                                        piece = current_piece_info["piece"]
                                        notation = get_primitive_chess_notation(
                                            video_data[video_key]["all_chess_positions"],
                                            piece,
                                            old_position,
                                            new_position
                                        )
                                        if len(video_data[video_key]["moves_detected"]) == 0 and current_piece_info["color"] == "black":
                                            print("Black move first")
                                            video_data[video_key]["first_move_black"] = True
                                        print("Detected move:", notation)
                                        video_data[video_key]["moves_detected"].append(notation)
                                        matched_vacated.add(vacated)
                                        matched_occupied.add(occupied)
                                        break  # Break inner loop to avoid multiple matches

                        # remove matched tiles
                        vacated_tiles -= matched_vacated
                        occupied_tiles -= matched_occupied

                previous_pieces = current_pieces.copy()

                # save piece information for this frame
                video_data[video_key]["positions_per_frame"].append({
                    'frame_number': num_frames,
                    'pieces': current_pieces
                })

    cap.release()

print("\nAll Detected Moves:")
for video_key in video_paths:
    video_name = video_key.split("/")[-1]
    if(video_name == "Bonus Long Video Label.mp4"):
        video_name = '(Bonus)Long_video_student.mp4'
    print(f"Moves for video {video_name}:")
    for move in video_data[video_key]["moves_detected"]:
        print(move)
        
data_for_csv = []


for video_key in video_paths:
    video_name = video_key.split("/")[-1]
    moves = ""
    first_black = video_data[video_key]["first_move_black"]
    detected_moves = video_data[video_key]["moves_detected"]

    if first_black:
        moves = f"1... {detected_moves[0]}"
        move_number = 2
        idx = 1
        while idx < len(detected_moves):
            moves += f" {move_number}."
            moves += f" {detected_moves[idx]}"        
            idx += 1
            if idx < len(detected_moves):
                moves += f" {detected_moves[idx]}"   
                idx += 1
            move_number += 1
    
    else:
        
        moves = f"1. {detected_moves[0]}"
        idx = 1
        # If there is a black move following the first white move, it should be on the same "1." line
        if idx < len(detected_moves):
            moves += f" {detected_moves[idx]}"
            idx += 1
    
        move_number = 2
        while idx < len(detected_moves):
            moves += f" {move_number}."
            moves += f" {detected_moves[idx]}"       
            idx += 1
            if idx < len(detected_moves):
                moves += f" {detected_moves[idx]}"    
                idx += 1
            move_number += 1

    data_for_csv.append({"row_id": video_name, "output": moves})

submission_df = pd.DataFrame(data_for_csv)

submission_df.to_csv("result.csv", index=False, encoding="utf-8")

