
import cv2
import numpy as np
from ultralytics import YOLO

import logging

logger = logging.getLogger(__name__)

class BoardProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def order_points(self, pts):
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

    def process_board(self, image: np.ndarray):
        """
        Processes the chessboard from the image and returns the board information,
        including the corners of each square. Also overlays the mask on the image.
        """
        if image is None:
            logger.error("Image could not be loaded.")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]

        board_results = self.model.predict(image, save=False, retina_masks=True, verbose=False)
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
                    logger.warning(f"Mask {idx+1}: No contours found.")
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
                contour_points = self.order_points(contour_points)

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
                logger.info(f"Mask {idx+1}: Detected corners - {all_corners[idx]}")

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
            logger.error("No masks found in the board results.")
            return None

        # transform all square corners to be a 2D array based on a max length of 8
        number = 8
        new_square_corners = []
        # check if the number of squares is exactly 64 (8x8)
        expected_squares = number * number
        actual_squares = len(all_square_corners)
        if actual_squares != expected_squares:
            logger.warning(f"Expected {expected_squares} squares, but got {actual_squares}.")

        # transform the flat list into a 2D list (8x8)
        for i in range(actual_squares):
            row_index = i // number  
            if row_index >= len(new_square_corners):
                new_square_corners.append([])
            # append the current square's corners to the appropriate row
            new_square_corners[row_index].append(all_square_corners[i])

        return image_rgb, all_corners, new_square_corners, M, inverse_M

    def get_warped_coordinates(self, transform_matrix, original_point):
        # convert the original point to coords
        original_point_homogeneous = np.array([*original_point, 1], dtype='float32')

        warped_point_homogeneous = transform_matrix @ original_point_homogeneous

        # convert back to Cartesian coordinates
        warped_point = warped_point_homogeneous[:2] / warped_point_homogeneous[2]

        return warped_point

    def find_rotation(self, pieces, board_rgb, board_corners):
        '''
        Returns how many clockwise 90 degree rotations that should be made to get to correct state.
        That is how many counter-clockwise 90 degree rotations the chess board is rotated.
        '''
        rotations = 0

        # first find the 90 degree orientation. If chess board is rotated 90 degrees 
        # a white square is colored instead. Program finds this by finding an empty square,
        # then deciding whether its white or colored by comparing its brightness
        # to its also empty neighboring square. From coordinates of square determine whether
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
                for y1, y2 in [(y, y+1) for y in range(0, 7)]:
                    if not ((x, y1) in taken_spots or (x, y2) in taken_spots):
                        squares = ((x, y1), (x, y2))
                        break

        if not squares:
            logger.warning("Could not find empty adjacent squares to determine rotation.")
            return rotations

        gray_image = cv2.cvtColor(board_rgb, cv2.COLOR_RGB2GRAY)

        # warp image for a top-down view to divide board into squares easier

        # preparations for built-in cv2.warpPerspective
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

        # Adjust indices to access correct rows and columns
        y1, x1 = squares[0]
        y2, x2 = squares[1]

        first_square = warped_image[y1*square_size:(y1+1)*square_size, x1*square_size:(x1+1)*square_size]
        avg_brightness_first = np.mean(first_square)

        second_square = warped_image[y2*square_size:(y2+1)*square_size, x2*square_size:(x2+1)*square_size]
        avg_brightness_second = np.mean(second_square)

        # even sum of coordinates for a square means it should be white
        if ((squares[0][0] + squares[0][1]) % 2) == 0: # sum even so it should be white
            if avg_brightness_first < avg_brightness_second: # first square should be, but is not, white. So it's rotated
                rotations += 1
        else: 
            if avg_brightness_first > avg_brightness_second: # first square is, but should not be, white. So it's rotated
                rotations += 1

        # Now determine 180 degree orientation based on number of pieces on bottom half of y-axis

        # Normally the y-axis coordinates is found in warped_coordinates[1], but if image is rotated, its on the "0"-index.
        axis = 1 if rotations == 0 else 0

        # heuristic: count pieces of each color on bottom half to determine if it belongs to black or white
        b_pieces_bottom_half = 0
        w_pieces_bottom_half = 0
        for piece in pieces:
            warped_coordinates = self.get_warped_coordinates(matrix, (piece["pos_x"], piece["pos_y"]))
            if warped_coordinates[axis] > height / 2:
                if piece["piece_info"]["color"] == "black":
                    b_pieces_bottom_half += 1
                else:
                    w_pieces_bottom_half += 1

        if b_pieces_bottom_half > w_pieces_bottom_half:
            rotations += 2

        return rotations