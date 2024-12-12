# Chess Move Extraction from Videos Using Deep Learning

<div align="center">
    <img src="https://github.com/user-attachments/assets/5e4a5970-6017-47c4-b3a5-bc41209ff9c8" 
         alt="Chess board with inference annotations on it" 
         width="500" />
</div>

This project focuses on automating the transcription of chess piece movements from a video feed. By leveraging image processing, deep learning, and machine learning techniques, it identifies the chessboard, detects individual pieces, and records the sequence of moves in standard chess notation.

## Overview

The goal of this project is to take a video of a chess game and automatically produce a list of moves in standard algebraic notation. This involves:

1. **Chessboard Detection**: Using a YOLO-based model to detect the chessboard and calculate its orientation.
2. **Piece Detection & Tracking**: Identifying and locating chess pieces on each square using a trained model.
3. **Hand Detection**: Filtering out frames where hands appear to ensure captured moves are from stable positions.
4. **Move Extraction**: Comparing consecutive frames and determining piece movements to produce a move list in standard chess notation.

## Features

- **Board Recognition**: Identifies the 8x8 grid and corrects for rotation.
- **Piece Classification**: Distinguishes between different pieces (pawn, knight, bishop, rook, queen, king) and colors (black/white).
- **Move Generation**: Outputs moves in algebraic notation, such as `1. e4 e5 2. Nf3 Nc6 ...`
- **Hand Detection**: Frames with a player’s hand over the board are excluded from move extraction to reduce noisy data.

## How It Works

1. **Preprocessing**: Each processed frame is passed through a YOLO model that detects the board polygon and pieces.
2. **Calibration and Rotation**: The system determines the board’s orientation and adjusts the mapping between image coordinates and chess squares.
3. **Piece Localization**: Pieces detected in a frame are mapped onto their corresponding squares.
4. **Move Determination**: By comparing consecutive frames, the system infers moves. If a piece disappears from one square and appears on another, it’s considered a move.
5. **Notation Generation**: Moves are appended in standard notation. For example, a white pawn moving from e2 to e4 is recorded simply as `e4`, and a white knight moving from g1 to f3 is noted as `Nf3`.

## Models

To complete the project, three models based on YOLOv11 were trained, and they are being included in this repository for usage.

- **Board Model**: Trained YOLOv11m-seg instance segmentation model to detect the chessboard polygon.
- **Pieces Model**: YOLOv11l object detection model for piece detection and classification.
  - Achieved precision of 99.37% and recall of 99.37% on untrained data. Better performance on some chess boards than others, so finetuning is recommended if you get poor results.
- **Hand Model**: YOLOv11l model to detect hands and exclude frames with interference.
  - Achieved precision of 97.53% and recall of 96.48% on untrained data. Can be replaced with hand/arm detection model instead, if instances of only the arm being over the board are in videos.

The datasets for these models are described at the bottom of this README.

If you want more accurate results, you can retrain these models after your own wishes.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/oliverfrost1/chess-video-move-detection.git
   cd chess-video-move-detection
   ```
2. **Install Dependencies**:  
   Make sure you have Python 3.11 (or a compatible version) and `pip` installed. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The project uses [Ultralytics YOLO](https://docs.ultralytics.com/) for object detection, as well as OpenCV, NumPy, Shapely, and other common Python libraries.

3. **Place Models and Input Videos**:
   - Ensure the pretrained models (`board-model.pt`, `pieces-model.pt`, `hand-model.pt`) are in `src/models`. (They should be by default.)
   - Place your input video in `src/inputs` or use the provided `sample_input_video.mp4` as an example.

## Usage

1. **Run the Main Script**:
   From the root of the directory run:

   ```bash
   python main.py
   ```

Remember to input your inputs in `config.py`, otherwise the correct inputs will not be targeted.

2. **Output**:
   After processing, the moves will be printed out into the `results.csv` file. It will look like this:

   ```csv
   row_id,output
   sample_input_video.mp4,1... Qh4 2. g3
   sample_input_video2.mp4,1. g3 Qh4 2. h4 Ng5
   ```

  **Note:** The three dots after the first "1" is due to black starting in the video. 
  
   If you need strict PGN formating, I would suggest you change the code yourself. When using strict PGN formatting, it can cause problems if illegal moves are made or if illegal moves are registered. 

## Showcase

**1. Detected Chessboard Area**

Below an example of the chessboard corners being detect can be seen.

<div align="center">
    <img src="https://github.com/user-attachments/assets/03d5aa92-9555-46b0-a98c-2b95ae0d9567" 
         alt="Chess board with inference annotations on it" 
         width="500" />
</div>



**2. Detected Pieces with Assigned Tiles**

Below an example of a full board along with all labels can be seen. 
- The green dot is the detected center of the chess pieces and the red dot is the angle and height adjusted midpoint of the piece. The square that the red dot is on is the one where the pieces location will be registered.
- The small blue dots are the board square centers. These are calculated from the polygon of the board.
- The big blue dot is the middle of a square with a piece detected on it.
  
<div align="center">
    <img src="https://github.com/user-attachments/assets/5e4a5970-6017-47c4-b3a5-bc41209ff9c8" 
         alt="Chess board with inference annotations on it" 
         width="500" />
</div>




## Project Structure

```
src/
├─ video_processing.py     # Handles video frame iteration, hand detection, and orchestrating board/piece processors
├─ config.py               # Configuration file for input paths, model paths, inference settings
├─ piece_processing.py     # Responsible for piece detection, mapping pieces to squares
├─ utils.py                # Data structures and utility classes
├─ board_processing.py     # Processes the board, finds corners, computes perspective transforms
├─ chess_notation.py       # Converts detected moves into standard algebraic chess notation
├─ inputs/
│  ├─ sample_input_video.mp4
└─ models/
   ├─ board-model.pt
   ├─ pieces-model.pt
   └─ hand-model.pt
```

**Note**: The `main.py` file and final output files are not shown above but are at the root.

## Configuration

All configuration options, including confidence thresholds, model paths, and frame processing intervals, can be adjusted in `config.py`.

## Notes and Limitations

- **Lighting and Glare**: Varying lighting conditions can affect detection accuracy.
- **Camera Angle and Stability**: Extreme angles or a moving camera can make board detection less reliable.
- **Partial Occlusions**: A hand model is used to filter out frames with disturbances, but if pieces are partially obscured, the move detection might be less accurate.
- **Rotation Detection**: Rotation detection can fail in instances where there are more white pieces on the wrong side of the board. This could be fixed in the future by using OCR to detect the numbers outside the board.

## Datasets for Training

Datasets used for training:

- Board: [Simple own labelled data](https://universe.roboflow.com/chess-r5mgx/chess_board_actual_shape/model).
- Chess Pieces: [merge_chess](https://universe.roboflow.com/chess-r5mgx/merge_chess-zqlpy) and own data [own refinement data](https://universe.roboflow.com/chess-r5mgx/chess_pieces_refinement/dataset/2).
- Hands: [Egohands](http://vision.soic.indiana.edu/projects/egohands/) and own labelled hands (not published for privacy reasons).

## License

This project is licensed under the [MIT License](LICENSE).
