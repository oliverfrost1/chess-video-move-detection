import logging
from src.config import VIDEO_PATHS, MODEL_PATHS
from src.board_processing import BoardProcessor
from src.piece_processing import PieceProcessor
from src.chess_notation import NotationGenerator
from src.video_processing import VideoProcessor
import pandas as pd

logging.basicConfig(level=logging.INFO)

def main():
    data_for_csv = []
    for video_path in VIDEO_PATHS:
        board_processor = BoardProcessor(MODEL_PATHS['board'])
        piece_processor = PieceProcessor(MODEL_PATHS['pieces'])
        notation_generator = NotationGenerator()

        video_processor = VideoProcessor(video_path, board_processor, piece_processor, notation_generator)
        moves_notation = video_processor.process_video()

        video_name = video_path.split('/')[-1]
        data_for_csv.append({"row_id": video_name, "output": moves_notation})

    submission_df = pd.DataFrame(data_for_csv)
    submission_df.to_csv("result.csv", index=False, encoding="utf-8")
    logging.info("Processing complete. Results saved to result.csv")

if __name__ == "__main__":
    main()