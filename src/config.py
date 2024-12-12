# Configuration settings
VIDEO_PATHS = [
    './src/inputs/sample_input_video.mp4',
]

MODEL_PATHS = {
    'pieces': './src/models/pieces-model.pt',
    'board': './src/models/board-model.pt',
    'hands': './src/models/hand-model.pt'
}

INFERENCE_SETTINGS = {
    'hands_confidence': 0.65,
    'frame_interval_seconds': 1  # Process every N seconds
} 