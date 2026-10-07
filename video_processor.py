import os
import cv2
import tempfile
import numpy as np
from PIL import Image
from typing import List, Tuple

def extract_frames_at_1fps(video_source, max_seconds: int = 10) -> List[Tuple[int, Image.Image]]:
    """
    Extracts frames at 1 frame per second (1 FPS) from a video file path or uploaded file buffer.
    
    Returns:
        List of tuples: (second_timestamp, PIL.Image)
    """
    temp_file_path = None
    
    # Check if video_source is an uploaded file buffer from Streamlit
    if hasattr(video_source, "read"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(video_source.read())
        temp_file.flush()
        temp_file.close()
        video_path = temp_file.name
        temp_file_path = video_path
    elif isinstance(video_source, str) and os.path.exists(video_source):
        video_path = video_source
    else:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    frames = []
    num_seconds = min(max_seconds, int(video_duration) if video_duration >= 1 else 1)

    for sec in range(num_seconds):
        target_frame_num = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append((sec, pil_img))
        else:
            break

    cap.release()
    
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except OSError:
            pass

    return frames
