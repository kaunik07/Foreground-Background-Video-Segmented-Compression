import numpy as np
import cv2
import os


def read_rgb_video(file_path, width, height):
    """Reads the .RGB video file and returns a list of frames."""
    frame_size = width * height * 3
    file_size = os.path.getsize(file_path)
    frame_count = file_size // frame_size
    print(f"Calculated Frame Count: {frame_count}")
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    if len(raw_data) % frame_size != 0:
        print("Warning: File size does not perfectly match frame dimensions.")
    frames = [
        np.frombuffer(raw_data[i * frame_size:(i + 1) * frame_size], dtype=np.uint8).reshape((height, width, 3))
        for i in range(frame_count)
    ]
    return frames, frame_count


def rgb_to_yuv(frame):
    """Convert an RGB frame to YUV color space."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)


def compute_motion_vectors(prev_y, curr_y, block_size=16):
    """
    Compute motion vectors using block-based Mean Absolute Difference (MAD).
    Parameters:
        - prev_y: Previous frame's Y component (grayscale).
        - curr_y: Current frame's Y component (grayscale).
        - block_size: Size of macroblocks (default: 16x16).
    Returns:
        - motion_vectors: Array of motion vectors (dx, dy) for each block.
    """
    h, w = curr_y.shape
    # Motion vector array dimensions (number of blocks)
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    print(f"[DEBUG] Frame dimensions: {h}x{w}")
    print(f"[DEBUG] Macroblock grid: {num_blocks_y}x{num_blocks_x} (YxX)")
    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=np.int32)
    # Pad the frames to ensure consistent block sizes
    padded_prev = np.pad(prev_y, ((0, block_size), (0, block_size)), mode='constant', constant_values=0)
    padded_curr = np.pad(curr_y, ((0, block_size), (0, block_size)), mode='constant', constant_values=0)
    for by in range(num_blocks_y):  # Iterate over blocks in the y-direction
        for bx in range(num_blocks_x):  # Iterate over blocks in the x-direction
            y = by * block_size
            x = bx * block_size
            block = padded_curr[y:y + block_size, x:x + block_size]
            best_match = (0, 0)
            min_mad = float('inf')
            # print(f"[DEBUG] Processing MAD for block ({by}, {bx}) at ({y}, {x})...")
            # Search for the best matching block in the padded previous frame
            for dy in range(-block_size, block_size + 1, 4):  # Search range ±block_size
                for dx in range(-block_size, block_size + 1, 4):
                    ref_y = y + dy
                    ref_x = x + dx
                    # Ensure search stays within valid bounds of the padded frame
                    if 0 <= ref_y < padded_prev.shape[0] - block_size and 0 <= ref_x < padded_prev.shape[1] - block_size:
                        ref_block = padded_prev[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        if ref_block.shape == block.shape:
                            mad = np.mean(np.abs(block - ref_block))
                            if mad < min_mad:
                                min_mad = mad
                                best_match = (dx, dy)
            
            motion_vectors[by, bx] = best_match  # Assign motion vector for the block
            
        # print(f"[DEBUG] Processed row {by + 1}/{num_blocks_y} of macroblocks.")
    print(f"[DEBUG] Motion vectors computed for the entire frame.")
    return motion_vectors


def segment_frame(motion_vectors, threshold=2):
    """
    Segment macroblocks into foreground and background based on motion vectors.
    Parameters:
        - motion_vectors: Array of motion vectors (dx, dy) for each block.
        - threshold: Threshold for motion vector magnitude to classify as foreground.
    Returns:
        - segmentation: Binary mask with foreground (1) and background (0) blocks.
    """
    magnitudes = np.linalg.norm(motion_vectors, axis=2)
    segmentation = (magnitudes > threshold).astype(np.uint8)
    return segmentation
def process_video(frames, block_size=16, threshold=2):
    """
    Process the video to segment each frame into foreground and background macroblocks.
    Parameters:
        - frames: List of video frames (RGB format).
        - block_size: Size of macroblocks (default: 16x16).
        - threshold: Threshold for motion vector magnitude.
    Returns:
        - segmentations: List of binary masks (1: foreground, 0: background) for each frame.
    """
    segmentations = []
    prev_y = rgb_to_yuv(frames[0])[:, :, 0]  # Extract Y component from first frame
    print(f"[DEBUG] Processing {len(frames)} frames...")
    for i in range(1, len(frames)):
        print(f"[DEBUG] Processing frame {i}/{len(frames) - 1}...")
        curr_frame = rgb_to_yuv(frames[i])
        curr_y = curr_frame[:, :, 0]  # Y component
        
        motion_vectors = compute_motion_vectors(prev_y, curr_y, block_size)
        
        print(f"[DEBUG] Motion vectors computed for frame {i}.")
        segmentation = segment_frame(motion_vectors, threshold)
        segmentations.append(segmentation)
        prev_y = curr_y  # Update previous frame
    print(f"[DEBUG] Segmentation completed for all frames.")
    return segmentations


def visualize_segmentation(frames, segmentations, block_size=16):
    """
    Visualize the segmentation by displaying foreground and background regions separately.
    Allows play/pause functionality using the spacebar, next frame (n), and previous frame (p).
    Automatically resizes the windows to the video frame size.
    
    Parameters:
        - frames: List of video frames (RGB format).
        - segmentations: List of binary masks (1: foreground, 0: background) for each frame.
        - block_size: Size of macroblocks.
    """
    # Create OpenCV windows
    cv2.namedWindow("Foreground", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Background", cv2.WINDOW_NORMAL)
    # Resize windows to match frame size
    height, width, _ = frames[0].shape
    cv2.resizeWindow("Foreground", width, height)
    cv2.resizeWindow("Background", width, height)
    is_playing = False  # Start in paused state
    current_frame = 1  # Track the current frame index (start from the second frame)
    print(f"[DEBUG] Loaded {len(frames)} frames.")
    print(f"[DEBUG] Loaded {len(segmentations)} segmentations.")
    print("[DEBUG] Video is paused. Click on the 'Foreground' or 'Background' window and press SPACE to start playing.")
    print("[DEBUG] Use 'n' for next frame, 'p' for previous frame, and 'q' to exit.")
    while True:
        if is_playing and current_frame <= len(segmentations):
            # Process the current frame
            segmentation = segmentations[current_frame - 1]  # Align segmentation with frame
            frame = frames[current_frame].copy()  # Start from the second frame
            foreground = np.zeros_like(frame)
            background = np.zeros_like(frame)
            h, w = segmentation.shape
            for y in range(h):
                for x in range(w):
                    if segmentation[y, x] == 1:  # Foreground block
                        foreground[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = \
                            frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
                    else:  # Background block
                        background[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = \
                            frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            # Display foreground and background
            cv2.imshow("Foreground", foreground)
            cv2.imshow("Background", background)
            print(f"[DEBUG] Displaying frame {current_frame}/{len(frames)}")
            current_frame += 1  # Move to the next frame
            if current_frame > len(segmentations):
                print("[DEBUG] End of video reached. Pausing playback.")
                is_playing = False  # Pause when the video ends
                current_frame = 1  # Reset to the second frame
        # Wait for user input
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):  # Spacebar toggles play/pause
            is_playing = not is_playing
            print("[DEBUG] Toggled play/pause:", "Playing" if is_playing else "Paused")
        elif key == ord('n'):  # Next frame (step forward)
            is_playing = False
            if current_frame < len(segmentations):
                current_frame += 1
                print(f"[DEBUG] Step forward to frame {current_frame}")
        elif key == ord('p'):  # Previous frame (step backward)
            is_playing = False
            if current_frame > 1:
                current_frame -= 1
                print(f"[DEBUG] Step backward to frame {current_frame}")
        elif key == ord('q'):  # Quit the visualization
            print("[DEBUG] Exiting visualization.")
            break
        # Process the current frame in step mode
        if not is_playing:
            segmentation = segmentations[current_frame - 1]
            frame = frames[current_frame].copy()
            foreground = np.zeros_like(frame)
            background = np.zeros_like(frame)
            h, w = segmentation.shape
            for y in range(h):
                for x in range(w):
                    if segmentation[y, x] == 1:  # Foreground block
                        foreground[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = \
                            frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
                    else:  # Background block
                        background[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = \
                            frame[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            # Display in step mode
            cv2.imshow("Foreground", foreground)
            cv2.imshow("Background", background)
    cv2.destroyAllWindows()
    

# Example usage
sample = "car"
file_path = f"960x540/{sample}.rgb"
width, height = 960, 540
if os.path.exists(file_path):
    frames, total_frames = read_rgb_video(file_path, width, height)
    print(f"Loaded {total_frames} frames.")
    segmentations = process_video(frames)
    visualize_segmentation(frames, segmentations)
else:
    print("Video file not found. Please check the file path.")