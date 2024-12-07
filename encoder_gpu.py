import numpy as np
import cv2
import os
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid TensorFlow reserving all GPU memory upfront
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[DEBUG] TensorFlow is configured to use GPU.")
    except RuntimeError as e:
        print(f"[ERROR] Failed to set GPU: {e}")
else:
    print("[ERROR] No GPU found! Running on CPU instead.")


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

def compute_motion_vectors_tf(prev_y, curr_y, block_size=16, search_range=16):
    """
    Compute motion vectors using block-based Mean Absolute Difference (MAD) on GPU.
    Parameters:
        - prev_y: Previous frame's Y component (grayscale).
        - curr_y: Current frame's Y component (grayscale).
        - block_size: Size of macroblocks (default: 16x16).
        - search_range: Range of motion vector search (default: ±16 pixels).
    Returns:
        - motion_vectors: Tensor of motion vectors (dx, dy) for each block.
    """
    h, w = curr_y.shape

    # Number of macroblocks
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size

    print(f"[DEBUG] Frame dimensions: {h}x{w}")
    print(f"[DEBUG] Macroblock grid: {num_blocks_y}x{num_blocks_x} (YxX)")

    # Pad frames to handle edge cases
    padded_prev = tf.pad(prev_y, [[0, block_size], [0, block_size]], mode="CONSTANT")
    padded_curr = tf.pad(curr_y, [[0, block_size], [0, block_size]], mode="CONSTANT")

    print(f"[DEBUG] Padding applied to frames.")

    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=np.int32)

    # Function to compute MAD for a single macroblock
    def compute_mad(y, x):
        block = padded_curr[y:y + block_size, x:x + block_size]

        if block.shape != (block_size, block_size):  # Ensure block is valid
            return tf.constant([0, 0], dtype=tf.int32)

        # Define the search range
        min_mad = float("inf")
        best_match = (0, 0)

        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                ref_y = y + dy
                ref_x = x + dx

                # Ensure indices stay within valid bounds
                if (
                    ref_y >= 0
                    and ref_y + block_size <= padded_prev.shape[0]
                    and ref_x >= 0
                    and ref_x + block_size <= padded_prev.shape[1]
                ):
                    ref_block = padded_prev[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

                    if ref_block.shape == (block_size, block_size):  # Ensure shapes match
                        mad = tf.reduce_mean(tf.abs(tf.cast(block, tf.float32) - tf.cast(ref_block, tf.float32)))

                        if mad < min_mad:
                            min_mad = mad
                            best_match = (dx, dy)

        return tf.constant(best_match, dtype=tf.int32)

    # Compute motion vectors for each macroblock
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y = by * block_size
            x = bx * block_size
            print(f"[DEBUG] Processing MAD for block ({by}, {bx}) at ({y}, {x})...")
            motion_vector = compute_mad(y, x)
            motion_vectors[by, bx] = motion_vector.numpy()  # Convert Tensor to NumPy

        print(f"[DEBUG] Processed row {by + 1}/{num_blocks_y} of macroblocks.")

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

def process_video(frames, block_size=16, threshold=2, compute_motion_vectors=None):
    """
    Process the video to segment each frame into foreground and background macroblocks.
    Parameters:
        - frames: List of video frames (RGB format).
        - block_size: Size of macroblocks (default: 16x16).
        - threshold: Threshold for motion vector magnitude.
        - compute_motion_vectors: Function to compute motion vectors (can be GPU-optimized or CPU-based).
    Returns:
        - segmentations: List of binary masks (1: foreground, 0: background) for each frame.
    """
    if compute_motion_vectors is None:
        raise ValueError("A motion vector computation function must be provided.")

    segmentations = []
    prev_y = rgb_to_yuv(frames[0])[:, :, 0]  # Extract Y component from first frame

    print(f"[DEBUG] Processing {len(frames)} frames...")
    for i in range(1, len(frames)):
        print(f"[DEBUG] Processing frame {i}/{len(frames) - 1}...")
        curr_frame = rgb_to_yuv(frames[i])
        curr_y = curr_frame[:, :, 0]  # Y component

        # Compute motion vectors
        motion_vectors = compute_motion_vectors(prev_y, curr_y, block_size)
        print(f"[DEBUG] Motion vectors computed for frame {i}.")

        # Segment foreground and background
        segmentation = segment_frame(motion_vectors, threshold)
        segmentations.append(segmentation)

        prev_y = curr_y  # Update previous frame

    print(f"[DEBUG] Segmentation completed for all frames.")
    return segmentations


def visualize_segmentation(frames, segmentations, block_size=16):
    """
    Visualize the segmentation by displaying foreground and background regions separately.
    Parameters:
        - frames: List of video frames (RGB format).
        - segmentations: List of binary masks (1: foreground, 0: background) for each frame.
        - block_size: Size of macroblocks.
    """
    for i, segmentation in enumerate(segmentations):
        frame = frames[i + 1].copy()  # Skip first frame
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
        if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit visualization
            break

    cv2.destroyAllWindows()

# Example usage
sample = "car"
file_path = f"960x540/{sample}.rgb"
width, height = 960, 540

if os.path.exists(file_path):
    frames, total_frames = read_rgb_video(file_path, width, height)
    print(f"Loaded {total_frames} frames.")

    # Use the GPU-optimized motion vector computation
    segmentations = process_video(frames, block_size=16, threshold=2, compute_motion_vectors=compute_motion_vectors_tf)

    # Visualize the segmented video
    visualize_segmentation(frames, segmentations)
else:
    print("Video file not found. Please check the file path.")
