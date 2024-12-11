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

def compute_motion_vectors(prev_frame, curr_frame, block_size=16, search_method='tss'):
    """
    Compute motion vectors using either Three Step Search or Diamond Search.
    
    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame (grayscale)
        block_size: Size of macroblocks (default: 16x16)
        search_method: 'tss' for Three Step Search or 'ds' for Diamond Search
    """
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
    h, w = curr_frame.shape
    mb_h = h // block_size
    mb_w = w // block_size
    vectors = np.zeros((mb_h, mb_w, 2), dtype=np.int32)
    
    # Pad frames for border blocks
    p_size = block_size + 16  # Search area padding
    prev_pad = cv2.copyMakeBorder(prev_frame, p_size, p_size, p_size, p_size, cv2.BORDER_REPLICATE)
    
    for i in range(mb_h):
        for j in range(mb_w):
            # Current block position
            y = i * block_size
            x = j * block_size
            
            # Extract current block
            curr_block = curr_frame[y:y+block_size, x:x+block_size]
            
            # Find motion vector
            if search_method == 'tss':
                dy, dx = three_step_search(prev_pad[y:y+block_size*3, x:x+block_size*3], 
                                        curr_block, block_size)
            else:  # Diamond search
                dy, dx = diamond_search(prev_pad[y:y+block_size*3, x:x+block_size*3], 
                                     curr_block, block_size)
                
            vectors[i, j] = [dy, dx]
            
    return vectors

def three_step_search(search_area, curr_block, block_size):
    """Three Step Search algorithm for block matching."""
    h, w = search_area.shape
    center_y, center_x = h//2, w//2
    step_size = 4
    min_cost = float('inf')
    best_dy, best_dx = 0, 0
    
    while step_size >= 1:
        for dy in [-step_size, 0, step_size]:
            for dx in [-step_size, 0, step_size]:
                y = center_y + dy
                x = center_x + dx
                if 0 <= y < h-block_size and 0 <= x < w-block_size:
                    cost = calculate_mad(search_area[y:y+block_size, x:x+block_size], 
                                      curr_block)
                    if cost < min_cost:
                        min_cost = cost
                        best_dy, best_dx = dy, dx
        
        center_y += best_dy
        center_x += best_dx
        step_size //= 2
        
    return best_dy-block_size, best_dx-block_size

def diamond_search(search_area, curr_block, block_size):
    """Diamond Search algorithm for block matching."""
    h, w = search_area.shape
    center_y, center_x = h//2, w//2
    
    # Large Diamond Search Pattern
    LDSP = [(0,-2), (-1,-1), (0,-1), (1,-1), (-2,0), (-1,0), (1,0), (2,0),
            (-1,1), (0,1), (1,1), (0,2)]
    
    # Small Diamond Search Pattern
    SDSP = [(0,-1), (-1,0), (0,0), (1,0), (0,1)]
    
    min_cost = float('inf')
    best_dy, best_dx = 0, 0
    
    # First use LDSP
    while True:
        best_cost = min_cost
        for dy, dx in LDSP:
            y = center_y + dy
            x = center_x + dx
            if 0 <= y < h-block_size and 0 <= x < w-block_size:
                cost = calculate_mad(search_area[y:y+block_size, x:x+block_size], 
                                  curr_block)
                if cost < min_cost:
                    min_cost = cost
                    best_dy, best_dx = dy, dx
                    
        if min_cost == best_cost:  # No improvement, switch to SDSP
            break
            
        center_y += best_dy
        center_x += best_dx
    
    # Finally use SDSP
    for dy, dx in SDSP:
        y = center_y + dy
        x = center_x + dx
        if 0 <= y < h-block_size and 0 <= x < w-block_size:
            cost = calculate_mad(search_area[y:y+block_size, x:x+block_size], 
                              curr_block)
            if cost < min_cost:
                min_cost = cost
                best_dy, best_dx = dy, dx
                
    return best_dy-block_size, best_dx-block_size

def calculate_mad(block1, block2):
    """Calculate Mean Absolute Difference between two blocks."""
    return np.mean(np.abs(block1.astype(float) - block2.astype(float)))

# def compute_motion_vectors(prev_gray, cur_gray, block_size=16, search_range=4):
#     """
#     Compute motion vectors using a brute force block matching (MAD).
#     Here we use the grayscale images directly.
#     """
#     scale_percent = 50  # Reduce size by 50%
#     width = int(prev_gray.shape[1] * scale_percent / 100)
#     height = int(prev_gray.shape[0] * scale_percent / 100)
#     dim = (width, height)

#     prev_gray_small = cv2.resize(prev_gray, dim, interpolation=cv2.INTER_AREA)
#     cur_gray_small = cv2.resize(cur_gray, dim, interpolation=cv2.INTER_AREA)

#     flow = cv2.calcOpticalFlowFarneback(
#         prev_gray_small, cur_gray_small, None,
#         pyr_scale=0.5, levels=2,
#         winsize=9, iterations=3,
#         poly_n=5, poly_sigma=1.2, flags=0
#     )

#     return flow

# def compute_motion_vectors(prev_y, curr_y, block_size=16):
#     """
#     Compute motion vectors using block-based Mean Absolute Difference (MAD).
#     Parameters:
#         - prev_y: Previous frame's Y component (grayscale).
#         - curr_y: Current frame's Y component (grayscale).
#         - block_size: Size of macroblocks (default: 16x16).
#     Returns:
#         - motion_vectors: Array of motion vectors (dx, dy) for each block.
#     """
#     h, w = curr_y.shape
#     # Motion vector array dimensions (number of blocks)
#     num_blocks_y = h // block_size
#     num_blocks_x = w // block_size
    
#     print(f"[DEBUG] Frame dimensions: {h}x{w}")
#     print(f"[DEBUG] Macroblock grid: {num_blocks_y}x{num_blocks_x} (YxX)")
#     motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=np.int32)
#     # Pad the frames to ensure consistent block sizes
#     padded_prev = np.pad(prev_y, ((0, block_size), (0, block_size)), mode='constant', constant_values=0)
#     padded_curr = np.pad(curr_y, ((0, block_size), (0, block_size)), mode='constant', constant_values=0)
#     for by in range(num_blocks_y):  # Iterate over blocks in the y-direction
#         for bx in range(num_blocks_x):  # Iterate over blocks in the x-direction
#             y = by * block_size
#             x = bx * block_size
#             block = padded_curr[y:y + block_size, x:x + block_size]
#             best_match = (0, 0)
#             min_mad = float('inf')
#             # print(f"[DEBUG] Processing MAD for block ({by}, {bx}) at ({y}, {x})...")
#             # Search for the best matching block in the padded previous frame
#             for dy in range(-block_size, block_size + 1, 4):  # Search range ±block_size
#                 for dx in range(-block_size, block_size + 1, 4):
#                     ref_y = y + dy
#                     ref_x = x + dx
#                     # Ensure search stays within valid bounds of the padded frame
#                     if 0 <= ref_y < padded_prev.shape[0] - block_size and 0 <= ref_x < padded_prev.shape[1] - block_size:
#                         ref_block = padded_prev[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
#                         if ref_block.shape == block.shape:
#                             mad = np.mean(np.abs(block - ref_block))
#                             if mad < min_mad:
#                                 min_mad = mad
#                                 best_match = (dx, dy)
            
#             motion_vectors[by, bx] = best_match  # Assign motion vector for the block
            
#         # print(f"[DEBUG] Processed row {by + 1}/{num_blocks_y} of macroblocks.")
#     print(f"[DEBUG] Motion vectors computed for the entire frame.")
#     return motion_vectors


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
        
        # motion_vectors = compute_motion_vectors(prev_y, curr_y, block_size)
        motion_vectors = compute_motion_vectors(prev_y, curr_y, block_size=16, search_method='tss')
        # motion_vectors = compute_motion_vectors(prev_y, curr_y, block_size=16, search_method='ds')
        
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