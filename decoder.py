import sys
import numpy as np
import cv2
import wave
import pyaudio
from scipy.fftpack import idct
import time

try:
    import torch
    USE_TORCH = torch.backends.mps.is_available()  # Check for Metal support
except ImportError:
    USE_TORCH = False
from scipy.fftpack import idct
from functools import lru_cache

@lru_cache(maxsize=2)
def get_block_positions(total_blocks, blocks_per_row, block_size=8):
    """Cache block positions to avoid recalculation"""
    block_rows = np.arange(total_blocks) // blocks_per_row
    block_cols = np.arange(total_blocks) % blocks_per_row
    return block_rows * block_size, block_cols * block_size

def block_idct_batch(coeffs):
    """Vectorized IDCT for batch of blocks"""
    if USE_TORCH:
        # Convert to float32 for MPS compatibility
        coeffs = coeffs.astype(np.float32)
        device = torch.device("mps")
        coeffs_torch = torch.from_numpy(coeffs).to(device)
        
        # Implement IDCT using DCT matrix multiplication
        N = coeffs.shape[-1]
        n = torch.arange(N, dtype=torch.float32, device=device)
        k = n.reshape(-1, 1)
        
        # Create DCT matrix with proper tensor types
        dct_mat = torch.cos(torch.pi * (2*n + 1) * k / (2*N))
        scale = torch.sqrt(torch.tensor(2.0/N, dtype=torch.float32, device=device))
        dct_mat *= scale
        dct_mat[0] *= 1/torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))
        
        # Apply IDCT to both dimensions
        result = torch.matmul(dct_mat.T, torch.matmul(coeffs_torch, dct_mat))
        return result.cpu().numpy()
    else:
        # Fall back to scipy's IDCT
        # return cv2.idct(coeffs)
        return idct(idct(coeffs, axis=2, norm='ortho'), axis=1, norm='ortho')

def reconstruct_frame_fast(frame_blocks, n1, n2, padded_width, padded_height, original_width, original_height):
    block_size = 8
    blocks_per_row = padded_width // block_size
    total_blocks = len(frame_blocks)
    
    # Prepare arrays - explicitly use float32
    block_types = np.array([block[0] for block in frame_blocks])
    r_coeffs = np.stack([block[1] for block in frame_blocks]).astype(np.float32)
    g_coeffs = np.stack([block[2] for block in frame_blocks]).astype(np.float32)
    b_coeffs = np.stack([block[3] for block in frame_blocks]).astype(np.float32)

    # Vectorized dequantization
    q_steps = np.where(block_types == 1, 2.0**n1, 2.0**n2)[:, None, None].astype(np.float32)
    
    # Process all channels simultaneously
    coeffs = np.stack([r_coeffs, g_coeffs, b_coeffs])  # Shape: [3, blocks, 8, 8]
    dct_blocks = coeffs * q_steps
    
    # Batch IDCT
    blocks = np.clip(block_idct_batch(dct_blocks), 0, 255)
    
    # Get cached block positions
    y_coords, x_coords = get_block_positions(total_blocks, blocks_per_row)
    
    # Initialize output and place blocks
    reconstructed = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    for i in range(total_blocks):
        y, x = y_coords[i], x_coords[i]
        reconstructed[y:y+block_size, x:x+block_size] = blocks[:, i].transpose(1, 2, 0)
    
    return reconstructed[:original_height, :original_width]

def read_cmp_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    print("Read compressed data from", filename)

    # Parse header
    n1, n2 = map(int, lines[0].split())
    data_lines = lines[1:]
    
    print("Parsing data...")
    # Concatenate all data lines into a single string
    data_str = ' '.join(data_lines)

    # Convert the concatenated string into a NumPy array of integers
    data_array = np.fromstring(data_str, sep=' ', dtype=int)

    # Define block and frame parameters
    block_size = 8
    coeffs_per_channel = block_size * block_size  # 64
    ints_per_block = 1 + 3 * coeffs_per_channel  # 1 block_type + 64*3 coefficients = 193

    # Calculate total number of blocks
    total_blocks = data_array.size // ints_per_block
    data_array = data_array[:total_blocks * ints_per_block]  # Truncate any extra data

    print("Total blocks:", total_blocks)
    print("Reshaping data array...")
    # Reshape data_array to have one block per row
    blocks = data_array.reshape(total_blocks, ints_per_block)

    # Known dimensions
    original_width = 960
    original_height = 540
    # Calculate padded dimensions
    padded_width = ((original_width + 15) // 16) * 16
    padded_height = ((original_height + 15) // 16) * 16

    blocks_per_row = padded_width // block_size
    blocks_per_col = padded_height // block_size
    blocks_per_frame = blocks_per_row * blocks_per_col
    frame_count = total_blocks // blocks_per_frame

    frames = []
    for i in range(frame_count):
        print("Reading frame", i+1, "/", frame_count)
        frame_data = blocks[i * blocks_per_frame : (i + 1) * blocks_per_frame]
        frame_blocks = []
        for block_data in frame_data:
            block_type = block_data[0]
            r_coeffs = block_data[1:65].reshape((block_size, block_size))
            g_coeffs = block_data[65:129].reshape((block_size, block_size))
            b_coeffs = block_data[129:193].reshape((block_size, block_size))
            frame_blocks.append((block_type, r_coeffs, g_coeffs, b_coeffs))
        frames.append(frame_blocks)

    return n1, n2, frames, padded_width, padded_height, original_width, original_height

def playback(decoded_frames):
    """
    Function to play back decoded frames with controls:
    - Spacebar: Play/Pause
    - 'n': Next frame
    - 'p': Previous frame
    """
    total_frames = len(decoded_frames)
    current_frame_idx = 0
    paused = True  # Start paused

    print("Playback controls: 'Spacebar' to Play/Pause, 'n' for Next frame, 'p' for Previous frame, 'q' to Quit.")

    while True:
        if not paused:
            frame = decoded_frames[current_frame_idx]
            cv2.imshow("Decoded Video", frame)
            current_frame_idx = (current_frame_idx + 1) % total_frames
        else:
            frame = decoded_frames[current_frame_idx]
            cv2.imshow("Decoded Video", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar toggles pause/play
            paused = not paused
        elif key == ord('n'):
            paused = True
            current_frame_idx = (current_frame_idx + 1) % total_frames
            frame = decoded_frames[current_frame_idx]
            cv2.imshow("Decoded Video", frame)
        elif key == ord('p'):
            paused = True
            current_frame_idx = (current_frame_idx - 1) % total_frames
            frame = decoded_frames[current_frame_idx]
            cv2.imshow("Decoded Video", frame)
        elif key == ord('r'):  # Reset playback to the beginning
            paused = True
            current_frame_idx = 0
            frame = decoded_frames[current_frame_idx]
            cv2.imshow("Decoded Video", frame)

    cv2.destroyAllWindows()

def playback_with_audio(decoded_frames, audio_file, fps=30):
    """
    Playback function to synchronize decoded video frames with audio, including seeking.
    """
    # Initialize audio
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    
    # Audio stream setup
    audio_stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    # Synchronization setup
    total_frames = len(decoded_frames)
    frame_interval = 1 / fps  # Time per frame
    audio_rate = wf.getframerate()
    audio_bytes_per_frame = wf.getnchannels() * wf.getsampwidth()
    audio_frames_per_video_frame = int(audio_rate / fps)

    start_time = time.time()
    frame_index = 0
    paused = False

    print("Playback controls: 'Spacebar' to Play/Pause, 'n' for Next frame, 'p' for Previous frame, 'q' to Quit.")

    while True:
        elapsed_time = time.time() - start_time

        if not paused:
            # Calculate current frame index based on elapsed time
            expected_frame_index = int(elapsed_time / frame_interval)

            if frame_index < total_frames and frame_index <= expected_frame_index:
                # Play video frame
                frame = decoded_frames[frame_index]
                cv2.imshow("Decoded Video", frame)

                # Play audio corresponding to the frame
                audio_start = frame_index * audio_frames_per_video_frame
                wf.setpos(audio_start)
                audio_data = wf.readframes(audio_frames_per_video_frame)
                if audio_data:
                    audio_stream.write(audio_data)
                
                frame_index += 1

        # Handle keyboard input
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit playback
            break
        elif key == ord(' '):  # Spacebar to toggle pause/play
            paused = not paused
            if not paused:
                # Adjust start time to keep synchronization
                start_time = time.time() - (frame_index * frame_interval)
        elif key == ord('n'):  # Next frame
            paused = True
            frame_index = min(frame_index + 1, total_frames - 1)
            frame = decoded_frames[frame_index]
            cv2.imshow("Decoded Video", frame)
            # Update audio position
            wf.setpos(frame_index * audio_frames_per_video_frame)
        elif key == ord('p'):  # Previous frame
            paused = True
            frame_index = max(frame_index - 1, 0)
            frame = decoded_frames[frame_index]
            cv2.imshow("Decoded Video", frame)
            # Update audio position
            wf.setpos(frame_index * audio_frames_per_video_frame)
        elif key == ord('r'):  # Reset playback to the beginning
            paused = True
            frame_index = 0
            wf.rewind()  # Reset audio to the beginning
            start_time = time.time()  # Reset playback timing
            frame = decoded_frames[frame_index]
            cv2.imshow("Decoded Video", frame)

    # Cleanup
    cv2.destroyAllWindows()
    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()
    wf.close()

def main():
    # Ensure correct command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 decoder.py input_video.cmp input_audio.wav")
        return
    
    cmp_file = sys.argv[1]
    audio_file = sys.argv[2]

    # Read the compressed video file
    n1, n2, frames, padded_width, padded_height, original_width, original_height = read_cmp_file(cmp_file)
    total_frames = len(frames)
    fps = 30  # Adjust FPS as needed
    print(f"n1: {n1}, n2: {n2}")
    print(f"Frame dimensions: {original_width}x{original_height}")
    print(f"Total frames: {total_frames}")
    
    # Decode all frames
    decoded_frames = []
    for i in range(total_frames):
        print(f"Reconstructing frame {i+1}/{total_frames}")
        frame_blocks = frames[i]
        frame = reconstruct_frame_fast(frame_blocks, n1, n2, padded_width, padded_height, original_width, original_height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert for proper display in OpenCV
        decoded_frames.append(frame)

    # Start playback
    print("Starting synchronized playback...")
    playback_with_audio(decoded_frames, audio_file, fps=fps)

if __name__ == "__main__": 
    main()