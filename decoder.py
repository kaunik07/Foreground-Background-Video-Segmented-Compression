import sys
import numpy as np
import cv2
from scipy.fftpack import idct

def block_idct(coeff_block):
    return idct(idct(coeff_block, axis=1, norm='ortho'), axis=0, norm='ortho')

def dequantize_block(q_block, q_step):
    Q = 2**q_step
    return q_block * Q

def reconstruct_frame(frame_blocks, n1, n2, padded_width, padded_height, original_width, original_height):
    reconstructed = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    blocks_per_row = padded_width // 8

    for i, (block_type, rQ, gQ, bQ) in enumerate(frame_blocks):
        q_step = n1 if block_type == 1 else n2
        r_dct = dequantize_block(rQ, q_step)
        g_dct = dequantize_block(gQ, q_step)
        b_dct = dequantize_block(bQ, q_step)

        r_block = np.clip(block_idct(r_dct), 0, 255)
        g_block = np.clip(block_idct(g_dct), 0, 255)
        b_block = np.clip(block_idct(b_dct), 0, 255)

        block_row = i // blocks_per_row
        block_col = i % blocks_per_row

        y = block_row*8
        x = block_col*8
        reconstructed[y:y+8, x:x+8, 0] = r_block
        reconstructed[y:y+8, x:x+8, 1] = g_block
        reconstructed[y:y+8, x:x+8, 2] = b_block

    reconstructed = reconstructed.astype(np.uint8)
    # Crop to original size
    cropped_frame = reconstructed[0:original_height, 0:original_width, :]
    return cropped_frame

def read_cmp_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    n1, n2 = map(int, lines[0].split())
    data_lines = lines[1:]

    # Known or stored dimensions:
    original_width = 960
    original_height = 540
    padded_width = 960
    padded_height = 544

    blocks_per_row = padded_width // 8
    blocks_per_col = padded_height // 8
    blocks_per_frame = blocks_per_row * blocks_per_col

    frame_count = len(data_lines) // blocks_per_frame

    frames = []
    idx = 0
    for _ in range(frame_count):
        frame_blocks = []
        for _b in range(blocks_per_frame):
            parts = data_lines[idx].split()
            idx += 1
            block_type = int(parts[0])
            r_coeffs = np.array(list(map(int, parts[1:65]))).reshape(8,8)
            g_coeffs = np.array(list(map(int, parts[65:129]))).reshape(8,8)
            b_coeffs = np.array(list(map(int, parts[129:193]))).reshape(8,8)
            frame_blocks.append((block_type, r_coeffs, g_coeffs, b_coeffs))
        frames.append(frame_blocks)

    return n1, n2, frames, padded_width, padded_height, original_width, original_height

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: mydecoder.exe input_video.cmp input_audio.wav")
    #     return

    cmp_file = "WalkingMovingBackground.cmp"
    # audio_file = sys.argv[2]

    n1, n2, frames, padded_width, padded_height, original_width, original_height = read_cmp_file(cmp_file)
    total_frames = len(frames)
    print(f"n1: {n1}, n2: {n2}")
    print(f"Frame dimensions: {original_width}x{original_height}")
    print(f"Total frames: {total_frames}")
    
    # Decode all frames upfront
    decoded_frames = []
    for i in range(total_frames):
        print(f"Reconstructing frame {i+1}/{total_frames}")
        frame_blocks = frames[i]
        frame = reconstruct_frame(frame_blocks, n1, n2, padded_width, padded_height, original_width, original_height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert for proper display in OpenCV
        decoded_frames.append(frame)

    # Playback loop from memory
    current_frame_idx = 0
    paused = False
    step = False

    while True:
        if not paused or step:
            if current_frame_idx < total_frames:
                cv2.imshow("Decoded Video", decoded_frames[current_frame_idx])
                current_frame_idx += 1
            else:
                current_frame_idx = 0  # loop the video, or break if desired

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            paused = True
            step = True
        else:
            step = False

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
