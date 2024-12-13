import sys
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import os
import time
# from line_profiler import LineProfiler


def compute_motion_vectors_phase_corr(prev_frame, cur_frame, block_size=16):
    prev_gray = prev_frame[:, :, 0].astype(np.float32)
    cur_gray = cur_frame[:, :, 0].astype(np.float32)
    
    h, w = cur_gray.shape
    vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cur_block = cur_gray[by:by + block_size, bx:bx + block_size]
            ref_block = prev_gray[by:by + block_size, bx:bx + block_size]

            # Use Phase Correlation to find motion
            shift = cv2.phaseCorrelate(ref_block, cur_block)
            vectors[by // block_size, bx // block_size] = [int(shift[0][1]), int(shift[0][0])]

    return vectors

def compute_motion_vectors_ssd(prev_frame, cur_frame, block_size=16, search_range=4):
    prev_gray = prev_frame[:, :, 0].astype(np.float32)
    cur_gray = cur_frame[:, :, 0].astype(np.float32)
    
    h, w = cur_gray.shape
    vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cur_block = cur_gray[by:by + block_size, bx:bx + block_size]
            best_mv = (0, 0)
            best_ssd = float('inf')

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_y = by + dy
                    ref_x = bx + dx
                    if ref_y < 0 or ref_y + block_size > h or ref_x < 0 or ref_x + block_size > w:
                        continue
                    
                    ref_block = prev_gray[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                    ssd = np.sum((cur_block - ref_block) ** 2)
                    
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_mv = (dy, dx)
            vectors[by // block_size, bx // block_size] = best_mv
    return vectors

def compute_motion_vectors_mad(prev_frame, cur_frame, block_size=16, search_range=4):
    """
    Compute motion vectors using a brute force block matching (MAD).
    Here we only use the R channel for simplicity, or convert to Y and use Y.
    """
    # For simplicity, let's just use R channel
    prev_gray = prev_frame[:,:,0].astype(np.float32)
    cur_gray = cur_frame[:,:,0].astype(np.float32)
    
    h, w = cur_gray.shape
    vectors = np.zeros((h//block_size, w//block_size, 2), dtype=np.int32)
    
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            cur_block = cur_gray[by:by+block_size, bx:bx+block_size]
            
            # Search in prev_frame around (by,bx)
            best_mad = 1e9
            best_mv = (0, 0)
            for dy in range(-search_range, search_range+1):
                for dx in range(-search_range, search_range+1):
                    ref_y = by+dy
                    ref_x = bx+dx
                    if (ref_y < 0 or ref_y+block_size>h or ref_x<0 or ref_x+block_size> w):
                        continue
                    ref_block = prev_gray[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                    diff = np.abs(cur_block - ref_block)
                    mad = np.mean(diff)
                    if mad < best_mad:
                        best_mad = mad
                        best_mv = (dy, dx)
            vectors[by//block_size, bx//block_size] = best_mv
    return vectors



def segment_foreground_background(motion_vectors, threshold=2):
    """
    Segment into foreground and background using motion vector consistency.
    We consider background as having very low or uniform motion.
    A simplistic heuristic: 
      - Compute magnitude of motion vectors, 
      - If magnitude < threshold, mark background, else foreground.
    """
    mag = np.sqrt(motion_vectors[:,:,0]**2 + motion_vectors[:,:,1]**2)
    # Here we use a simple global threshold
    fg_mask = (mag >= threshold)
    # fg_mask = True indicates foreground block
    return fg_mask

def rgb_to_blocks(frame, block_size=8):
    """Split the frame's R,G,B channels into 8x8 blocks for DCT."""
    h, w, _ = frame.shape
    r_blocks = []
    g_blocks = []
    b_blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            rb = frame[y:y+block_size, x:x+block_size, 0]
            gb = frame[y:y+block_size, x:x+block_size, 1]
            bb = frame[y:y+block_size, x:x+block_size, 2]
            r_blocks.append(rb)
            g_blocks.append(gb)
            b_blocks.append(bb)
    return r_blocks, g_blocks, b_blocks

def block_dct(block):
    """Compute 2D DCT of an 8x8 block using OpenCV."""
    return cv2.dct(block.astype(np.float32))

def block_idct(coeff_block):
    """Compute inverse DCT of an 8x8 block using OpenCV."""
    return cv2.idct(coeff_block)

def quantize_block(dct_block, q_step):
    """Uniform quantization with step = 2^(q_step)."""
    Q = 2**q_step
    return np.round(dct_block / Q).astype(np.int16)

def dequantize_block(q_block, q_step):
    Q = 2**q_step
    return q_block * Q


def encode_frame(frame, fg_mask, n1, n2, block_size=8):
    """
    Encode one frame:
    1) For each 8x8 block, check if it's foreground or background.
    2) Compute DCT and quantize using either n1 or n2.
    3) Output: [(block_type, [Rcoeffs], [Gcoeffs], [Bcoeffs]) ...]
    """
    # encode_start = time.time()
    h, w, _ = frame.shape

    # DCT at 8x8 level
    # First get all 8x8 blocks
    r_blocks, g_blocks, b_blocks = rgb_to_blocks(frame, block_size=8)

    # Determine block type from fg_mask:
    # The block index corresponds to (b_row, b_col) in 8x8 indexing
    # Macroblock index in fg_mask is at 16x16. One MB = 2x2 of these 8x8 blocks.
    block_data = []
    blocks_per_row = w//8
    for i, (rb, gb, bb) in enumerate(zip(r_blocks, g_blocks, b_blocks)):
        block_row = i // blocks_per_row
        block_col = i % blocks_per_row
        # Map this block to its macroblock coordinates
        mb_row = block_row // 2
        mb_col = block_col // 2
        is_fg = fg_mask[mb_row, mb_col]

        # DCT and quant
        r_dct = block_dct(rb)
        g_dct = block_dct(gb)
        b_dct = block_dct(bb)

        q_step = n1 if is_fg else n2
        r_q = quantize_block(r_dct, q_step)
        g_q = quantize_block(g_dct, q_step)
        b_q = quantize_block(b_dct, q_step)

        block_type = 1 if is_fg else 0
        # Flatten coefficients for output
        r_coeffs = r_q.flatten()
        g_coeffs = g_q.flatten()
        b_coeffs = b_q.flatten()
        block_data.append((block_type, r_coeffs, g_coeffs, b_coeffs))

    return block_data

def read_rgb_file(filename, width, height, num_frames):
    """
    Read a .rgb file which contains frames in RGB888 format: width*height*3 bytes per frame.
    Return as a list of numpy arrays of shape (height, width, 3).
    """
    with open(filename, 'rb') as f:
        frames = []
        frame_size = width * height * 3
        for _ in range(num_frames):
            raw = f.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            frames.append(frame)
    return frames


def write_cmp_file(filename, n1, n2, all_frame_blocks):
    """
    Optimized writing of compressed data to .cmp file
    """
    write_start = time.time()
    print("Writing compressed data to", filename)
    
    # Pre-calculate total number of blocks
    total_blocks = sum(len(frame_blocks) for frame_blocks in all_frame_blocks)
    
    # Pre-allocate array for all data
    all_data = np.zeros((total_blocks, 193), dtype=np.int32)  # 1 block_type + 3*64 coeffs
    
    # Fill array with data
    idx = 0
    for frame_blocks in all_frame_blocks:
        for block_type, r_coeffs, g_coeffs, b_coeffs in frame_blocks:
            all_data[idx, 0] = block_type
            all_data[idx, 1:65] = r_coeffs.flatten()
            all_data[idx, 65:129] = g_coeffs.flatten()
            all_data[idx, 129:193] = b_coeffs.flatten()
            idx += 1
    
    # Write to file using buffer
    with open(filename, 'w', buffering=1024*1024) as f:
        # Write header
        f.write(f"{n1} {n2}\n")
        
        # Convert array to string efficiently
        np.savetxt(f, all_data, fmt='%d', delimiter=' ')
        
    write_time = time.time() - write_start
    print(f"File writing time: {write_time:.3f} seconds")
    print(f"Saved {total_blocks} blocks to {filename}")
    
    
def pad_frame(frame, pad_height, pad_width):
    """
    Pad the frame to be divisible by 16 in both dimensions.
    If the frame is already divisible by 16, no padding is done.
    """
    h, w, c = frame.shape
    new_h = ((h + 15) // 16) * 16  # rounds up to next multiple of 16
    new_w = ((w + 15) // 16) * 16
    padded = np.zeros((new_h, new_w, c), dtype=frame.dtype)
    padded[:h, :w, :] = frame
    return padded, h, w, new_h, new_w


def main():
    
    start_time = time.time()
    
    input_file = sys.argv[1]
    n1 = int(sys.argv[2])
    n2 = int(sys.argv[3])
    output_file = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{output_file}.cmp"

    # Suppose your 540p video is 960x540
    width = 960
    height = 540
    
    frame_size = width * height * 3  # 3 bytes per pixel for RGB
    # Get the total file size
    file_size = os.path.getsize(input_file)
    # Calculate the total number of frames
    num_frames = file_size // frame_size

    frames = read_rgb_file(input_file, width, height, num_frames)


    padded_frames = []
    for f in frames:
        pframe, orig_h, orig_w, new_h, new_w = pad_frame(f, 16, 16)
        padded_frames.append(pframe)

    # Now use padded_frames for motion estimation and compression
    all_frame_blocks = []
    prev_frame = None

    total_encode_start = time.time()
    print(f"Encoding {len(padded_frames)} frames...")
    for i, frame in enumerate(padded_frames):
        each_frame_start = time.time()
        print(f"Processing frame {i+1}/{len(padded_frames)}")
        if i == 0:
            # First frame is I-frame
            mb_rows = (frame.shape[0] // 16)
            mb_cols = (frame.shape[1] // 16)
            fg_mask = np.zeros((mb_rows, mb_cols), dtype=bool)
        else:
            motion_vectors = compute_motion_vectors_phase_corr(prev_frame, frame)
            # motion_vectors = compute_motion_vectors_ssd(prev_frame, frame)
            fg_mask = segment_foreground_background(motion_vectors)

        print("Encoding frame...")
        frame_blocks = encode_frame(frame, fg_mask, n1, n2, block_size=8)
        
        # print("Adding frame to output_blocks...")
        all_frame_blocks.append(frame_blocks)
        prev_frame = frame

        each_frame_time = time.time() - each_frame_start
        print(f"Frame processing time: {each_frame_time:.3f} seconds")
    
    
    print(f"Detected resolution: {width}x{height}")
    print(f"Detected number of frames: {num_frames}")
    
    total_encode_time = time.time() - total_encode_start
    print(f"Total encoding time: {total_encode_time/60:.3f} minutes")
        
    write_time_start = time.time()
    write_cmp_file(output_file, n1, n2, all_frame_blocks)
    write_time_end = time.time() - write_time_start
    print(f"File writing time: {write_time_end/60:.3f} minutes")
    
    print(f"Compression complete. Output: {output_file}")
    
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time)/60} minutes")

if __name__ == "__main__":
    main()
