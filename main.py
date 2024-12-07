import numpy as np
import cv2
import os
import pygame
import threading

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

def play_audio(audio_path):
    """Plays the audio using pygame with its native sampling rate (44.1 kHz)."""
    pygame.mixer.init(frequency=44100)  # Set audio to 44.1 kHz
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

def play_rgb_video_with_keyboard(frames, audio_path):
    """Plays the RGB video frames with keyboard controls."""
    fps = 30  # Video playback is fixed at 30 FPS
    audio_thread = threading.Thread(target=play_audio, args=(audio_path,))
    audio_thread.start()

    is_playing = True  # Start in play mode
    frame_index = 0

    while True:
        # Display current video frame
        if frame_index < len(frames):
            frame_bgr = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR)
            cv2.imshow("RGB Video Player", frame_bgr)

        # Keyboard controls
        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Play/Pause toggle
            is_playing = not is_playing
            if is_playing:
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.pause()
        elif key == ord('n'):  # Next frame
            is_playing = False
            pygame.mixer.music.pause()
            frame_index = min(frame_index + 1, len(frames) - 1)
            pygame.mixer.music.set_pos(frame_index / fps)  # Sync audio to frame
        elif key == ord('p'):  # Previous frame
            is_playing = False
            pygame.mixer.music.pause()
            frame_index = max(frame_index - 1, 0)
            pygame.mixer.music.set_pos(frame_index / fps)  # Sync audio to frame
        elif key == ord('r'):  # Replay
            frame_index = 0
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
            is_playing = True

        # Automatic playback if in play mode
        if is_playing:
            frame_index += 1
            if frame_index >= len(frames):
                print("End of video reached.")
                break

    pygame.mixer.music.stop()
    cv2.destroyAllWindows()

# Example usage
sample = "Stairs"
file_path = f"960x540/{sample}.rgb"
audio_path = f"audio/{sample}.wav"
width, height = 960, 540

if os.path.exists(file_path) and os.path.exists(audio_path):
    frames, total_frames = read_rgb_video(file_path, width, height)
    print(f"Loaded {total_frames} frames.")
    play_rgb_video_with_keyboard(frames, audio_path)
else:
    print("Video or audio file not found. Please check the file paths.")
