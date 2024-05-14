import cv2
import os
# video_path = "./jimcarrey_cut.mp4"
# video_path = './Doc/sample.mp4'
# video = cv2.VideoCapture(video_path)
# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# print(total_frames)


# import cv2
# import numpy as np

# def video_to_gif(video_path, output_path, x, y):
#     # Read the video
#     video = cv2.VideoCapture(video_path)

#     # Get the total number of frames
#     frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Check if the frame range is valid
#     if x > frame_count or y > frame_count or x < 1 or y < 1:
#         print("Invalid frame range.")
#         return

#     # Set the frame position
#     video.set(cv2.CAP_PROP_POS_FRAMES, x - 1)

#     # Initialize the gif writer
#     gif_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'gif'), 10, (640, 480))

#     # Process the frames
#     for i in range(x, y + 1):
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Resize the frame to 640x480
#         frame = cv2.resize(frame, (640, 480))

#         # Write the frame to the gif
#         gif_writer.write(frame)

#     # Release the video and gif writer
#     video.release()
#     gif_writer.release()

# # Example usage
# video_path = "./Doc/sample.mp4"
# output_path = "output_gif.gif"
# x = 830
# y = 900

from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def cut_video_and_create_gif(input_video, output_gif, start_frame, end_frame):
    # Load the video clip
    video_clip = VideoFileClip(input_video)

    # Extract the subclip from start_frame to end_frame
    subclip = video_clip.subclip(start_frame / video_clip.fps, end_frame / video_clip.fps)

    # Write the subclip as a GIF with loop
    subclip.write_gif(output_gif, program='ffmpeg', loop=0)

    # Close the video clip
    video_clip.close()

# Example usage
input_video_path = "./Doc/sample.mp4"
output_gif_path = "./output_gif.gif"
start_frame = 1100  # Change to the starting frame you want
end_frame = 1400   # Change to the ending frame you want
cut_video_and_create_gif(input_video_path, output_gif_path, start_frame, end_frame)


# import cv2
# import numpy as np
# import imageio

# def video_to_gif(video_path, output_path, x, y):
#     # Read the video
#     video = cv2.VideoCapture(video_path)

#     # Get the total number of frames
#     frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Check if the frame range is valid
#     if x > frame_count or y > frame_count or x < 1 or y < 1:
#         print("Invalid frame range.")
#         return

#     # Set the frame position
#     video.set(cv2.CAP_PROP_POS_FRAMES, x - 1)

#     # Initialize the gif writer
#     gif_writer = imageio.get_writer(output_path, mode='I', duration=20)

#     # Process the frames
#     for i in range(x, y + 1):
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Resize the frame to 640x480 
#         frame = cv2.resize(frame, (960, 540))

#         # Convert the frame to a PIL image
#         frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_pil = np.array(frame_pil)

#         # Add the frame to the gif
#         gif_writer.append_data(frame_pil)

#     # Close the gif writer
#     gif_writer.close()

#     # Release the video
#     video.release()

# # Example usage
# video_path = "./Doc/sample.mp4"
# output_path = "output_gif.gif"
# x = 1060
# y = 1300

# video_to_gif(video_path, output_path, x, y)



# def cut_video(x, y):
#     if x < 0 or y < 0 or x >= total_frames or y >= total_frames or x > y:
#         raise ValueError("Invalid input values")

#     start_frame = x
#     end_frame = y

#     output_path = "./jimcarrey_cut.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for the output video format
#     output_fps = video.get(cv2.CAP_PROP_FPS)
#     output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     output_video = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

#     for i in range(start_frame, end_frame + 1):
#         video.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = video.read()
#         if ret:
#             output_video.write(frame)

#     video.release()
#     output_video.release()


# x = 60
# y = 560
# cut_video(x, y)