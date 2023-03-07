import os
import cv2

# Create the output directory if it doesn't already exist
if not os.path.exists('ext_imag'):
    os.mkdir('ext_imag')
else:
    # Get the last image filename in the 'ext_imag' directory
    last_img_filename = sorted(os.listdir('ext_imag'))[-1]
    last_img_index = int(last_img_filename.split('.')[0][3:])
    img_index = last_img_index + 1

# Define the video file extensions to search for
video_extensions = ['.mp4', '.avi', '.mov']

# Loop through all files in the 'vids' directory
for filename in os.listdir('vids'):
    # Check if the file is a video file
    if any(filename.endswith(ext) for ext in video_extensions):
        # Get the full path to the video file
        filepath = os.path.join('vids', filename)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(filepath)

        # Loop through each frame of the video
        frame_count = 0
        while cap.isOpened():
            # Read the next frame of the video
            ret, frame = cap.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            # Save the current frame as an image file
            img_filename = f'img{img_index:05}.jpg'
            img_filepath = os.path.join('ext_imag', img_filename)
            cv2.imwrite(img_filepath, frame)

            # Increment the image index and frame count
            img_index += 1
            frame_count += 1

        # Print the number of frames extracted from the video
        print(f'Extracted {frame_count} frames from {filename}')

        # Release the OpenCV video capture object
        cap.release()

print('Finished extracting frames from all videos.')
