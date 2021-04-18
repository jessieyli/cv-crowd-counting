import os
import cv2

num_of_images = len(os.listdir('img1'))

image_folder = 'img1' # make sure to use your folder
video_name = 'mygeneratedvideo.avi'
  
images = [img for img in os.listdir(image_folder)
          if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

# setting frame width, height
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 30, (width, height))

# write images into video
for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image)))

# cleanup
cv2.destroyAllWindows()
video.release()