import cv2
import os

image_folder = "../data/all_navcam/output/test/left_jpgs"
video_name = "../data/all_navcam/valid_1.avi"
video_name_2 = "../data/all_navcam/valid_2.avi"
video_name_3 = "../data/all_navcam/valid_3.avi"

images = [img for img in os.listdir(image_folder)] #if img.enswith(".png")
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))
video2 = cv2.VideoWriter(video_name_2, 0, 1, (width, height))
video3 = cv2.VideoWriter(video_name_3, 0, 1, (width, height))

count = 0

for image in images:
    if count < 500:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    if (500 <= count < 1000):
        video2.write(cv2.imread(os.path.join(image_folder, image)))
    if count >= 1000:
        video3.write(cv2.imread(os.path.join(image_folder, image)))
    count += 1

cv2.destroyAllWindows()
video.release()
video2.release()
video3.release()

