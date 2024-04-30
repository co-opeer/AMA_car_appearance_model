import cv2

def video_to_frames(video_path, output_path, desired_fps):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = frame_count / fps
    print("Seconds in video: ",total_seconds)
    interval = int(round(fps / desired_fps))
    success, image = vidcap.read()
    count = 0
    frame_num = 0
    while success:
        if frame_num % interval == 0:
            cv2.imwrite(f"{output_path}/frame{count}.jpg", image)
            print(f'Frame {count} extracted successfully')
            count += 1
        success, image = vidcap.read()
        frame_num += 1

# Вказати шлях до вашого відео тут
video_path = r"C:\Users\PC\PycharmProjects\lab1\AI\Video\Videos\istockphoto-2031690257-640_adpp_is.mp4"

# Вказати шлях до вихідної теки, де зберігатимуться фотографії
output_path = r"C:\Users\PC\PycharmProjects\lab1\AI\Video\Photos"

# Бажана частота кадрів на секунду (fps)
desired_fps = 1  # Наприклад, 1 кадр на секунду

# Створення теки виводу, якщо вона не існує
import os
os.makedirs(output_path, exist_ok=True)

# Виклик функції перетворення з бажаною частотою кадрів
video_to_frames(video_path, output_path, desired_fps)
