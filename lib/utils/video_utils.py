import os
import cv2
from lib.utils.etqdm import etqdm

def image_to_video(file, output, fps=10, resize=None):
    num = sorted(os.listdir(file))
    try:
        num = sorted(os.listdir(file), key=lambda item: int(item.split('.')[0]))
    except:
        pass
    tmp_img = cv2.imread(os.path.join(file, num[0]))
    H, W = tmp_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if resize is not None:
        videowriter = cv2.VideoWriter(output, fourcc, fps, (resize[0], resize[1]))
    else:
        videowriter = cv2.VideoWriter(output, fourcc, fps, (W, H))
    bar = etqdm(num)
    for n in bar:
        path = os.path.join(file, n)
        frame = cv2.imread(path)
        if resize is not None:
            frame = cv2.resize(frame, resize)
        videowriter.write(frame)
        bar.set_description(path)

    videowriter.release()

