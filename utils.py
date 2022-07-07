from tqdm import tqdm
from typing import Optional
from pathlib import Path
import cv2


def video2images(video_path: str,
                 images_dir: str,
                 image_name_prefix: Optional[str] = None,
                 interval_frames: int = 1,
                 ):
    video_path = Path(video_path)
    images_dir = Path(images_dir)
    images_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print('Cannot open camera!')
        exit()

    counter = 1
    current_frame_id = 1
    tbar = tqdm(total=num_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot receive frame (stream end?). Exiting ...\n')
            break

        tbar.update(1)
        if current_frame_id % interval_frames == 0:
            if image_name_prefix:
                img_name = image_name_prefix + '_' + str(counter) + '.jpg'
            else:
                img_name = str(counter) + '.jpg'
            counter += 1

            img_save_path = images_dir / img_name
            cv2.imwrite(str(img_save_path), frame)
        current_frame_id += 1
    tbar.close()
