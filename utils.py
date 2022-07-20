import os
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
from tqdm import tqdm


def video2images(video_path: str,
                 images_dir: str,
                 image_name_prefix: Optional[str] = None,
                 interval_frames: int = 1,
                 ) -> None:
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


def clip_video(video_path: str,
               video_save_dir: str,
               start_second: int,
               end_second: int,
               ) -> None:
    video_path = Path(video_path)
    video_save_dir = Path(video_save_dir)
    video_save_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Cannot open camera!')
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = start_second * fps
    end_frame = end_second * fps

    if 'avi' in video_path.suffix:
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    else:
        raise NotImplementedError
    video_save_path = video_save_dir / video_path.name
    video_writer = cv2.VideoWriter(str(video_save_path), fourcc, fps, (frame_width, frame_height))

    frame_counter = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot receive frame (stream end?). Exiting ...\n')
            break

        if start_frame <= frame_counter <= end_frame:
            video_writer.write(frame)
        elif frame_counter > end_frame:
            break

        frame_counter += 1


def concat_videos(videos_path: List[str],
                  video_save_path: str,
                  layout: Optional[Tuple[int, int]] = None,
                  videos_title: Optional[List[str]] = None,
                  ) -> None:
    assert videos_path, 'Videos path list must not be None!'

    if layout is None:
        layout = (len(videos_path), 1)
    elif isinstance(layout, list):
        layout = tuple(layout)

    if videos_title is None:
        videos_title = ['video_' + str(i) for i in range(len(videos_path))]

    cap_list = [cv2.VideoCapture(vid_path) for vid_path in videos_path]

    num_frames = int(cap_list[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap_list[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_list[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    for cap in cap_list[1:]:
        if num_frames != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise 'The number of frames in all videos should be the same!'

        if frame_width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
            raise 'The width of frame in all videos should be the same!'

        if frame_height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
            raise 'The height of frame in all videos should be the same!'

    fps = int(cap_list[0].get(cv2.CAP_PROP_FPS))
    if 'avi' in Path(videos_path[0]).suffix:
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    else:
        raise NotImplementedError

    num_row = layout[0]
    num_col = layout[1]
    big_frame_height = int(frame_height * num_row)
    big_frame_width = int(frame_width * num_col)
    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (big_frame_width, big_frame_height))

    width_center = frame_width // 2
    while True:
        big_frame = np.empty((big_frame_height, big_frame_width, 3))

        for i, (cap, vid_title) in enumerate(zip(cap_list, videos_title)):
            ret, frame = cap.read()
            if not ret:
                print('Cannot receive frame (stream end?). Exiting ...\n')
                return True
            cv2.putText(frame, vid_title, (width_center, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4, cv2.LINE_AA)

            row = i // num_row
            col = i % num_row
            big_frame[col * frame_height: (col + 1) * frame_height, row * frame_width:(row + 1) * frame_width,
            :] = frame
        video_writer.write(big_frame.astype(np.uint8))

 
def remove_labels_without_images(images_dir: str,
                                 labels_dir: str, 
                                ) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    images_path = list(images_dir.glob('*.jpg'))
    images_stem = [img_path.stem for img_path in images_path]
    labels_path = labels_dir.glob('*.txt')
    for _, lbl_path in enumerate(labels_path):
        lbl_stem = lbl_path.stem
        if lbl_stem not in images_stem:
            os.remove(str(lbl_path))
            
