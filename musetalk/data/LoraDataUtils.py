import glob
import os
import pickle
import numpy as np
from transformers import WhisperModel
import cv2
import torch
from torch.utils.data import Dataset
from musetalk.utils.audio_processor import AudioProcessor
from PIL import Image
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder


BATCH = 10
NUM_FRAMES = 10
extra_margin_1_5 = 10


def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def crop_resize_img(self, img, bbox, crop_type='crop_resize', extra_margin=None):
    """Crop and resize image

    Args:
        img: Input image
        bbox: Bounding box
        crop_type: Type of cropping
        extra_margin: Extra margin

    Returns:
        tuple: (Processed image, extra_margin, mask_scaled_factor)
    """
    mask_scaled_factor = 1.
    if crop_type == 'crop_resize':
        x1, y1, x2, y2 = bbox
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
    elif crop_type == 'dynamic_margin_crop_resize':
        x1, y1, x2, y2, extra_margin = self.dynamic_margin_crop(img, bbox, extra_margin)
        w_original, _ = img.size
        img = img.crop((x1, y1, x2, y2))
        w_cropped, _ = img.size
        mask_scaled_factor = w_cropped / w_original
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
    elif crop_type == 'resize':
        w, h = img.size
        scale = np.sqrt(self.image_size ** 2 / (h * w))
        new_w = int(w * scale) / 64 * 64
        new_h = int(h * scale) / 64 * 64
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img, extra_margin, mask_scaled_factor


def ProcessVideoV1_5(path):
    # Set output paths
    input_basename = os.path.basename(path).split('.')[0]

    # Create temporary directories
    temp_dir = './data/temp'
    os.makedirs(temp_dir, exist_ok=True)

    save_dir_org_frame = os.path.join(temp_dir, input_basename, 'org')
    save_dir_head_frame = os.path.join(temp_dir, input_basename, 'head1_5')
    crop_coord_save_path = os.path.join(temp_dir, input_basename, input_basename + ".pkl")
    os.makedirs(save_dir_org_frame, exist_ok=True)
    os.makedirs(save_dir_head_frame, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {path} -start_number 0 {save_dir_org_frame}/%08d.png"
    os.system(cmd)
    input_img_list = sorted(glob.glob(os.path.join(save_dir_org_frame, '*.[jpJP][pnPN]*[gG]')))
    fps = get_video_fps(path)

    coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)
    with open(crop_coord_save_path, 'wb') as f:
        pickle.dump(coord_list, f)

    head_img_index = 0

    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + extra_margin_1_5
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        # 定义保存图像的路径和文件名
        output_path = os.path.join(save_dir_head_frame, f'{head_img_index}.png')

        # 使用 OpenCV 的 imwrite 方法保存图像
        cv2.imwrite(output_path, crop_frame)
        head_img_index += 1

    # audio_processor = AudioProcessor(feature_extractor_path='./models/whisper')
    # # Extract audio features
    # weight_dtype = torch.float32
    # whisper = WhisperModel.from_pretrained('./models/whisper')
    # whisper = whisper.to(device='cuda:0', dtype=weight_dtype).eval()
    # whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    # whisper_chunks = audio_processor.get_whisper_chunk(
    #     whisper_input_features,
    #     'cuda:0',
    #     weight_dtype,
    #     whisper,
    #     librosa_length,
    #     fps=fps,
    #     audio_padding_length_left=2,
    #     audio_padding_length_right=2,
    # )


# （bsz, num_frames, c, h, w）batch个（num_frames, c, h, w）的sample，一个视频可以提取一小个一小个的clip，其中包含num_frames
class CustomDataset(Dataset):
    def __init__(self, video_paths, num_frames):
        self.video_paths = video_paths
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_paths = self.video_paths[idx]


def GetDataLoader(file='Data\\TrainSeged.txt', num_limit=-1):
    dataloader = torch.utils.data.DataLoader(
        dataset=None,
        batch_size=BATCH,
        shuffle=True,
        num_workers=1,
    )
    return dataloader

ProcessVideoV1_5('D:\\PythonProject\\MimicTalkForWin\\data\\LeiJun\\LeiJun.mp4')