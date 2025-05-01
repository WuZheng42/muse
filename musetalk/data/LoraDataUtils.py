import glob
import os
import pickle
import random
import torchvision.transforms as transforms
import librosa
import numpy as np
from transformers import WhisperModel
import cv2
import torch
from torch.utils.data import Dataset
from musetalk.utils.audio_processor import AudioProcessor
from transformers import AutoFeatureExtractor
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


def get_audio_file(wav_path, start_index, feature_extractor):
    """Get audio file features

    Args:
        wav_path: Audio file path
        start_index: Starting index

    Returns:
        tuple: (Audio features, start index)
    """
    if not os.path.exists(wav_path):
        return None
    audio_input_librosa, sampling_rate = librosa.load(wav_path, sr=16000)
    assert sampling_rate == 16000

    while start_index >= 25 * 30:
        audio_input = audio_input_librosa[16000 * 30:]
        start_index -= 25 * 30
    if start_index + 2 * 25 >= 25 * 30:
        start_index -= 4 * 25
        audio_input = audio_input_librosa[16000 * 4:16000 * 34]
    else:
        audio_input = audio_input_librosa[:16000 * 30]

    assert 2 * (start_index) >= 0
    assert 2 * (start_index + 2 * 25) <= 1500

    audio_input = feature_extractor(
        audio_input,
        return_tensors="pt",
        sampling_rate=sampling_rate
    ).input_features
    return audio_input, start_index


def ProcessVideoV1_5(path):
    fps = get_video_fps(path)
    assert fps == 25.0

    # Set output paths
    input_basename = os.path.basename(path).split('.')[0]

    # Create temporary directories
    temp_dir = './data/temp'
    os.makedirs(temp_dir, exist_ok=True)

    save_dir_org_frame = os.path.join(temp_dir, input_basename, 'org')
    save_dir_head_frame = os.path.join(temp_dir, input_basename, 'head1_5')
    save_audio_path = os.path.join(temp_dir, input_basename, "wav.wav")
    crop_coord_save_path = os.path.join(temp_dir, input_basename, input_basename + ".pkl")
    os.makedirs(save_dir_org_frame, exist_ok=True)
    os.makedirs(save_dir_head_frame, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {path} -start_number 0 {save_dir_org_frame}/%08d.png"
    os.system(cmd)
    cmd = f"ffmpeg -v fatal -i {path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {save_audio_path}"
    os.system(cmd)
    input_img_list = sorted(glob.glob(os.path.join(save_dir_org_frame, '*.[jpJP][pnPN]*[gG]')))

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


# （bsz, num_frames, c, h, w）batch个（num_frames, c, h, w）的sample，一个视频可以提取一小个一小个的clip，其中包含num_frames
class CustomDataset(Dataset):
    def __init__(self, project_dir, num_frames):
        self.project_dir = project_dir
        self.num_frames = num_frames
        self.contorl_face_min_size = True
        self.min_face_size = 150
        self.max_attempts = 200
        # Feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("./models/whisper")
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return 2500

    def __getitem__(self, idx):
        attempts = 0
        while attempts < self.max_attempts:
            save_dir_head_frame = os.path.join(self.project_dir, 'head1_5')
            input_head_img_list = sorted(glob.glob(os.path.join(save_dir_head_frame, '*.[jpJP][pnPN]*[gG]')))

            step = 1
            s = 0
            e = len(input_head_img_list)
            drive_idx_start = random.randint(s, e - self.num_frames * step)
            drive_idx_list = list(
                range(drive_idx_start, drive_idx_start + self.num_frames * step, step))
            assert len(drive_idx_list) == self.num_frames
            len_valid_clip = e - s

            if len_valid_clip < self.num_frames * 10:
                attempts += 1
                print(f"video {self.project_dir} has less than {self.num_frames * 10} frames")
                continue

            src_idx_list = drive_idx_list[:]  # 创建原列表的一个切片副本
            random.shuffle(src_idx_list)  # 打乱副本的顺序

            # Get reference images
            ref_face_valid_flag = True
            ref_imgs = []
            for src_idx in src_idx_list:
                imSrc = Image.open(input_head_img_list[src_idx]).convert("RGB")
                if self.contorl_face_min_size and min(imSrc.size[0], imSrc.size[1]) < self.min_face_size:
                    ref_face_valid_flag = False
                    break
                ref_imgs.append(imSrc)

            if not ref_face_valid_flag:
                attempts += 1
                print(
                    f"video {self.project_dir} has reference face size smaller than minimum required {self.min_face_size}")
                continue
            imSameIDs = []
            target_face_valid_flag = True
            for drive_idx in drive_idx_list:
                imSameID = Image.open(input_head_img_list[drive_idx]).convert("RGB")
                if self.contorl_face_min_size and min(imSameID.size[0], imSameID.size[1]) < self.min_face_size:
                    target_face_valid_flag = False
                    break
                imSameIDs.append(imSameID)
            if not target_face_valid_flag:
                attempts += 1
                print(
                    f"video {self.project_dir} has target face size smaller than minimum required {self.min_face_size}")
                continue

            # Process audio features
            audio_offset = drive_idx_list[0]
            audio_step = step
            fps = 25.0 / step

            save_audio_path = os.path.join(save_dir_head_frame, "wav.wav")
            audio_feature, audio_offset = get_audio_file(save_audio_path, audio_offset, self.feature_extractor)

            pixel_values = torch.stack(
                [self.to_tensor(imSameID) for imSameID in imSameIDs], dim=0)
            ref_pixel_values = torch.stack(
                [self.to_tensor(ref_img) for ref_img in ref_imgs], dim=0)
            return pixel_values, ref_pixel_values, audio_feature, audio_offset, audio_step

        raise ValueError("Unable to find a valid sample after maximum attempts.")


def GetDataLoader(project_dor, num_frames):
    dataset = CustomDataset(project_dor, num_frames)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=1,
    )
    return dataloader


# ProcessVideoV1_5('D:\\PythonProject\\MimicTalkForWin\\data\\LeiJun\\LeiJun.mp4')
