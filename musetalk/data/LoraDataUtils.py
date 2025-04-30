import torch
from torch.utils.data import Dataset

BATCH = 10
NUM_FRAMES = 10


def ProcessVideo(path):
    pass


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
    # 创建 Dataset 和 DataLoader，并启用 shuffle
    dataset = CustomDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataloader_dict['train_dataset'],
        batch_size=BATCH,
        shuffle=True,
        num_workers=1,
    )
    return dataloader
