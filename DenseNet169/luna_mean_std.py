import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')


torch.cuda.empty_cache()

device = 'cpu'
batchsize = 1000 # whole data set


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class LUNADataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        """
            Args:
                txt_path (string): Path to the txt file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied on a sample.
            File structure:
                - LUNA
                    - img1.png
                    - img2.png
                    - ......
        """
        self.img_list = [os.path.join(root_dir, item) for item in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image}
        return sample



if __name__ == '__main__':
    dataset = LUNADataset(
        root_dir='../LUNA',
    )
    print("dataset length:", dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=batchsize, drop_last=False, shuffle=True)

# 1 iteration
data = next(iter(data_loader))
data = data['img']

f = open("luna_stats.txt", "w")
f.write("mean: {}\nstd: {}\n".format(data.mean((0, 2, 3)), data.std((0, 2, 3))))
f.close()
