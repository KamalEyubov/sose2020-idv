from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import glob


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None, simClr=False):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.simClr = simClr
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.simClr:
            if self.transform:
                image1 = self.transform(image)
                image2 = self.transform(image)
            sample = {'img1': image1,
                      'img2': image2}
        else:
            if self.transform:
                image = self.transform(image)
            sample = {'img': image,
                      'label': int(self.img_list[idx][1])}
        return sample

class LungDataset(Dataset):
    def __init__(self, path, transform=None):
        self.img = []
        self.label = []
        self.transform = transform

        for filename in glob.glob(path+'/*.png'): #assuming gif
            self.img.append(filename)
            self.label.append(0.)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
            sample = {'img1': image1,
                      'img2': image2}
        return sample
