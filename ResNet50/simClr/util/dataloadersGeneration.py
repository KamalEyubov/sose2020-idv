import torchvision.transforms as transforms
from util.covidDataSet import *
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def getTransforms():
    ########## Mean and std are calculated from the train dataset
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                         std=[0.33165374, 0.33165374, 0.33165374])

    train_transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    return train_transformer, val_transformer

def getTransformsLuna():

    normalize = transforms.Normalize(mean=[0.3226, 0.3226, 0.3226],
                                         std=[0.3404, 0.3404, 0.3404])


    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(sigma=[.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    return train_transformer

def getTransformsSimClr():

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                             std=[0.33165374, 0.33165374, 0.33165374])
    train_transformer = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(sigma=[.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    return train_transformer

def getTransformedDataSplit():

    train_transformer, val_transformer = getTransforms()

    trainset = CovidCTDataset(root_dir='../../Images',
                              txt_COVID='../../Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='../../Images',
                              txt_COVID='../../Data-split/COVID/valCT_COVID.txt',
                              txt_NonCOVID='../../Data-split/NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='../../Images',
                              txt_COVID='../../Data-split/COVID/testCT_COVID.txt',
                              txt_NonCOVID='../../Data-split/NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)
    print('Trainset', trainset.__len__())
    print('Valset', valset.__len__())
    print('Testset', testset.__len__())

    return trainset, valset, testset

def getTransformedLUNA():
    train_transformer = getTransformsLuna()
    trainset = LungDataset(path='../train', transform=train_transformer)
    print('Trainset', trainset.__len__())
    return trainset

def getTransformedSimClr():
    train_transformer = getTransformsSimClr()
    trainset = CovidCTDataset(root_dir='../../Images',
                              txt_COVID='../../Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer,
                              simClr=True)
    return trainset
