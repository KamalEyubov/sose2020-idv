import torchvision.transforms as transforms
from covidDataSet import *


def getTransforms():
    ########## Mean and std are calculated from the train dataset
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                         std=[0.33165374, 0.33165374, 0.33165374])
    train_transformer = transforms.Compose([
        #transforms.Resize(256),
        transforms.Resize((224,224)),
        #transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation((90,90)),
        # random brightness and random contrast
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    return train_transformer, val_transformer

def getTransformedDataSplit():
    train_transformer, val_transformer = getTransforms()
    trainset = CovidCTDataset(root_dir='../Images',
                              txt_COVID='../Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='../Data-split/NonCOVID/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='../Images',
                              txt_COVID='../Data-split/COVID/valCT_COVID.txt',
                              txt_NonCOVID='../Data-split/NonCOVID/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='../Images',
                              txt_COVID='../Data-split/COVID/testCT_COVID.txt',
                              txt_NonCOVID='../Data-split/NonCOVID/testCT_NonCOVID.txt',
                              transform= val_transformer)
    print('Trainset', trainset.__len__())
    print('Valset', valset.__len__())
    print('Testset', testset.__len__())

    return trainset, valset, testset
