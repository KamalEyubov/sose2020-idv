# sose2020-idv
Summer Semester 2020 Deep Vision Project 

```
.
├── Data-split
│   ├── COVID
│   │   Contains Covid-positive data-split
│   │   
│   │   
│   └── NonCOVID
│       Contains Covid-negative data-split
│       
│       
├── DenseNet169
│   ├── plots
│   │   ├── PRET
│   │   │  
│   │   │       
│   │   │      
│   │   │     
│   │   │      
│   │   │       
│   │   ├── RAND
│   │   │   
│   │   │      
│   │   │      
│   │   │     
│   │   │      
│   │   │       
│   │   ├── RN_1
│   │   │   ├── fine-tuning
│   │   │   │   
│   │   │   │   
│   │   │   │   
│   │   │   │   
│   │   │   │  
│   │   │   └── ssl
│   │   │       
│   │   │       
│   │   │       
│   │   │       
│   │   ├── RN_2
│   │   │   ├── fine-tuning
|   |   |   |
│   │   │   └── ssl
│   │   │    
│   │   ├── RN_3
│   │   │   ├── fine-tuning
│   │   │   │  
│   │   │   └── ssl
│   │   │    
│   │   ├── RN_ST
│   │   │   ├── fine-tuning
│   │   │   |
│   │   │   ├── ssl1
│   │   │   │ 
│   │   │   └── ssl2
│   │   │     
│   │   ├── SC_1
│   │   │   ├── fine-tuning
│   │   │   │  
│   │   │   └── ssl
│   │   │      
│   │   ├── SC_2
│   │   │   ├── fine-tuning
│   │   │   │  
│   │   │   └── ssl
│   │   │      
│   │   ├── SC_3
│   │   │   ├── fine-tuning
│   │   │   │  
│   │   │   └── ssl
│   │   │     
│   │   └── SC_ST
│   │       ├── fine-tuning
│   │       │  
│   │       ├── ssl1
│   │       │  
│   │       └── ssl2
│   │          
│   ├── PRET
│   │  
│   ├── RAND
│   │  
│   ├── RN_1
│   │  
│   ├── RN_2
│   │  
│   ├── RN_3
│   │  
│   ├── RN_ST
│   │  
│   ├── SC_1
│   │  
│   ├── SC_2
│   │  
│   ├── SC_3
│   │  
│   └── SC_ST
│      
├── Images
│   ├── CT_COVID
|   |       Contains positive COVID-CT scans from the COVID-CT dataset
│   │  
│   └── CT_NonCOVID
|           Contains negative COVID-CT scans from the COVID-CT dataset
│      
├── LUNA
|       Contains the LUNA dataset
│  
├── README.md
├── ResNet50
│  
│   ├── rotnet
│   │   |    Contains all methods for RotNet including training with no SSL and random weights 
|   |   |    and training with  no SSL and pre-trained weights
|   |   |    Contains pipe.py for running the training for all methods
|   |   |    Contains grad_cam.py for RotNet
│   │   └── util
|   |           Contains auxillary files i.e. defining the training function
│   │    
│   ├── simClr
|   |   |    Contains all methods for SimCLR
|   |   |    Contains pipe.py for running the training for all methods
|   |   |    Contains grad_cam.py for SimCLR
│   │   |
│   │   └── util
│   │           Contains auxillary files i.e. defining the training function
│  
├── ResNet50RotNetSelfTransConv.png
├── ResNet50RotNetSelfTransNorm.png
└── weightChangePlot.py

```


## DenseNet169
## ResNet50
After installing the dependencies change in the directory of RotNet or SimCLR depending on which SSL approach you want to run (i.e. cd sose2020-idv/ResNet50/rotnet).
After changing into the directory of rotnet or simClr run the pipe.py file (i.e. python pipe.py) with python3. This starts the pipeline of running all the methods one afte another. For RotNet this would be: no SSL and no pre-training (noSslNoPretrain.py), no SSL with pretraining (noSslWithPretrain.py), RotNet Method 1 (sslRotateNoPretrain.py + sslRotateNoPretrainFinetune.py) ,RotNet Method 2 (sslRotateWithPretrain.py + sslRotateWithPretrainFinetune.py), RotNet Method 3 (sslRotateWithPretrainLUNA.py + sslRotateWithPretrainLUNAFinetune.py), RotNet Self-Trans (sslRotateWithPretrainLUNAsslRotate.py + sslRotateWithPretrainLUNAsslRotateFinetune.py).
For SimCLR this would be: SimCLR Method 1 (sslSimClrNoPretrain.py + sslSimClrNoPretrainFinetune.py), SimCLR Method 2 (sslSimClrWithPretrain.py + sslSimClrWithPretrainFinetune.py), SimCLR Method 3 (sslSimClrWithPretrainLUNA.py + sslSimClrWithPretrainLUNAFinetune.py), SimCLR Self-Trans (sslSimClrWithPretrainLUNAsslSimClr.py + sslSimClrWithPretrainLUNAsslSimClrFinetune.py). Depending on your hardware this process could take quite a while (i.e. running pipe.py of RotNet took little more than hour for me with NVIDIA GTX 1060 6GB).
