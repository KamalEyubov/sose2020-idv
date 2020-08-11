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
After installing the dependencies change into the directory of **RotNet** or **SimCLR** depending on which SSL approach you want to run (i.e. cd sose2020-idv/ResNet50/rotnet).<br>
After changing into the directory of rotnet or simClr run the **pipe.py** file (i.e. python pipe.py) with python3.<br> This starts the pipeline of running all the methods one after another.<br> For **RotNet** this would be: 
- no SSL and no pre-training (noSslNoPretrain.py)
- no SSL with pretraining (noSslWithPretrain.py)
- RotNet Method 1 (sslRotateNoPretrain.py + sslRotateNoPretrainFinetune.py)
- RotNet Method 2 (sslRotateWithPretrain.py + sslRotateWithPretrainFinetune.py)
- RotNet Method 3 (sslRotateWithPretrainLUNA.py + sslRotateWithPretrainLUNAFinetune.py)
- RotNet Self-Trans (sslRotateWithPretrainLUNAsslRotate.py + sslRotateWithPretrainLUNAsslRotateFinetune.py).<br><br>

For **SimCLR** this would be:
- SimCLR Method 1 (sslSimClrNoPretrain.py + sslSimClrNoPretrainFinetune.py)
- SimCLR Method 2 (sslSimClrWithPretrain.py + sslSimClrWithPretrainFinetune.py)
- SimCLR Method 3 (sslSimClrWithPretrainLUNA.py + sslSimClrWithPretrainLUNAFinetune.py)
- SimCLR Self-Trans (sslSimClrWithPretrainLUNAsslSimClr.py + sslSimClrWithPretrainLUNAsslSimClrFinetune.py).<br><br> Depending on your hardware this process could take quite a while (i.e. running pipe.py of RotNet took **little more than hour** for me with **NVIDIA GTX 1060 6GB**).<br>
It is also possible to run each of the files listed above on its own outside of the pipeline. To do this just call the file you want execute with python3 (i.e. python sslSimClrNoPretrain.py). One thing to keep in mind is that the corresponding **ssl stage** has to be run **before** the **finetuning stage** if there is no ssl model already available which the finetuning stage can load.

#### Dependencies
```
absl-py==0.9.0
astunparse==1.6.3
cachetools==4.1.1
certifi==2020.6.20
chardet==3.0.4
cycler==0.10.0
decorator==4.4.2
future==0.18.2
gast==0.3.3
google-auth==1.18.0
google-auth-oauthlib==0.4.1
google-pasta==0.2.0
grpcio==1.30.0
h5py==2.10.0
idna==2.10
imageio==2.8.0
joblib==0.15.1
jsonpatch==1.26
jsonpointer==2.0
Keras-Preprocessing==1.1.2
kiwisolver==1.2.0
Markdown==3.2.2
matplotlib==3.2.2
networkx==2.4
numpy==1.19.0
oauthlib==3.1.0
opt-einsum==3.2.1
packaging==20.4
pandas==1.0.5
Pillow==7.1.2
protobuf==3.12.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
PyWavelets==1.1.1
pyzmq==19.0.1
requests==2.24.0
requests-oauthlib==1.3.0
rsa==4.6
scikit-image==0.17.2
scikit-learn==0.23.1
scipy==1.4.1
six==1.15.0
sklearn==0.0
tensorboard==2.2.2
tensorboard-plugin-wit==1.7.0
tensorflow==2.2.0
tensorflow-estimator==2.2.0
termcolor==1.1.0
threadpoolctl==2.1.0
tifffile==2020.6.3
torch==1.5.1
torchfile==0.1.0
torchvision==0.6.1
tornado==6.0.4
tqdm==4.47.0
urllib3==1.25.9
websocket-client==0.57.0
Werkzeug==1.0.1
wrapt==1.12.1
```
Above all the packages are listed which are installed in my virtualenvironment.<br> 
If you are also using a virtualenvironment all copy and paste the dependencies in a textfile (i.e. denpendencies_resnet.txt)<br>
Then just create a new virtualenvironment in your shell with the line:<br>
```
python -m venv <name of virtualenvironment>
```
<br>
Then activate your virtualenvironment:<br>
```
source <name of virtualenvironment>/bin/activate
```
<br>
Finally install the dependencies:<br>
```
(<name of virtualenvironment>)$ pip install -r path/to/denpendencies_resnet.txt
```
<br>
If you do not want to use a virtualenvironment just ignore the first 2 lines.
