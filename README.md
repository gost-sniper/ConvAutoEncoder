# Convolutionnal Auto-Encoder (CAE)

This is an implementation of a paper 'Gyroscope-Aided Motion Deblurring with Deep Networks' 

## Installation

1 - you start by clonning this repo:
```
git clone https://github.com/gost-sniper/ConvAutoEncoder
cd ConvAutoEncoder
```
2 - create a virtual environment :
```
python -m venv venv/
source venv/bin/activate  # for linux/macos
or 
venv/Scripts/activate  # for windows
```
3 - then installing dependencies :
```
pip install -r requirements.txt
```
## Data 

The data is stored in the `data` folder as following:
```
ConvAutoEncoder
├── data
│   ├── blurry_images
│   │   ├── (1).jpg
│   │   ├── (2).jpg
│   │   ├── (3).jpg
│   │   └── (4).jpg
│   └── normal_images
│       ├── (1).jpg
│       ├── (2).jpg
│       ├── (3).jpg
│       └── (4).jpg
├── DataSet.py
├── gyroModel.py
├── main.py
├── models
│   └── ConvAutoEncoder
│       ├── model_001.pth
│       ├── model_002.pth
│       ├── model_003.pth
│       ├── model_004.pth
│       └── model_005.pth
├── Opers.py
├── README.md
└── requirements.txt

  
```
## Training 

Print parameters:

```bash
./main.py --help
```
```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--normal_data NORMAL_DATA]
               [--blurry_data BLURRY_DATA] [--epoch EPOCH] [--lr LR]

PyTorch ConvAutoEncoder

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size
  --normal_data NORMAL_DATA
                        path of the normal data images
  --blurry_data BLURRY_DATA
                        path of the blurry data images
  --epoch EPOCH         number of train epoches
  --lr LR               initial learning rate for Adam
```

You can use the arguments `--normal_data` and `--blurry_data` if stored elsewhere :

```
python main.py --normal_data <PATH_NORMAL_IMAGE> --blurry_data <PATH_BLURRY_IMAGE>
``` 
## Using the model 

after we done with the trainning you can use the model as the following :
```python
import os

import torch
from torchvision import transforms

import DataSet
from Opers import findLastCheckpoint, prepareLoaders

save_dir = os.path.join('models', 'ConvAutoEncoder')

Last_checkpoint = findLastCheckpoint(save_dir)

model_location = os.path.join(save_dir, 'model_%03d.pth' % Last_checkpoint)

model = torch.load(model_location)

with torch.no_grad():
    model.eval()
    ...     # you can call the model for deblurring 


```
## References
The implemented paper [gyro](https://arxiv.org/abs/1810.00986 "Gyroscope-Aided Motion Deblurring with Deep Networks").
