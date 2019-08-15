# Convolutionnal Auto-Encoder (CAE)

This is an implementation of a paper 'Gyroscope-Aided Motion Deblurring with Deep Networks' 

## Data 

The data is stored in the `data` folder as following:
```
/ConvAutoEncoder
|  /data
|  |  /blurry_images
|  |  |   (1).jpg
|  |  |  (2).jpg
|  |  |   (3).jpg
|  |  |   .
|  |  |   .
|  |  /normal_images
|  |  |   (1).jpg
|  |  |   (2).jpg
|  |  |   (3).jpg
|  |  |   .
|  |  |   .
|  /models
|  DataSet.py
|  gyroModel.py
|  main.py
|  Opers.py
  
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

You can specify the paths if stored elsewhere. this is an example of

```
python main.py --epoch 44 --lr 0.001 --normal_data <PATH_NORMAL_IMAGE> --blurry_data <PATH_BLURRY_IMAGE>
```

## References
the paper implemented [gyro](https://arxiv.org/abs/1810.00986 "The best search engine for privacy").
