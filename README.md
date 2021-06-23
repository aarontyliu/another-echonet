# Precision-Medicine-EchoNet


This repo provides scripts to load the EchoNet-Dynamic database (https://echonet.github.io/dynamic/index.html), and train a baseline model with UNet, ResNet-18 and bidirectional LSTM. Baseline model checkpoint can be downloaded from [here](https://drive.google.com/file/d/1wvTwb3RYrIqviocweQOxsRQN5bgsIiN2/view?usp=sharing). You can download the baseline checkpoint and place it under "Precision-Medicine-EchoNet/checkpoints" directory.

To train a model, one can run the follow script. There are arguments that can be utilized to adjust the hyperparameter settings, training strategy and log frequency. "--load" argument will allow user to load pretrained model weights (it needs to be placed in "Precision-Medicine-EchoNet/checkpoints" directory).


Example training command:
```python
python train.py --lr1 1e-5 --lr2 1e-4 --batch_size 16 --epochs 40 --use_gt_ef --log_every 200 --device 'cuda' --load 'foo.pt'
```

To perform instance-level inference, run the following code (paths need to be handled in current design):
```python
python inference.py
```


## Sample testing output:

Input video                       |  Model output
:--------------------------------:|:----------------------------------------:
<img src="pics/0X347C08CBDD9C7630.gif" width="250" height="250"/>  |  <img src="pics/output-0X347C08CBDD9C7630.gif" width="250" height="250"/>

The blue area is the mask produced by UNet, and the green line denotes the volume predicted by ResNet-18.


## Design flow:
![](pics/workflow.gif)


## Future plans
- Replace UNet with Residual UNet (implemented and tested) to aim for better segmentation performance and consider using region-based loss (eg. Dice)
- Adopt other encoder families (eg. EfficientNet family) to better volume prediction
- Replace LSTM with Transformers
- Apply early-regularization in segmentation network to resolve noisy label issue
- Denoise with clustering results on volume/segmentation estimates
- Perform deep compression to reduce model size and increase inference speed 
- Learn general representation of echocardiogram video for downstream tasks

