# Precision-Medicine-EchoNet

This repo provides scripts to load the EchoNet-Dynamic database (<https://echonet.github.io/dynamic/index.html>), and train a baseline model with UNet, ResNet-18 and bidirectional LSTM. 

To train a model, one can run the follow script. There are arguments that can be utilized to adjust the hyperparameter settings, training strategy and log frequency. "--load" argument will allow user to load pretrained model weights.

Example training command:

```python
python train.py --lr1 1e-5 --lr2 1e-4 --batch_size 16 --epochs 40 --use_gt_ef --log_every 200 --device 'cuda' --load 'foo.pt'
```

To perform instance-level inference, run the following code (paths need to be handled in current design):

```python
python inference.py
```

## Sample testing output

Input video                       |  Model output
:--------------------------------:|:----------------------------------------:
<img src="/Users/aaronliu/Projects/Archive/Precision-Medicine-EchoNet/pics/example.gif" alt="example" style="zoom:200%;" />  | <img src="/Users/aaronliu/Projects/Archive/Precision-Medicine-EchoNet/pics/output.gif" alt="output" style="zoom:155%;" /> 

The blue area is the mask produced by UNet, and the green line denotes the volume predicted by ResNet-18.

## Design flow

![workflow](/Users/aaronliu/Projects/Archive/Precision-Medicine-EchoNet/pics/workflow.gif)

## Future plans

- Replace UNet with Residual UNet (implemented and tested) to aim for better segmentation performance and consider using region-based loss (eg. Dice)
- Adopt other encoder families (eg. EfficientNet family) to better volume prediction
- Replace LSTM with Transformers
- Apply early-regularization in segmentation network to resolve noisy label issue
- Denoise with clustering results on volume/segmentation estimates
- Perform deep compression to reduce model size and increase inference speed
- Learn general representation of echocardiogram video for downstream tasks
