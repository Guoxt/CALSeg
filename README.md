### CALSeg Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing
------

<div style="text-align: justify;"> 

##### In practical medical image segmentation tasks, ensuring confidence calibration is crucial. However, medical image segmentation typically relies on hard labels (one-hot vectors), and when minimizing the cross-entropy loss, the model’s softmax predictions are compelled to align with hard labels, resulting in over-confident predictions. To alleviate above problems, this study proposes a novel framework on calibration of medical image segmentation, called CALSeg. The Variational Label Smoothing (VLS) method is innovatively proposed, which learns the latent joint distribution of images and labels through variational inference to capture complex relationships between images and labels. This enables the effective estimation of latent soft labels by learning pixel-level information and semantic probability distribution features. The training of a neural network based on estimated soft labels provides a regularization effect, effectively preventing model overfitting and improving the calibration of the model. Comprehensive experiments on two medical image segmentation datasets demonstrate that CALSeg achieved optimal network calibration while also improving segmentation accuracy.

</div>

------
### Framework
------

<img src="https://github.com/Guoxt/CALSeg/blob/master/image.png" alt="Image Alt Text" style="width:1000px; height:auto;">

------
### Run MRNet Code

1. train

```python main.py --patch_size 12 --in_channels 3 --latent_size 8 --labels 2```                        # Setting Training Parameters

2. test

```python test.py --patch_size 12 --in_channels 3 --latent_size 8 --labels 2```                        # Setting Testing Parameters
