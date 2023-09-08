import os
import numpy as np
import time
import torch
from torch import nn, optim, distributions

from model import ProbabilisticSegmentationNet, InjectionUNet, InjectionConvEncoder

from dataset_2d import Brats17
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import nibabel as nib

def run():
    
    # Para
    in_channels = 2
    latent_size = 3
    labels = [0, 1]  
    patch_size = 128
    
    # Base
    n_epochs=50000
    batch_size=10
    patch_size=patch_size
    in_channels=in_channels
    out_channels=len(labels)
    latent_size=latent_size
    seed=1
    device="cuda"

    # Model
    Model=ProbabilisticSegmentationNet
    Model_kwargs={
        "in_channels": in_channels,
        "out_channels": len(labels),
        "num_feature_maps": 24,
        "latent_size": latent_size,
        "depth": 5,
        "latent_distribution": distributions.Normal,
        "task_op": InjectionUNet,
        "task_kwargs": {
            "output_activation_op": nn.LogSoftmax,
            "output_activation_kwargs": {"dim": 1},
            "activation_kwargs": {"inplace": True}
        },
#         "prior_op": InjectionConvEncoder,
#         "prior_kwargs": {
#             "in_channels": 1,
#             "out_channels": latent_size * 2,
#             "depth": 5,
#             "block_depth": 2,
#             "num_feature_maps": 24,
#             "feature_map_multiplier": 2,
#             "activation_kwargs": {"inplace": True},
#             "norm_depth": 2,
#         },
        "posterior_op": InjectionConvEncoder,
        "posterior_kwargs": {
            "in_channels": in_channels + 1,
            "out_channels": latent_size * 2,
            "depth": 5,
            "block_depth": 2,
            "num_feature_maps": 24,
            "feature_map_multiplier": 2,
            "activation_kwargs": {"inplace": True},
            "norm_depth": 2,
        },
    }
    model_init_weights_args=[nn.init.kaiming_uniform_, 0]
    model_init_bias_args=[nn.init.constant_, 0]

    # Learning
    optimizer=optim.Adam
    optimizer_kwargs={"lr": 1e-4}
    scheduler=optim.lr_scheduler.StepLR
    scheduler_kwargs={"step_size": 200, "gamma": 0.985}
    criterion_segmentation=nn.NLLLoss()
    criterion_segmentation_kwargs={"reduction": "sum"}
    criterion_latent=distributions.kl_divergence
    criterion_latent_kwargs={}
    criterion_latent_init=False
    criterion_segmentation_seg_onehot=False
    criterion_segmentation_weight=1.0
    criterion_latent_weight=1.0
    criterion_segmentation_seg_dtype=torch.long
    
    # model
    model = Model(**Model_kwargs).cuda()
    if model_init_weights_args is not None:
        if isinstance(model_init_weights_args, dict):
            for key, val in model_init_weights_args.items():
                try:
                    model._modules[key].init_weights(*val)
                except Exception as e:
                    print("Tried to initialize {} with {}, but got the following error:".format(key, val))
                    print(repr(e))
        #elif isinstance(model_init_weights_args, (list, tuple)):
        #    model.init_weights(model_init_weights_args)
        #else:
        #    raise TypeError("model_init_weights_args must be dict, list or tuple, but found {}.".format(type(model_init_weights_args)))
    if model_init_bias_args is not None:
        if isinstance(model_init_bias_args, dict):
            for key, val in model_init_bias_args.items():
                try:
                    model._modules[key].init_bias(*val)
                except Exception as e:
                    print("Tried to initialize {} with {}, but got the following error:".format(key, val))
                    print(repr(e))
        #elif isinstance(model_init_bias_args, (list, tuple)):
        #    model.init_bias(model_init_bias_args)
        #else:
        #    raise TypeError("model_init_bias_args must be dict, list or tuple, but found {}.".format(type(model_init_bias_args)))
    #model.load_state_dict(torch.load('/userhome/GUOXUTAO/2023_08/90/11/check/pth/0_4444_0_0_10_0509_16:19:42.pth'))

    # optimizer
    optimizer = optimizer(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler(optimizer, **scheduler_kwargs)    
    
    # Dataloader
    train_data=Brats17('',train=True)
    train_dataloader = DataLoader(train_data,batch_size = 32,shuffle=False,num_workers=4)
    
    
if __name__ == '__main__':

    parser = get_default_experiment_parser()
    parser.add_argument("-p", "--patch_size", type=int, nargs="+", default=112)
    parser.add_argument("-in", "--in_channels", type=int, default=4)
    parser.add_argument("-lt", "--latent_size", type=int, default=3)
    parser.add_argument("-lb", "--labels", type=int, nargs="+", default=[0, 1, 2, 3])
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults(patch_size=args.patch_size, in_channels=args.in_channels, latent_size=args.latent_size, labels=args.labels)
    run()
