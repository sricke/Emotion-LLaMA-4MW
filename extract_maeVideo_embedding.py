# @Time    : 6/24/23 9:33 AM
# @Author  : bbbdbbb
# @File    : extract_maeVideo_embedding.py
# @Description : load maeVideo model to extract video feature embedding

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_
from timm.models import create_model

import sys

sys.path.append('../../')
from dataset import FaceDataset

from maeVideo import models_vit
from collections import OrderedDict
from maeVideo.modeling_finetune import vit_large_patch16_224
from maeVideo.dataset_MER import train_data_loader, test_data_loader


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = False
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = True
                break
        if keep_flag:
            ignore_missing_keys.append(key)
        else:
            warn_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

    del state_dict
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='VoxCeleb_ckp49', help='pth of pretrain MAE model')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train vit_large_patch16')
    parser.add_argument('--nb_classes', default=6, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=1, type=int)

    params = parser.parse_args()

    print(f'==> Extracting maeVideo embedding...')
    video_dir = "/mnt/external_drive/Affectnet/Frames/"
    save_dir = "/mnt/external_drive/emotion-llama/Features/maeV_199_UTT"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load model
    model = models_vit.__dict__[params.model](
        num_classes=params.nb_classes,
        drop_path_rate=params.drop_path,
        global_pool=params.global_pool,
    )
    
    # Load pretrained checkpoint
    checkpoint_file = f"/path/to/pretrained/{params.pretrain_model}.pth"  # Set your checkpoint path
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model = load_state_dict(model, checkpoint_model)
    else:
        print(f"Warning: Checkpoint file not found: {checkpoint_file}")

    device = torch.device(params.device)
    model.to(device)
    model.eval()

    # Get data loader
    if params.dataset in ['MER2023', 'MER2024', 'EMER']:
        data_loader = test_data_loader(params.dataset, batch_size=params.batch_size)
    else:
        # Custom data loader for other datasets
        from torch.utils.data import DataLoader
        # Implement your custom dataset class here
        data_loader = DataLoader([], batch_size=params.batch_size)

    i = 1
    vids = len(data_loader)
    for images, video_name in data_loader:
        print(f"Processing video ({i}/{vids})...")
        i = i + 1
        images = images.to(device)
        embedding = model(images)

        print("embedding :", embedding.shape)
        embedding = embedding.cpu().detach().numpy()

        # save results
        EMBEDDING_DIM = max(-1, np.shape(embedding)[-1])

        video_name = video_name[0]

        csv_file = os.path.join(save_dir, f'{video_name}.npy')
        if params.feature_level == 'FRAME':
            embedding = np.array(embedding).squeeze()
            if len(embedding) == 0:
                embedding = np.zeros((1, EMBEDDING_DIM))
            elif len(embedding.shape) == 1:
                embedding = embedding[np.newaxis, :]
            np.save(csv_file, embedding)
        elif params.feature_level == 'BLK':
            embedding = np.array(embedding)
            if len(embedding) == 0:
                embedding = np.zeros((257, EMBEDDING_DIM))
            elif len(embedding.shape) == 3:
                embedding = np.mean(embedding, axis=0)
            np.save(csv_file, embedding)
        else:
            embedding = np.array(embedding).squeeze()
            if len(embedding) == 0:
                embedding = np.zeros((EMBEDDING_DIM,))
            elif len(embedding.shape) == 2:
                embedding = np.mean(embedding, axis=0)
            print("csv_file: ", csv_file)
            print("embedding: ", embedding)
            np.save(csv_file, embedding)

# MER2023
# python -u extract_maeVideo_embedding.py    --dataset='MER2023' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# EMER
# python -u extract_maeVideo_embedding.py    --dataset='EMER' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'

# MER2024
# python -u extract_maeVideo_embedding.py    --dataset='MER2024' --feature_level='UTTERANCE' --device='cuda:0'  --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo'
