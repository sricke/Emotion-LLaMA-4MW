# @Time    : 6/24/23 9:33 AM
# @Author  : bbbdbbb
# @File    : extract_maeVideo_embedding.py
# @Description : load maeVideo model to extract video feature embedding

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from timm.models.layers import trunc_normal_

import sys

sys.path.append('../../')
from dataset import FaceDataset

from maeVideo import models_vit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--pretrain_model', type=str, default='maeVideo_ckp199', help='pth of pretrain MAE model')
    parser.add_argument('--feature_name', type=str, default='maeVideo_ckp199', help='pth of pretrain MAE model')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train vit_large_patch16')
    parser.add_argument('--nb_classes', default=7, type=int, help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--batch_size', default=512, type=int)

    params = parser.parse_args()

    print(f'==> Extracting maeVideo embedding...')
    face_dir = '/mnt/external_drive/Affectnet/Features/Preprocessed_features/blur_0.0'
    save_dir = "/mnt/external_drive/MAEVideo/Features/Preprocessed_features/blur_0.0"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    MAX_FRAMES = 64

    # load model
    model = models_vit.__dict__[params.model](
        num_classes=params.nb_classes,
        drop_path_rate=params.drop_path,
        global_pool=params.global_pool,
    )
    if True:
        checkpoint_file = f"/home/ricke/emotion-llama/feature_extract/models/{params.pretrain_model}.pth"  # set your maeVideo path
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % checkpoint_file)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg.missing_keys)
        trunc_normal_(model.head.weight, std=2e-5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in tqdm(enumerate(vids, 1), total=len(vids), desc="Processing videos"):
        frames = np.load(os.path.join(face_dir, vid)).astype(np.float32)
        idx = np.linspace(0, frames.shape[0] - 1, MAX_FRAMES).astype(int)
        data = frames[idx, ...]
        #print("frames sampled shape: ", data.shape)
        
        data_loader = torch.utils.data.DataLoader(data, batch_size=MAX_FRAMES, num_workers=10, pin_memory=True)

        model.eval()
        with torch.no_grad():
            embeddings = []
            for frames in data_loader:
                frames = frames.permute(0,3,1,2) # (B, H, W, C) -> (B, C, H, W)
                frames = frames.to(device)
                outputs = model(frames)
                embedding = outputs

                embeddings.append(embedding.cpu().detach().numpy())
            embeddings = np.row_stack(embeddings)

        # save results
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        npy_file = os.path.join(save_dir, vid)
        if params.feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, EMBEDDING_DIM))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(npy_file, embeddings)
        elif params.feature_level == 'BLK':
            embeddings = np.array(embeddings)
            if len(embeddings) == 0:
                embeddings = np.zeros((257, EMBEDDING_DIM))
            elif len(embeddings.shape) == 3:
                embeddings = np.mean(embeddings, axis=0)
            np.save(npy_file, embeddings)
            
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((EMBEDDING_DIM,))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(npy_file, embeddings)

# EMER
# python -u extract_maeVideo_embedding.py    --feature_level='UTTERANCE' --pretrain_model='maeVideo_ckp199' --feature_name='maeVideo_ckp199'
