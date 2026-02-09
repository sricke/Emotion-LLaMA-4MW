import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json
from collections import Counter

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class HCMWDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, fold, is_train, use_weights=True):
        
        self.vis_root = vis_root
        #self.image_root = '/home/ricke/emotion-llama/feature_extract/factory/raw/mer'
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.is_train = is_train
        self.mind_wandering_instruction = "Please classify the video into one of these categories: mind wandering or non-mind wandering."
        
        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = pd.read_csv(ann_path)
        self.mw = {"NonMW":"non-mind wandering", "MW":"mind wandering"}
        if is_train and use_weights:
            self.get_class_weights()
        else:
            self.class_weights = None
        
        if is_train:
            # use all folds except the current fold for training
            self.tmp = self.tmp[self.tmp['test_fold_num'] != fold] # get the data for the current fold
        else:
            # use the current fold for validation
            self.tmp = self.tmp[self.tmp['test_fold_num'] == fold] # get the data for the current fold
            
        print(('video number:%d' % (len(self.tmp))))

        # emos = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']

        """self.mw2idx, self.idx2mw = {}, {}
        for ii, mw in enumerate(emos): self.mw2idx[mw] = ii
        for ii, mw in enumerate(emos): self.idx2mw[ii] = mw"""

        # Set custom feature directory
        self.feature_dir = "/home/ricke/emotion-llama/feature_extract/features"
        
    def get_class_weights(self):
        class_weights = {}
        labels = [row["mw_label"] for _, row in self.tmp.iterrows()]
        class_counts = Counter(labels)
        unique_classes = sorted(class_counts.keys())
        
        assert len(unique_classes) == len(self.mw.keys())       
        total_samples = len(labels)
        for class_name in class_counts:
            # Weight = total_samples / (num_classes * class_count)
            # This gives more weight to underrepresented classes
            weight = total_samples / (len(unique_classes) * class_counts[class_name])
            class_weights[class_name] = weight

        self.class_weights = class_weights
        
        # Log class distribution
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {class_weights}")

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp.iloc[index]
        video_name = t["uuid"]
        if video_name.endswith(".mp4"):
            video_name = video_name[:-4]
        # use peak frame image as the image for eva encoder
        image_file = '{}_peak_frame.png'.format(video_name)
        image_path = os.path.join(self.vis_root, video_name, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)


        # task == emotion
        task = "classification"
        caption = t["mw_label"] # llama2 putput only emotion class
        caption = self.text_processor(caption)
        class_weight = self.class_weights[t["mw_label"]] if self.class_weights else 1.0
        mw_label = self.mw[t["mw_label"]]
        sentence = ""
        #character_line = "The person in video says: {}. ".format(sentence)
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> [{}] {} ".format(task, self.mind_wandering_instruction)

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "class_weight": class_weight,
            "label": mw_label,
            "video_id": video_name,
        }
    
    def extract_frame(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        if not success:
            raise ValueError("Failed to read video file:", video_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_capture.release()

        return frame_rgb


    def get(self, video_name):
        # FaceMAE feature - load from custom feature directory
        FaceMAE_feats_path = os.path.join(self.feature_dir, 'mae_340_UTT', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature - load from custom feature directory
        VideoMAE_feats_path = os.path.join(self.feature_dir, 'maeV_199_UTT', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature - return empty tensor with same shape as FaceMAE_feats
        Audio_feats = torch.zeros_like(FaceMAE_feats)

        return FaceMAE_feats, VideoMAE_feats, Audio_feats

