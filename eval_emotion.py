import csv
import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.registry import registry

from minigpt4.datasets.datasets.first_face import FeatureFaceDataset
from minigpt4.datasets.datasets.mer2024 import MER2024Dataset
from minigpt4.datasets.datasets.hcmw import HCMWDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from minigpt4.datasets.data_utils import prepare_sample


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='feature_face_caption', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

print("cfg:", args)
cfg = Config(args)


model, vis_processor = init_model(args)

model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

save_path = cfg.run_cfg.save_path

text_processor_cfg = cfg.datasets_cfg.feature_face_caption.text_processor.train
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

def extract_model_features(model, samples):
    
    # samples['image'].shape -> [1, 3, 448, 448]
    # samples['video_features'].shape -> [1, 3, 1024] 
    # prepare the embedding to condition and the embedding to regress
    cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, inputs_llama  = \
        model.preparing_embedding(samples) # handles the image and video features
     # after this, cond_embeds.shape -> [1, 259, 4096]
     # image features -> [1, 256, 4096] after eva model and projector layer
     # video features -> [1, 2, 4096] after projector layer (audio is not used)
     # cls token -> [1, 1, 4096]

    # concat the embedding to condition and the embedding to regress
    inputs_embeds, attention_mask, input_lens = \
        model.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

    # get bos token embedding
    bos = torch.ones_like(part_targets[:, :1]) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.embed_tokens(bos)
    bos_atts = cond_atts[:, :1]

    # add bos token at the begining
    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

    # ensemble the final targets
    targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                        dtype=torch.long).to(model.device).fill_(-100)

    for i, target in enumerate(part_targets):
        targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

    with model.maybe_autocast():
        with torch.no_grad():
            outputs = model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                reduction="mean",
                output_hidden_states=True,
            ) # outputs.hidden_states[-1] -> (batch_size, 309, 4096)

    feature_list = []
    with model.maybe_autocast():
        inputs_llama = inputs_llama.cpu().detach().numpy()
        video_projected_features = inputs_llama[:, 257:259, :]
        facemae_projected_features = video_projected_features[ :, 0, :]
        videomae_projected_features = video_projected_features[ :, 1, :]
        image_features = inputs_llama[:, :256, :]
        
        
        outputs_hidden_states = outputs.hidden_states[-1].cpu().detach().numpy()
        features_idx = [6, 7, 8, 9] # took this feature list from authors code 
        llama_features = outputs_hidden_states[:, features_idx, :]  # (batch_size, 4, 4096)
        
        return facemae_projected_features, videomae_projected_features, image_features, llama_features

print(args.dataset)
if 'feature_face_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["feature_face_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["feature_face_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["feature_face_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["feature_face_caption"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = FeatureFaceDataset(vis_processor, text_processor, img_path, eval_file_path)
    # print(data)
    # print(data[0])
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    targets_list = []  
    answers_list = [] 
    names_list   = []
    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        # print("instruction_input:", instruction_input)
        targets = batch['answer']
        video_features = batch['video_features']

        texts = prepare_texts(instruction_input, conv_temp)
        answers = model.generate(images, video_features, texts, max_new_tokens=max_new_tokens, do_sample=False)
        
        for j in range(len(answers)):
            # print("raw answer:", answers[j])
            answers[j] = answers[j].split(" ")[-1]
            targets[j] = targets[j].split(" ")[-1]
            if answers[j] not in ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']:
                print("Error: ", answers[j], " Target:", targets[j])
                answers[j] = 'neutral'
        
        targets_list.extend(targets)  
        answers_list.extend(answers)
        names_list.extend(batch['image_id'])

    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted')
    recall = recall_score(targets_list, answers_list, average='weighted')
    f1 = f1_score(targets_list, answers_list, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    cm = confusion_matrix(targets_list, answers_list)
    print(cm)

if 'mer2024_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["mer2024_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["mer2024_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["mer2024_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["mer2024_caption"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = MER2024Dataset(vis_processor, text_processor, img_path, eval_file_path)
    # print(data)
    # print(data[0])
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    targets_list = []  
    answers_list = [] 
    names_list   = []
    for batch in tqdm(eval_dataloader):
        images = batch['image']
        instruction_input = batch['instruction_input']
        # print("instruction_input:", instruction_input)
        targets = batch['answer']
        video_features = batch['video_features']

        texts = prepare_texts(instruction_input, conv_temp)
        answers = model.generate(images, video_features, texts, max_new_tokens=max_new_tokens, do_sample=False)
        
        for j in range(len(answers)):
            # print("raw answer:", answers[j])
            answers[j] = answers[j].split(" ")[-1]
            targets[j] = targets[j].split(" ")[-1]
            if answers[j] not in ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']:
                print("Error: ", answers[j], " Target:", targets[j])
                answers[j] = 'neutral'
        
        targets_list.extend(targets)  
        answers_list.extend(answers)
        names_list.extend(batch['image_id'])

    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted')
    recall = recall_score(targets_list, answers_list, average='weighted')
    f1 = f1_score(targets_list, answers_list, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    cm = confusion_matrix(targets_list, answers_list)
    print(cm)

if 'hcmw_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["hcmw_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["hcmw_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["hcmw_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["hcmw_caption"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = HCMWDataset(vis_processor, text_processor, img_path, eval_file_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    targets_list = []  
    answers_list = [] 
    names_list   = []
    facemae_projected_features_list = []
    videomae_projected_features_list = []
    image_projected_features_list = []
    llama_features_list = []
    names_list = []
    for batch in tqdm(eval_dataloader):
        images = batch['image']
        instruction_input = batch['instruction_input']
        video_features = batch['video_features']
        
        # Convert to device (dtype conversion will be handled by autocast in encode_img)
        images = images.to(model.device)
        video_features = video_features.to(model.device)
        
        # Create samples dict (similar to what forward() expects)
        samples = {
            'image': images,
            'video_features': video_features,
            'instruction_input': instruction_input,
            'answer': [''] * len(instruction_input)  # Dummy answers, we won't use them
        }
        
        facemae_projected_features, videomae_projected_features, image_projected_features, llama_features = extract_model_features(model, samples)
        
        facemae_projected_features_list.append(facemae_projected_features)
        videomae_projected_features_list.append(videomae_projected_features)
        image_projected_features_list.append(image_projected_features)
        llama_features_list.append(llama_features)
        names_list.extend(batch['image_id'])

   
    facemae_projected_features_list = np.concatenate(facemae_projected_features_list, axis=0)
    videomae_projected_features_list = np.concatenate(videomae_projected_features_list, axis=0)
    image_projected_features_list = np.concatenate(image_projected_features_list, axis=0)
    llama_features_list = np.concatenate(llama_features_list, axis=0)
  
    save_dir = "features"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "embeddings_emotion_llama_selectedS.npy"), llama_features_list)
    np.save(os.path.join(save_dir, "facemae_projected_features.npy"), facemae_projected_features_list)
    np.save(os.path.join(save_dir, "videomae_projected_features.npy"), videomae_projected_features_list)
    np.save(os.path.join(save_dir, "image_features.npy"), image_projected_features_list)

    with open(os.path.join(save_dir, "names_embeddings.txt"), "w") as f:
        for name in names_list:
            f.write(f"{name}\n")

# torchrun  --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
# torchrun  --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset mer2024_caption
# torchrun  --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset hcmw_caption
