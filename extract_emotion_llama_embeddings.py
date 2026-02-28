import torch
import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm
from minigpt4.common.eval_utils import init_model, eval_parser
import argparse
def extract_model_features(model, samples):
    
    # samples['image'].shape -> [1, 3, 448, 448]
    # samples['video_features'].shape -> [1, 3, 1024] 
    # prepare the embedding to condition and the embedding to regress
    cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, inputs_llama,_  = \
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

    with model.maybe_autocast():
        outputs_hidden_states = outputs.hidden_states[-1].cpu().detach().numpy()
        # [6,7,8, 9]
        #features_idx = [0,1, 2, 3, 4, 5, 6, 7, 8, 9] # took this feature list from authors code 
        llama_features = outputs_hidden_states[:, :, :]  # (batch_size, 4, 4096)
        
        return llama_features
    
if __name__ == "__main__":
    parser = eval_parser()
    # Make cfg_path optional with default value
    for action in parser._actions:
        if '--cfg-path' in action.option_strings:
            action.required = False
            action.default = "eval_configs/extract_embeddings.yaml"
            break
    
    parser.add_argument("--output_dir", type=str, default="/mnt/external_drive/Emotion-LLaMA/Features/Preprocessed_features_all/blur_0.0", 
                       help="Directory to save extracted embeddings")
    args = parser.parse_args()
    
    # Initialize model and vis_processor
    print("Initializing model...")
    model, vis_processor = init_model(args)
    model.eval()
    
    # Set paths
    peak_frame_dir = "/home/ricke/emotion-llama/HCMW/factory/raw/mer"
    face_dir = '/mnt/external_drive/Affectnet/Features/Preprocessed_features/blur_0.0'
    mae_feature_dir = "/mnt/external_drive/MAE/Features/Preprocessed_features/blur_0.0"
    maevideo_feature_dir = "/mnt/external_drive/MAEVideo/Features/Preprocessed_features/blur_0.0"
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of video directories
    if not os.path.exists(peak_frame_dir):
        print(f"Error: Peak frame directory not found: {peak_frame_dir}")
        sys.exit(1)
    
    # Process each video
    device = next(model.parameters()).device
    vids = os.listdir(face_dir)
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in tqdm(enumerate(vids, 1), total=len(vids), desc="Processing videos"):
        video_name = vid.replace('.npy', '')
        # Load peak frame image
        image_file = '{}_peak_frame.png'.format(video_name)
        image_path = os.path.join(peak_frame_dir, video_name, image_file)
            
        image = Image.open(image_path).convert("RGB")
        image = vis_processor(image).unsqueeze(0).to(device)
        
        # Load FaceMAE features
        facemae_feats_path = os.path.join(mae_feature_dir, video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(facemae_feats_path))
        
        # Load VideoMAE features
        videomae_feats_path = os.path.join(maevideo_feature_dir, video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(videomae_feats_path))
        
        # Audio feature - return empty tensor with same shape as FaceMAE_feats
        Audio_feats = torch.zeros_like(FaceMAE_feats)
        
        # Ensure proper dimensions
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        
        # Concatenate video features: [FaceMAE, VideoMAE, Audio] -> shape: [3, feature_dim]
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)
        # Add batch dimension -> shape: [1, 3, feature_dim]
        video_features = video_features.unsqueeze(0).to(device)
        
        # Prepare instruction
        task = "emotion"
        emotion_instruction = "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt."
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> [{}] {} ".format(task, emotion_instruction)
        
        # Prepare samples (matching the dataset structure)
        samples = {
            "image": image,
            "video_features": video_features,
            "instruction_input": [instruction],
            "answer": [""],  # Dummy answers, we won't use them
            "class_weight": [1.0]  # Default weight
        }
        
        # Extract features
        llama_features = extract_model_features(model, samples)
        
        # Remove batch dimension: (1, 4, 4096) -> (4, 4096)
        llama_features = llama_features.squeeze(0)
        # Save embeddings as .npy file (same pattern as extract_mae_embedding.py)
        npy_file = os.path.join(output_dir, video_name + '.npy')
        np.save(npy_file, llama_features)
            