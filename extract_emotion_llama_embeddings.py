import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import csv
from PIL import Image
from tqdm import tqdm
from minigpt4.common.eval_utils import init_model, eval_parser, prepare_texts
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
import argparse
 
# Emotion labels (must match the instruction)
EMOTION_LABELS = ["happy", "sad", "neutral", "angry", "worried", "surprise", "fear", "contempt", "doubt"]
 
 
def build_emotion_token_map(tokenizer, emotion_labels):
    """
    Pre-compute token IDs for each emotion label.
    Some emotions tokenize to multiple tokens (e.g. "surprise" -> [tok1, tok2]).
 
    Returns:
        token_map: dict {emotion_str: [token_id, ...]}
        max_tokens: int, max token length across all emotion labels
    """
    token_map = {}
    for emotion in emotion_labels:
        ids = tokenizer.encode(emotion, add_special_tokens=False)
        token_map[emotion] = ids
        print(f"  '{emotion}' -> {ids} ({len(ids)} token{'s' if len(ids) > 1 else ''})")
    max_tokens = max(len(ids) for ids in token_map.values())
    return token_map, max_tokens
 
 
def compute_confidence_from_scores(generate_scores, token_map, pred_emotion):
    """
    Compute true autoregressive joint probabilities for all emotions using
    the per-step logits returned by HuggingFace generate (output_scores=True).
 
    `generate_scores` is a tuple of tensors, one per generated token step.
    Each tensor has shape (batch_size, vocab_size).
    The emotion label tokens are always the LAST N steps of generation,
    where N = number of tokens in that emotion label.
 
    For each emotion:
        joint_log_prob = sum of log P(token_i) at step [-N + i] for i in range(N)
 
    These are true autoregressive probabilities: each step is conditioned on
    all previously generated tokens, not on a dummy teacher-forced answer.
 
    We then softmax over all emotion joint log-probs to get a normalized
    confidence distribution.
 
    Args:
        generate_scores: tuple of (batch, vocab_size) tensors — one per generated step
        token_map:       dict {emotion_str: [token_id, ...]}
        pred_emotion:    the emotion string the model actually generated
 
    Returns:
        probs_dict:  {emotion: normalized confidence float}
        confidence:  float, confidence of pred_emotion specifically
    """
    num_steps = len(generate_scores)  # total number of generated tokens
 
    joint_log_probs = {}
    for emotion, token_ids in token_map.items():
        n = len(token_ids)
        if n > num_steps:
            # Model generated fewer tokens than this emotion needs — assign -inf
            joint_log_probs[emotion] = float('-inf')
            continue
 
        log_p = 0.0
        for step_offset, tok_id in enumerate(token_ids):
            # Emotion tokens occupy the last n steps: [num_steps-n, ..., num_steps-1]
            step_idx = num_steps - n + step_offset
            step_logits = generate_scores[step_idx][0]  # (vocab_size,) — squeeze batch dim
            # log_softmax is used to convert the logits to a probability distribution
            # steps logits is 32k shape -> vocab
            log_p += F.log_softmax(step_logits.float(), dim=-1)[tok_id].item()
 
        joint_log_probs[emotion] = log_p
 
    # Softmax over joint log-probs -> normalized confidence distribution
    emotions = list(joint_log_probs.keys())
    scores_tensor = torch.tensor([joint_log_probs[e] for e in emotions])
    # normalized so probabilities sum to 1 over 9 emotions but persists the relative probabilities  
    normalized = F.softmax(scores_tensor, dim=0)
 
    probs_dict = {e: normalized[i].item() for i, e in enumerate(emotions)}
    confidence = probs_dict.get(pred_emotion, 0.0)
 
    return probs_dict, confidence
 
 
def get_emotion_via_generate(model, images, video_features, instruction_input, token_map, max_new_tokens=500):
    """
    Run generation with output_scores=True to get both:
      - predicted emotion string (last-word parsing, original Emotion-LLaMA method)
      - per-step logits for true autoregressive confidence scoring
 
    Args:
        model:             Emotion-LLaMA model
        images:            Tensor [1, 3, H, W]
        video_features:    Tensor [1, 3, 1024]
        instruction_input: List of instruction strings
        token_map:         {emotion: [token_id, ...]} from build_emotion_token_map
        max_new_tokens:    Max tokens to generate
 
    Returns:
        pred_emotion:  str, predicted emotion (last-word parse)
        probs_dict:    {emotion: normalized confidence}
        confidence:    float, confidence of pred_emotion
        raw_answer:    str, full decoded generation output
    """
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    if isinstance(instruction_input, str):
        instruction_input = [instruction_input]
    texts = prepare_texts(instruction_input, conv_temp)
 
    # Encode inputs — replicating the non-input_embeds branch of model.generate
    img_embeds, atts_img = model.encode_img(images, video_features)
    image_lists = [[image_emb[None]] for image_emb in img_embeds]
    batch_embs = [model.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]
 
    max_len = max(emb.shape[1] for emb in batch_embs)
    emb_dim = batch_embs[0].shape[2]
    dtype = batch_embs[0].dtype
    device = batch_embs[0].device
 
    embs = torch.zeros([1, max_len, emb_dim], dtype=dtype, device=device)
    attn_mask = torch.zeros([1, max_len], dtype=torch.int, device=device)
    for i, emb in enumerate(batch_embs):
        emb_len = emb.shape[1]
        embs[i, -emb_len:] = emb[0]
        attn_mask[i, -emb_len:] = 1
 
    with model.maybe_autocast():
        with torch.no_grad():
            hf_output = model.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=1,                   # must be greedy for scores to be valid token probs
                do_sample=False,
                min_length=1,
                output_scores=True,            # return per-step logits
                return_dict_in_generate=True,  # wrap output as a dict
            )
 
    # hf_output.sequences: (batch, seq_len)
    # hf_output.scores:    tuple of (batch, vocab_size), length = num_generated_tokens
    generated_ids = hf_output.sequences
    scores = hf_output.scores
 
    # Decode predicted text — same logic as original model.generate
    raw_answer = ""
    for output_token in generated_ids:
        if output_token[0] == 0:
            output_token = output_token[1:]
        text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
        text = text.split('</s>')[0].replace("<s>", "").split(r'[/INST]')[-1].strip()
        raw_answer = text
        break  # batch size 1
 
    # Parse last word as emotion (original Emotion-LLaMA method)
    last_word = raw_answer.split()[-1].lower() if raw_answer.strip() else ""
    pred_emotion = last_word if last_word in [e.lower() for e in EMOTION_LABELS] else "neutral"
 
    # ── Alignment check ───────────────────────────────────────────────────────
    # compute_confidence_from_scores assumes the emotion tokens are the LAST N
    # steps in `scores`. We need to verify this — but first strip any trailing
    # EOS tokens, since the model always emits </s> as the final generated token
    # and its logit step should not count as part of the emotion sequence.
    #
    # generated_ids[0] contains the full sequence (prompt + new tokens).
    # scores has exactly len(scores) entries — one per newly generated token —
    # so the generated token IDs are the last len(scores) tokens of generated_ids[0].
    eos_token_id = model.llama_tokenizer.eos_token_id
    actual_generated_ids = generated_ids[0][-len(scores):].tolist()
    scores_list = list(scores)  # mutable copy so we can trim without affecting caller
 
    # Strip all trailing EOS tokens from both the ID list and scores
    while actual_generated_ids and actual_generated_ids[-1] == eos_token_id:
        actual_generated_ids.pop()
        scores_list.pop()
 
    pred_token_ids = token_map.get(pred_emotion, [])
    n = len(pred_token_ids)
    alignment_ok = len(actual_generated_ids) >= n and (actual_generated_ids[-n:] == pred_token_ids)
 
    if not alignment_ok:
        # Emotion word is not flush at the tail of generation after EOS stripping
        # (e.g. trailing punctuation or extra words after the emotion label).
        print(f"  [WARN] Alignment failed: expected tail {pred_token_ids}, "
              f"got {actual_generated_ids[-n:]} | raw_answer: '{raw_answer}'")
        probs_dict = {e: float('nan') for e in EMOTION_LABELS}
        confidence = float('nan')
        return pred_emotion, probs_dict, confidence, raw_answer
 
    # Compute true autoregressive confidence using EOS-stripped scores
    probs_dict, confidence = compute_confidence_from_scores(tuple(scores_list), token_map, pred_emotion)
 
    return pred_emotion, probs_dict, confidence, raw_answer
 
 
def extract_hidden_states(model, samples):
    """
    Run a teacher-forced forward pass to extract LLaMA hidden states for downstream use.
    This is kept separate from generation — hidden states capture the full
    sequence representation and are used as features.
 
    Returns:
        hidden_states: np.ndarray of shape (seq_len, 4096)
    """
    
    # samples['image'].shape -> [1, 3, 448, 448]
    # samples['video_features'].shape -> [1, 3, 1024] 
    # prepare the embedding to condition and the embedding to regress
    cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, inputs_llama, _ = \
        model.preparing_embedding(samples)
        
    # after this, cond_embeds.shape -> [1, 259, 4096]
     # image features -> [1, 256, 4096] after eva model and projector layer
     # video features -> [1, 2, 4096] after projector layer (audio is not used)
     # cls token -> [1, 1, 4096]
 
    inputs_embeds, attention_mask, input_lens = \
        model.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
 
    # get bos token embedding
    bos = torch.ones_like(part_targets[:, :1]) * model.llama_tokenizer.bos_token_id
    # add bos token at the begining
    bos_embeds = model.embed_tokens(bos)
    bos_atts = cond_atts[:, :1]
 
    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([bos_atts, attention_mask], dim=1)
 
    with model.maybe_autocast():
        with torch.no_grad():
            outputs = model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    with model.maybe_autocast():
        outputs_hidden_states = outputs.hidden_states[-1].cpu().detach().numpy()
        # [6,7,8, 9]
        #features_idx = [0,1, 2, 3, 4, 5, 6, 7, 8, 9] # took this feature list from authors code 
        # set [:, :10, :] to extract the first 10 features for exampple
        llama_features = outputs_hidden_states[:, :, :]  # (batch_size, 4, 4096)
        
        return llama_features
 
 
if __name__ == "__main__":
    parser = eval_parser()
    for action in parser._actions:
        if '--cfg-path' in action.option_strings:
            action.required = False
            action.default = "eval_configs/extract_embeddings.yaml"
            break
 
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/external_drive/Emotion-LLaMA/Features/Preprocessed_features_all/blur_0.0",
                        help="Directory to save extracted hidden states (.npy)")
    args = parser.parse_args()
 
    print("Initializing model...")
    model, vis_processor = init_model(args)
    model.eval()
 
    # Pre-compute token IDs for all emotion labels
    print("\nTokenizing emotion labels:")
    token_map, max_emotion_tokens = build_emotion_token_map(model.llama_tokenizer, EMOTION_LABELS)
    print(f"\nMax tokens across all emotions: {max_emotion_tokens}\n")
 
    # Paths
    peak_frame_dir     = "/home/ricke/emotion-llama/HCMW/factory/raw/mer"
    face_dir           = '/mnt/external_drive/Affectnet/Features/Preprocessed_features/blur_0.0'
    mae_feature_dir    = "/mnt/external_drive/MAE/Features/Preprocessed_features_16_pool/blur_0.0"
    maevideo_feature_dir = "/mnt/external_drive/MAEVideo/Features/Preprocessed_features_16_pool/blur_0.0"
    output_dir         = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
 
    if not os.path.exists(peak_frame_dir):
        print(f"Error: Peak frame directory not found: {peak_frame_dir}")
        sys.exit(1)
 
    device = next(model.parameters()).device
    vids = os.listdir(face_dir)
    results = []
    n_alignment_failures = 0
    print(f'Found {len(vids)} videos.')
 
    for i, vid in tqdm(enumerate(vids, 1), total=len(vids), desc="Processing videos"):
        video_name = vid.replace('.npy', '')
 
        # Load peak frame image
        image_path = os.path.join(peak_frame_dir, video_name, f'{video_name}_peak_frame.png')
        image = vis_processor(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
 
        # Load features
        FaceMAE_feats    = torch.tensor(np.load(os.path.join(mae_feature_dir,        video_name + '.npy')))
        VideoMAE_feats   = torch.tensor(np.load(os.path.join(maevideo_feature_dir,   video_name + '.npy')))
        # Audio_feats      = torch.zeros_like(FaceMAE_feats) -> no audio features
 
        for feats in [FaceMAE_feats, VideoMAE_feats]:
            if feats.dim() == 1:
                feats.unsqueeze_(0) # add a batch dimension
            elif feats.shape[1] > 1: # time dimension > 1 -> mean pooling
                feats = torch.mean(feats, axis=0).unsqueeze(0) # average over the time dimension
                
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats), dim=0).unsqueeze(0).to(device) # [1, 3, 1024]
 
        # Instruction
        emotion_instruction = ("Please determine which emotion label in the video represents: "
                               "happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.")
        instruction = f"<video><VideoHere></video> <feature><FeatureHere></feature> [emotion] {emotion_instruction} "
 
        # ── 1. Generate prediction + true autoregressive confidence ──────────
        pred_emotion, probs_dict, confidence, raw_answer = get_emotion_via_generate(
            model, image, video_features, [instruction], token_map, max_new_tokens=500
        )
 
        if np.isnan(confidence):
            n_alignment_failures += 1
 
        # ── 2. Extract hidden states for downstream feature use ───────────────
        samples = {
            "image":             image,
            "video_features":    video_features,
            "instruction_input": [instruction],
            "answer":            [""],
            "class_weight":      [1.0],
        }
        hidden_states = extract_hidden_states(model, samples)  # (seq_len, 4096)
        np.save(os.path.join(output_dir, video_name + '.npy'), hidden_states)
 
        results.append({
            "video_name":   video_name,
            "pred_emotion": pred_emotion,
            "confidence":   confidence if np.isnan(confidence) else round(confidence, 4),
            "raw_answer":   raw_answer,
            **{f"prob_{e}": (probs_dict[e] if np.isnan(probs_dict[e]) else round(probs_dict[e], 4))
               for e in EMOTION_LABELS},
        })
 
    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = 'emotion_llama_predictions_with_confidence.csv'
    fieldnames = (
        ["video_name", "pred_emotion", "confidence", "raw_answer"]
        + [f"prob_{e}" for e in EMOTION_LABELS]
    )
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
 
    print(f"\nDone. Results saved to: {out_csv}")
    print(f"Columns: video_name | pred_emotion | confidence | raw_answer | prob_<emotion> x{len(EMOTION_LABELS)}")
    print(f"Alignment failures (confidence=nan): {n_alignment_failures}/{len(vids)} "
          f"({100*n_alignment_failures/len(vids):.1f}%)")