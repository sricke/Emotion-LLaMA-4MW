# Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning  

This is a fork of the [Emotion-LLaMA GitHub repository](https://github.com/ZebangCheng/Emotion-LLaMA), used for experimenting with Emotion-LLaMA as an encoder for mind wandering detection.

The most useful code is located in the `minigpt4` folder, specifically in the `datasets`, `models`, `runners`, and `task` directories.

This iteration supports training Emotion-LLaMA end-to-end (in addition to using it as a feature extractor), though experiments with end-to-end training were unsuccessful.
For only feature extraction follow the following guideline:

## 🔍 Extracting Emotion-LLaMA Features from Frames


The codebase provides functionality to extract emotion-llama features from image frames. The main extraction pipeline consists of the following components:

### Main Function: `extract_model_features`

**Prerequisites**: Before extracting Emotion-LLaMA features, you must first extract FaceMAE and VideoMAE features:
- `extract_mae_embedding.py`: Extracts FaceMAE features from frames (e.g., `python extract_mae_embedding.py  --device='cuda:0'`)
- `extract_maeVideo_embedding.py`: Extracts VideoMAE features from videos (e.g., `python extract_maeVideo_embedding.py  --feature_level='UTTERANCE' --device='cuda:0'`)

**File:** `eval_emotion.py` (lines 53-108)

This is the primary function for extracting emotion llama features from frames. It performs the following steps:

1. **Image and Video Feature Encoding**: Calls `model.preparing_embedding(samples)` which internally processes:
   - Images through the visual encoder (EVA model) → `[256, 4096]` image embeddings
   - Video features (FaceMAE and VideoMAE) through projector layers → `[2, 4096]` video embeddings
   - Concatenates: `[256 image patches + 2 video features + 1 CLS token] = [259, 4096]`

  

2. **LLaMA Processing**: Passes the concatenated embeddings through the LLaMA model to obtain hidden states

3. **Feature Extraction**: Extracts specific hidden states (indices 6, 7, 8, 9) from the last layer as the final llama features. These features were selected because
they provided the best results while training downstream task of mind wandering. 

**Output**: Returns four types of features:
- `facemae_projected_features`: FaceMAE projected features `[batch_size, 4096]`
- `videomae_projected_features`: VideoMAE projected features `[batch_size, 4096]`
- `image_projected_features`: Image features `[batch_size, 256, 4096]`
- `llama_features`: LLaMA hidden states `[batch_size, 4, 4096]`

### Key Model Methods

#### `model.encode_img(image, video_features)`
**File:** `minigpt4/models/minigpt_v2.py` (lines 92-118)

Encodes images and video features:
- Uses the visual encoder (EVA model) to process images
- Projects image features through `llama_proj` layer
- Projects video features (FaceMAE and VideoMAE) through separate projector layers
- Returns concatenated embeddings ready for LLaMA processing

#### `model.preparing_embedding(samples)`
**File:** `minigpt4/models/minigpt_base.py` (lines 289-340)

Orchestrates the embedding preparation:
- Calls `encode_img()` to get visual embeddings
- Wraps embeddings with text instructions
- Prepares tokens for the LLaMA model forward pass

### Usage Example

The feature extraction is used in the `hcmw_caption` evaluation section:

```python
# Create samples dict with images and video features
samples = {
    'image': images,  # [batch_size, 3, 448, 448]
    'video_features': video_features,  # [batch_size, 3, 1024]
    'instruction_input': instruction_input,
    'answer': [''] * len(instruction_input)
}

# Extract features
facemae_projected_features, videomae_projected_features, \
    image_projected_features, llama_features = extract_model_features(model, samples)
```

The extracted features are saved as numpy arrays in the `features/` directory when running evaluation with the `hcmw_caption` dataset.


## 📜 License
This repository is under the [BSD 3-Clause License](./LICENSE.md). The code is based on MiniGPT-4, which uses the BSD 3-Clause License (see [LICENSE_MiniGPT4.md](./LICENSE_MiniGPT4.md)). Data from MER2023 is licensed under [EULA](./LICENSE_EULA.md) for research purposes only.
