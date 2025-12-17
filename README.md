<div align="center">

# VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation

[![arXiv](https://img.shields.io/badge/arXiv-2112.09133-B31B1B.svg)](https://arxiv.org/abs/2112.09133)[![Hugging Face](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/xxx)[![GitHub](https://img.shields.io/badge/GitHub-Project-black?logo=github)](https://github.com/xxx)

[Xinyao Liao*](https://lil-shake.github.io/), [Qiyuan He*†](https://qy-h00.github.io/), [Kai Xu](https://kai422.github.io/), [Xiaoye Qu](https://scholar.google.com/citations?user=rT3hqdcAAAAJ&hl=zh-CN), [Yicong Li](https://yl3800.github.io/), [Wei Wei](https://www.eric-weiwei.com/), [Angela Yao](https://www.comp.nus.edu.sg/~ayao/)

$^*$ Equal contribution. $^†$ Project lead.
</div>

<p align="center">
  <img src="assets/vis-c2i-t2i_00.png" width="720">
</p>

## 📌 Release
- [03/2025] Code is publicly available.

## 📑 Abstract
We propose **VA-π**, a lightweight post-training framework that directly optimizes **visual AR models** with a principled **pixel-space objective**, solved by introducing **evidence lower bound (ELBO)** that unifies pixel reconstruction and autoregressive modeling. To optimize under the discrete token space, VA-π introduces a **reinforcement-based alignment strategy** that treats the AR generator as a policy, uses pixel-space reconstruction quality as its intrinsic reward and Next Token Prediction (NTP) loss with noisy context as regularizer. VA-π enables rapid adaptation of existing AR generators, without neither tokenizer retraining nor external reward models. With only 1\% ImageNet-1K data and 25 minutes of tuning, it reduces FID from 14.36 to 7.65 and improves IS from 86.55 to 116.70 on LlamaGen-XXL, while also yielding notable gains in the text-to-image task on GenEval for both visual generation model (LlamaGen: from 0.306 to 0.339) and unified multi-modal model (Janus-Pro: from 0.725 to 0.744). 

## 🔧 Installation

First, clone this repository.
```
git clone https://github.com/Lil-Shake/VA-Pi
cd VA-Pi
```
Create environments and intall packages for LlamaGen and Janus-Pro.
```
# For LlamaGen
conda create -n vapi_llamagen python=3.10
conda activate vapi_llamagen
pip install -r llamaGen/requirements.txt

# For Janus-Pro
conda create -n vapi_janus python=3.10
conda activate vapi_janus
pip install -r llamaGen/requirements.txt
```

Download [ImageNet-1k](https://www.image-net.org/download.php?utm_source=chatgpt.com) for C2I training on LlamaGen, [LAION-COCO](https://laion.ai/blog/laion-coco/) for T2I training on LlamaGen. Load dataset [Flux-Reason](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) dataset from HuggingFace for T2I training on Janus-Pro. Download pretrained [LlamaGen](https://huggingface.co/FoundationVision/LlamaGen/tree/main) and [Janus-Pro 1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B).

## 🚀 Training

Post-train C2I task on LlamaGen.

```
cd /path/to/VA-Pi/LlamaGen

bash scripts/autoregressive/train_c2i_grpo_8gpu.sh \
  /path/to/vq_ds16_c2i.pt \
  /path/to/c2i_XXL_384.pt \
  /path/to/imagenet-1k
```

Post-train T2I task on LlamaGen (LAION-COCO tar shards). Run T5 feature extraction first:

```
cd /path/to/VA-Pi/LlamaGen

# 1) Extract T5 features 
bash scripts/language/extract_flan_t5_feat_laion_coco_stage1.sh \
  /path/to/laion-coco-50m \
  /path/to/models \
  /path/to/models/flan-t5-xl/t5_features

# 2) Train T2I GRPO
bash scripts/autoregressive/train_t2i_grpo_8gpu.sh \
  /path/to/vq_ds16_t2i.pt \
  /path/to/t2i_XL_stage1_256.pt \
  /path/to/laion-coco-50m \
  /path/to/models/flan-t5-xl/t5_features
```
Post-train T2I task on Janus-Pro.

```
cd /path/to/VA-Pi/Janus

# Single node (8 GPUs), HF streaming download + save-to-disk shards
bash train/run_t2i_grpo_janus_deepspeed_16g_hf.sh \
  /path/to/Janus-Pro-1B \
  LucasFang/FLUX-Reason-6M \
  /path/to/hf_stream/FLUX-Reason-6M-random
```

## 💫 Evaluation

### C2I Evaluation

First sample 50k images using the post-trained model.

```
cd /path/to/VA-Pi/LlamaGen

bash scripts/autoregressive/sample_c2i.sh \
  /path/to/vq_ds16_c2i.pt \
  /path/to/c2i_XXL_384.pt \
  /path/to/output_samples
```
The evaluation environment setup can refer to [LlamaGen](https://github.com/FoundationVision/LlamaGen/blob/main/evaluations/c2i/README.md) code base, and then download reference samples from [ImageNet_256x256_reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz). Compute FID between reference batch and post-trained models' samples.

```
bash fid-eval.sh /path/to/reference_batches /path/to/output_samples
``` 

### T2I Evaluation

First generate images on GenEval prompts using the post-trained models.

**For LlamaGen-XL:**
```
cd /path/to/VA-Pi/LlamaGen

bash scripts/autoregressive/sample_t2i_geneval.sh \
  /path/to/vq_ds16_t2i.pt \
  /path/to/t2i_XL_or_your_grpo_ckpt
  /path/to/t5_cache_dir \
  /path/to/geneval_prompts.jsonl \
  /path/to/output_geneval_samples
```
**For Janus-Pro 1B:**
```
cd /path/to/VA-Pi/Janus

bash run_geneval_infer.sh \
  --prompts-dir /path/to/evaluation_metadata_geneval.jsonl \
  --base-model-path /path/to/Janus-Pro-1B_or_hf_repo \
  --model-path /path/to/janus_t2i_grpo_ckpt_or_hf_repo \
  --save-root /path/to/output_geneval_samples
```

Then run GenEval benchmark evaluation following the official GenEval repo: [djghosh13/geneval](https://github.com/djghosh13/geneval).
Our generated folder `output_geneval_samples/` matches GenEval's expected layout (`00000/metadata.jsonl` + `00000/samples/*.png`, etc.), so you can directly evaluate it after setting up GenEval and downloading the object detector.


## 🤗 Acknowledgments

This project builds upon and is inspired by several excellent open-source codebase [LlamaGen](https://github.com/FoundationVision/LlamaGen), [Janus](https://github.com/deepseek-ai/Janus), and [geneval](https://github.com/djghosh13/geneval).

## ⭐ Citation
