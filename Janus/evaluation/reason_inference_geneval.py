import os
from PIL import Image
from tqdm import tqdm
import re
import numpy as np
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM
from torchvision.utils import make_grid
from einops import rearrange
from torchvision.transforms import ToTensor

from janus.models import MultiModalityCausalLM, VLChatProcessor
from typing import List, Dict
import time
import argparse
from collections import OrderedDict
from transformers import AutoTokenizer
from transformers import GenerationConfig, TextStreamer
from typing import List
import hpsv2
import random
import copy
import pdb
import json
import torch
from torchvision import transforms
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

# Optional: support loading DeepSpeed ZeRO-3 checkpoints directly
try:
    import deepspeed
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
except Exception:
    deepspeed = None
    load_state_dict_from_zero_checkpoint = None
try:
    # PyTorch >=2.6: allowlist DS classes for torch.load when reading optimizer states
    from torch.serialization import add_safe_globals as _add_safe_globals
except Exception:
    _add_safe_globals = None

def _register_ds_safe_globals():
    if _add_safe_globals is None:
        return
    try:
        from deepspeed.runtime.zero.config import ZeroStageEnum
        _add_safe_globals([ZeroStageEnum])
    except Exception:
        pass
    try:
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        _add_safe_globals([LossScaler])
    except Exception:
        pass

def save_image_grid(images, save_path, grid_size=(2, 2), img_size=(1024, 1024)):
    grid_img = Image.new('RGB', (grid_size[1] * img_size[1], grid_size[0] * img_size[0]))
    for idx, img in enumerate(images):
        x = idx % grid_size[1] * img_size[1]
        y = idx // grid_size[1] * img_size[0]
        grid_img.paste(img, (x, y))
    grid_img.save(save_path)

def process_text_tokens(vl_chat_processor,cot_prompt,prompt_list):
    prompt_list_return=[];prompt_text_list_return=[]
    for prompt in prompt_list:
        prompt_text = copy.deepcopy(prompt)
        conversation = [
            {
                "role": "<|User|>",
                "content": cot_prompt.format(prompt),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'
        system_prompt = ""
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=system_prompt,
        )
        prompt_ = sft_format 
        prompt_list_return.append(prompt_)
        prompt_text_list_return.append(prompt_text)
    return prompt_list_return,prompt_text_list_return

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: List[str],
    prompt_text: List[str],
    semantic_cot: bool,
    temperature: float = 1,
    num_generation: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    conversation: List[Dict[str, str]] = None,
):  

    print('num_generations:',num_generation)
    prompt_inputs = vl_chat_processor.tokenizer(
            text=prompt,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True
    )
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    prompt_ids = prompt_ids.to('cuda')
    prompt_mask = prompt_mask.to('cuda')
    input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)

    if semantic_cot:
        # TODO: if num_generations is too large, we need to split it into multiple batches
        if num_generation > 20:
            total_generations = []
            for i in range(prompt_ids.shape[0] // num_generation):
                current_input_embeds = input_embeds[i*num_generation: (i+1)*num_generation]
                current_attn_mask = prompt_mask[i*num_generation: (i+1)*num_generation]
                prompt_completion_ids = mmgpt.language_model.generate(
                    inputs_embeds=current_input_embeds,
                    attention_mask=current_attn_mask,
                    pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                    bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                    eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
                total_generations.append(prompt_completion_ids)
            prompt_completion_ids = torch.cat(total_generations, dim=0)
        else: # if num_generations == 1, we directly generate all for the batch data
            prompt_completion_ids = mmgpt.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_ids
        completion_ids = prompt_completion_ids

        image_gen_prompt_list = []
        
        prompt = vl_chat_processor.tokenizer.decode(prompt_ids[0].cpu().tolist(), skip_special_tokens=True)
        for i in range(completion_ids.shape[0]):
            answer = vl_chat_processor.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            image_gen_prompt = f"{prompt_text[i]}. {answer}"

            conversation = [
                {
                    "role": "<|User|>",
                    "content": image_gen_prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )

            print(f"Prompt {i}: {sft_format}\nSemantic-CoT {i}: {answer}")
            image_gen_prompt_list.append(sft_format)

        prompt_inputs = vl_chat_processor.tokenizer(
            text=image_gen_prompt_list,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True,
        ) # {'input_ids', 'attention_mask'}

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    else:
        processed_prompt_text=[]
        for prompt_item in prompt_text:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt_item,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            processed_prompt_text.append(sft_format)
        print(processed_prompt_text)
            
        prompt_inputs = vl_chat_processor.tokenizer(
                text=processed_prompt_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=True
        )

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    
    prompt_ids = prompt_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    image_start_token_id = vl_chat_processor.tokenizer.encode(vl_chat_processor.image_start_tag)[1]
    prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), image_start_token_id)], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids)
    pad_input_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), vl_chat_processor.pad_id))
    total_generated_tokens_img = []

    for j in range(inputs_embeds.shape[0] // num_generation):
        cond_inputs_embeds = inputs_embeds[j*num_generation: (j+1)*num_generation]
        cond_attention_mask = attention_mask[j*num_generation: (j+1)*num_generation]
        uncond_inputs_embeds = cond_inputs_embeds.clone()
        uncond_inputs_embeds[:, 1:-1] = pad_input_embeds
        
        inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
        inputs_embeds_img[1::2] = uncond_inputs_embeds
        attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
        attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

        split_size = 2 * num_generation
        for jj in range(0, inputs_embeds_img.shape[0], split_size):
            print(f"Generating image {jj}")
            start = jj
            end = min(jj + split_size, inputs_embeds_img.shape[0])
            generated_tokens = torch.zeros(((end-start)//2, image_token_num_per_image), dtype=torch.int64).cuda()
            cur_inputs_embeds_img = inputs_embeds_img[start: end]
            cur_attention_mask_img = attention_mask_img[start: end]

            for k in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(
                    inputs_embeds=cur_inputs_embeds_img, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if k != 0 else None, 
                    attention_mask=cur_attention_mask_img
                )
                
                hidden_states = outputs.last_hidden_state
                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, k] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.long)], dim=1)


            print(generated_tokens.shape)
            total_generated_tokens_img.append(generated_tokens)

    total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

    torch.cuda.empty_cache()
    num_imgs = int(total_generated_tokens_img.shape[0])
    dec = mmgpt.gen_vision_model.decode_code(total_generated_tokens_img.to(dtype=torch.int), shape=[num_imgs, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((num_imgs, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    
    return visual_img

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def transform_image(image, size=512):
    transform = transforms.Compose([
        transforms.Resize(size, max_size=None),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ])
    return transform(image)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """销毁分布式进程"""
    dist.destroy_process_group()
    
def deduplicate_annotations(meta_json):
    unique_annotations = {}
    for annotation in meta_json['annotations']:
        unique_annotations[annotation['image_id']] = annotation
    meta_json['annotations'] = list(unique_annotations.values())

def get_data_for_rank(prompts_dir, rank, world_size, batch_size, imsize,save_root):
    # Load metadata supporting both JSONL and JSON files
    metadatas = None
    with open(prompts_dir, 'r') as fp:
        content = fp.read()
        stripped = content.strip()
        try:
            # Try full-file JSON first if it looks like JSON
            if stripped.startswith('{') or stripped.startswith('['):
                data = json.loads(stripped)
                if isinstance(data, list):
                    metadatas = data
                elif isinstance(data, dict):
                    # Try to find a list of dicts (with 'prompt' or 'caption')
                    candidate_lists = [v for v in data.values() if isinstance(v, list)]
                    chosen = None
                    for cand in candidate_lists:
                        if len(cand) > 0 and isinstance(cand[0], dict):
                            if ('prompt' in cand[0]) or ('caption' in cand[0]):
                                chosen = cand
                                break
                    if chosen is not None:
                        metadatas = chosen
                    else:
                        # Fallback: wrap dict as single sample
                        metadatas = [data]
            else:
                # Not JSON text; fall back to JSONL
                raise json.JSONDecodeError("Invalid JSON", stripped, 0)
        except json.JSONDecodeError:
            # Fallback: treat as JSONL (one JSON object per line)
            fp.seek(0)
            metadatas = [json.loads(line) for line in fp if line.strip()]
    indices=[i for i in range(len(metadatas))]

    total_samples=len(metadatas)
    print('total samples to generate: ',total_samples)

    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank != world_size - 1 else total_samples

    captions, names, meta_data = [], [], []
    for idx, metadata in zip(indices[start_idx:end_idx],metadatas[start_idx:end_idx]):
        # Normalize: support either 'prompt' or 'caption'
        if 'prompt' in metadata:
            captions.append(metadata['prompt'])
        elif 'caption' in metadata:
            captions.append(metadata['caption'])
            metadata = dict(metadata)
            metadata['prompt'] = metadata['caption']
        else:
            raise KeyError("Each metadata item must contain 'prompt' or 'caption'")
        names.append(idx)
        meta_data.append(metadata)
        

    return [{'caption': captions[i:i+batch_size], 'name': names[i:i+batch_size], \
        'meta_data': meta_data[i:i+batch_size]} 
            for i in range(0, len(captions), batch_size)]

    
    
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def pick_dtype_for_device(device: torch.device) -> torch.dtype:
    if os.environ.get("FORCE_NO_BF16", "0") == "1":
        return torch.float32

    name = torch.cuda.get_device_name(device).lower()
    is_h20 = ("h20" in name) or bool(re.search(r"\bh-?20\b", name))
    bf16_supported = torch.cuda.is_bf16_supported()

    return torch.float32 if (is_h20 or not bf16_supported) else torch.bfloat16


def _load_processor_and_model(model_path: str, base_model_path: str | None, amp_dtype: torch.dtype):
    # Try to load as a complete HF repo first (processor + model)
    try:
        proc: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        mdl: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=amp_dtype,
        )
        return proc, mdl
    except Exception:
        pass

    # Fallback to loading base model + restore weights from checkpoint folder
    if base_model_path is None:
        raise ValueError(
            f"'{model_path}' is not a HF repo (missing processor/config). Provide --base_model_path "
            f"(e.g., deepseek-ai/Janus-Pro-1B) so we can load configs and then restore weights."
        )

    proc: VLChatProcessor = VLChatProcessor.from_pretrained(base_model_path)
    mdl: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=amp_dtype,
    )

    # 1) Try loading consolidated weights
    candidates = [
        os.path.join(model_path, 'consolidated.pth'),
        os.path.join(model_path, 'pytorch_model.bin'),
        os.path.join(model_path, 'model.bin'),
    ]
    loaded = False
    for f in candidates:
        if os.path.isfile(f):
            state = torch.load(f, map_location='cpu')
            try:
                mdl.load_state_dict(state, strict=False)
            except Exception:
                inner = state.get('model', state.get('module', state))
                mdl.load_state_dict(inner, strict=False)
            loaded = True
            break

    # 2) Try loading ZeRO shards directly
    if not loaded:
        # Handle ZeRO-3 shards whether model_path points to the parent dir or the tag dir
        try:
            entries = list(os.scandir(model_path))
        except Exception:
            entries = []

        names = {e.name for e in entries}
        found_zero_here = any(n.startswith('zero_pp_rank_') and n.endswith('_model_states.pt') for n in names)

        if found_zero_here:
            # model_path is the tag directory; call loader with parent dir and explicit tag
            if load_state_dict_from_zero_checkpoint is None:
                raise RuntimeError(
                    "Found DeepSpeed ZeRO shards but deepspeed is not installed. Install deepspeed or export a consolidated checkpoint."
                )
            # Register DS classes to avoid weights_only UnpicklingError
            _register_ds_safe_globals()
            parent_dir = os.path.dirname(model_path.rstrip('/'))
            tag = os.path.basename(model_path.rstrip('/'))
            load_state_dict_from_zero_checkpoint(mdl, parent_dir, tag=tag)
            loaded = True
        else:
            # Search immediate subdirectories for a ZeRO tag folder
            tag_dirs = [e for e in entries if e.is_dir()]
            for td in tag_dirs:
                try:
                    sub_names = set(os.listdir(td.path))
                except Exception:
                    continue
                if any(n.startswith('zero_pp_rank_') and n.endswith('_model_states.pt') for n in sub_names):
                    if load_state_dict_from_zero_checkpoint is None:
                        raise RuntimeError(
                            "Found DeepSpeed ZeRO shards but deepspeed is not installed. Install deepspeed or export a consolidated checkpoint."
                        )
                    _register_ds_safe_globals()
                    load_state_dict_from_zero_checkpoint(mdl, model_path, tag=td.name)
                    loaded = True
                    break

    if not loaded:
        raise FileNotFoundError(
            f"No loadable weights found in '{model_path}'. Expected consolidated.pth or ZeRO shard files."
        )

    return proc, mdl


def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    seed=20
    batch_size=1
    repeat_size=4
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    amp_dtype = pick_dtype_for_device(device)
    print('using dtype: ',amp_dtype)
    
    print('setup inference...')
    setup(rank, world_size)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    print('begin inference..')
    
    model_path = getattr(args, 'model_path', None)
    base_model_path = getattr(args, 'base_model_path', None) or "deepseek-ai/Janus-Pro-1B"
    # If no MODEL_PATH provided (or placeholder), load base model directly
    if (model_path is None) or (str(model_path).strip() in ("", "/your-janus-ckpt")):
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(base_model_path)
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=amp_dtype,
        )
    else:
        vl_chat_processor, vl_gpt = _load_processor_and_model(model_path, base_model_path, amp_dtype)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = vl_gpt.cuda().eval()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    print("VQ model loaded")

    prompts_dir = args.prompts_dir
    batched_data=get_data_for_rank(prompts_dir,rank,world_size,batch_size,imsize=384,save_root=args.save_root)

    # 处理数据集
    for item in batched_data:
        t2=time.time()
        # print(item)
        text_prompts = item['caption']*repeat_size
        name = item['name'][0]
        metadata=item['meta_data'][0]
        print('rank %d: '%rank,name)
        output_path=os.path.join(args.save_root,f"{name:05}")

        required_files = [f"{i:05d}.png" for i in range(4)]  # 00000.png-00003.png
        missing_files = any(not os.path.exists(os.path.join(output_path, "samples", f)) for f in required_files)

        if (not os.path.exists(output_path)) \
            or (not os.path.exists(os.path.join(output_path, 'samples'))) \
            or missing_files:
            os.makedirs(output_path,exist_ok=True)
            with open(os.path.join(output_path, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)
            prompt,prompt_text=process_text_tokens(vl_chat_processor,cot_prompt,text_prompts)

            samples =  generate(
                                vl_gpt,
                                vl_chat_processor,
                                prompt,
                                prompt_text,
                                semantic_cot=args.semantic_cot,
                                num_generation=len(text_prompts),
                                temperature=args.temperature
                            )
            all_samples=[]
            os.makedirs(os.path.join(output_path, 'samples'),exist_ok=True)
            for idx in range(samples.shape[0]):
                img_pred = samples[idx]
                img_pred = Image.fromarray(img_pred)
                img_pred.save(os.path.join(output_path, 'samples', f"{idx:05}.png"))
                
                print(f"Saved: {output_path}")
                print(item['caption'])
                all_samples.append(ToTensor()(img_pred))
                
            grid = make_grid(all_samples, nrow=4)
            grid = (255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()).astype(np.uint8)
            grid_img = Image.fromarray(grid)
            grid_img.save(os.path.join(output_path, 'grid.png'))
        else:
            print(f"{output_path} exists")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_dir", type=str, default="/your-code-root/STAGE/data/evaluation_metadata_geneval.jsonl")
    parser.add_argument("--model_path", type=str, default="/your-janus-ckpt")
    parser.add_argument("--base_model_path", type=str, default=None, help="Base HF model repo/path (e.g., deepseek-ai/Janus-Pro-1B) when --model_path is a DS checkpoint directory.")
    parser.add_argument("--reasoning_prompt_path", type=str, default="/your-code-root/STAGE/data/prompt/reasoning_prompt.txt")
    parser.add_argument("--semantic_cot", type=bool, default=False)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_root", type=str, default='/your-save-root')
    args = parser.parse_args()
    main(args)