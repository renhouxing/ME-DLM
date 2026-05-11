## Edit-Based Refinement for Parallel Masked Diffusion Language Models

<p align="center">
    <a href="https://arxiv.org/abs/2605.09603">📄 Paper</a> •
    <a href="https://github.com/renhouxing/ME-DLM">🏠 Repo</a> •
    <a href="https://huggingface.co/renhouxing/ME-DLM-Stage3">🤖 Models</a>
</p>

## Introduction

ME-DLM is a lightweight edit-based refinement framework for masked diffusion language models. It first generates a complete response through parallel diffusion decoding, then refines the output with minimal edit operations such as replacement, deletion, and insertion, conditioned on the full sequence. By using edit distance as deterministic training supervision, ME-DLM improves sequence-level consistency while preserving the decoding efficiency of diffusion models. Built on LLaDA, it achieves consistent gains on HumanEval and GSM8K while using only one-eighth of the total diffusion steps.

## Models

| Model | Checkpoint |
|:-------|:------------|
| ME-DLM Stage 1   | 🤗 [HF Link](https://huggingface.co/renhouxing/ME-DLM-Stage1) |
| ME-DLM Stage 2   | 🤗 [HF Link](https://huggingface.co/renhouxing/ME-DLM-Stage2) |
| ME-DLM Stage 3   | 🤗 [HF Link](https://huggingface.co/renhouxing/ME-DLM-Stage3) |

## Evaluation

```bash
# Evaluate the Stage 3 model on HumanEval, MBPP, GSM8K, and Math500 benchmarks
# Note: you can adjust the diffusion steps for faster inference or better performance
python test.py --path renhouxing/ME-DLM-Stage3 --task humaneval mbpp gsm8k math500 --mask_diffusion_step 48 --edit_diffusion_step 16
```

## Reproducibility Statement

### Model Weight Transfer

The original implementation of **LLaDA** does not support gradient checkpointing.
Therefore, we rewrote the LLaDA codebase and provide a weight conversion script to ensure compatibility.

To transfer the pretrained model weights, run:

```bash
python transfer.py -i <your_llada_path>
```

After execution, the converted model will be saved to:

```
data/LLaDA-8B-Base
```

**Important Note**

During conversion, we also modified several special tokens.
Please make sure to use the converted model under `data/LLaDA-8B-Base` directly, rather than the original checkpoint.

---

### Training Pipeline

The training procedure consists of three sequential stages.
Each stage requires a different data format and training configuration.

---

#### Install Edit Distance Helper

This project requires an additional C++ edit distance acceleration module for efficient training, especially in Stage 3 diffusion refinement.

```bash
cd utils/edit_cpp_helper
pip install .
```

---

### Stage 1: Pretraining

#### Data Format

Stage 1 uses plain text supervision.
The training file should be in **JSONL** format, where each line contains a JSON object with a mandatory `text` field.

Example:

```json
{
  "text": "test text"
}
```

---

#### Training Command

```bash
torchrun \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} \
    --nproc_per_node 8 \
    train.py \
    --seed 3407 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --save_steps 5000 \
    --max_len 2048 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --do_train \
    --save_safetensors \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --deepspeed config/stage_1.json \
    --model_cfg data/LLaDA-8B-Base \
    --train_file <train_file> \
    --output_dir <output_dir> \
    --min_t 0.0 \
    --max_t 1.0 \
    --copy_head
```

---

### Stage 2: Supervised Fine-Tuning (Instruction Format)

Stage 2 introduces dialogue-style instruction tuning.

#### Data Format

The training data should be in **JSONL** format.
Each example must contain:

* `history`: conversation context
* `response`: target assistant reply

Example:

```json
{
  "history": [
    {
      "role": "user",
      "content": "def reverse_string(s: str) -> str: ..."
    }
  ],
  "response": "To solve this problem, we need to reverse the characters..."
}
```

---

#### Training Command

```bash
torchrun \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} \
    --nproc_per_node 8 \
    train.py \
    --seed 3407 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --save_steps 1000 \
    --max_len 2048 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --do_train \
    --save_safetensors \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --deepspeed config/stage_1.json \
    --model_cfg <model_path> \
    --train_file <train_file> \
    --output_dir <output_dir> \
    --min_t 0.0 \
    --max_t 1.0 \
    --pad_len 128
```

---

### Stage 3: Mask-and-Edit Diffusion Training (Full Model Refinement)

Stage 3 further extends Stage 2 by enabling intermediate refinement and multi-step diffusion prediction.

#### Data Format

Stage 3 uses the same format as Stage 2:

* `history`
* `response`

---

#### Training Command

```bash
torchrun \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nnodes ${WORLD_SIZE} \
    --nproc_per_node 8 \
    train.py \
    --seed 3407 \
    --report_to tensorboard \
    --dataloader_num_workers 8 \
    --remove_unused_columns False \
    --save_steps 1000 \
    --max_len 2048 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --do_train \
    --save_safetensors \
    --gradient_checkpointing \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --deepspeed config/stage_1.json \
    --model_cfg <model_path> \
    --train_file <train_file> \
    --output_dir <output_dir> \
    --min_t 0.0 \
    --max_t 1.0 \
    --pad_len 128 \
    --intermediate_ratio 0.5 \
    --mask_diffusion_pred_per_step 2 4 8 16 \
    --max_diffusion_steps 32 \
    --intermediate_min_t 0.0 \
    --intermediate_max_t 1.0
```

---

### Summary

| Stage   | Objective                      | Data Format                         |
| ------- | ------------------------------ | ----------------------------------- |
| Stage 1 | Text-only pretraining          | `{"text": ...}`                     |
| Stage 2 | Instruction fine-tuning        | `{"history": ..., "response": ...}` |
| Stage 3 | Mask-edit diffusion refinement | Same as Stage 2                     |

## Acknowledgments

We thank the following amazing projects that truly inspired us:

- [LLaDA](https://github.com/ML-GSAI/LLaDA)