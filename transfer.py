import os
import json
import glob
import argparse

from collections import OrderedDict
from safetensors.torch import save_file, load_file
from huggingface_hub import split_torch_state_dict_into_shards

def load_model(path):

    state = OrderedDict()

    for file in glob.glob(os.path.join(path, 'model*.safetensors')):
        state.update(load_file(file, device='cpu'))
    
    return state

def save_model(state, path):

    state_dict_split = split_torch_state_dict_into_shards(state)
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state[tensor] for tensor in tensors}
        save_file(
            shard,
            os.path.join(path, filename),
            metadata={"format": "pt"},
        )
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))

def convert(state):
    new_state = OrderedDict()

    new_state["model.embed_tokens.weight"] = state["model.transformer.wte.weight"]
    new_state["lm_head.weight"] = state["model.transformer.ff_out.weight"]
    new_state["model.norm.weight"] = state["model.transformer.ln_f.weight"]

    for layer in range(32):
        new_state[f"model.layers.{layer}.self_attn.q_proj.weight"] = state[f"model.transformer.blocks.{layer}.q_proj.weight"]
        new_state[f"model.layers.{layer}.self_attn.k_proj.weight"] = state[f"model.transformer.blocks.{layer}.k_proj.weight"]
        new_state[f"model.layers.{layer}.self_attn.v_proj.weight"] = state[f"model.transformer.blocks.{layer}.v_proj.weight"]
        new_state[f"model.layers.{layer}.self_attn.o_proj.weight"] = state[f"model.transformer.blocks.{layer}.attn_out.weight"]

        new_state[f"model.layers.{layer}.input_layernorm.weight"] = state[f"model.transformer.blocks.{layer}.attn_norm.weight"]
        new_state[f"model.layers.{layer}.post_attention_layernorm.weight"] = state[f"model.transformer.blocks.{layer}.ff_norm.weight"]

        new_state[f"model.layers.{layer}.mlp.up_proj.weight"] = state[f"model.transformer.blocks.{layer}.up_proj.weight"]
        new_state[f"model.layers.{layer}.mlp.down_proj.weight"] = state[f"model.transformer.blocks.{layer}.ff_out.weight"]
        new_state[f"model.layers.{layer}.mlp.gate_proj.weight"] = state[f"model.transformer.blocks.{layer}.ff_proj.weight"]
    
    return new_state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', default='data/LLaDA-8B-Instruct', type=str)

    args = parser.parse_args()

    state = load_model(args.input_path)
    save_model(convert(state), args.output_path)