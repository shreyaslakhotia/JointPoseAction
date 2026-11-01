import os
import sys
import argparse
import torch
from PIL import Image

# 1. Fix the import path so Python can find notebook/inference.py
sys.path.append(os.path.join(os.path.dirname(__file__), "notebook"))
from inference import Inference

# 2. Surgical YAML Patch: Delete Gaussian decoders and FORCE device=cpu
from omegaconf import OmegaConf
original_load = OmegaConf.load

def patched_load(path):
    config = original_load(path)
    
    # Save VRAM by deleting unused Gaussian decoders
    for key in ["slat_decoder_gs", "slat_decoder_gs_4"]:
        if key in config:
            del config[key]
            
    # CRITICAL FIX: If the config specifies a device, force it to 'cpu'.
    # This prevents inference_pipeline.py line 297 from calling model.to('cuda')
    if "device" in config:
        config.device = "cpu"
        
    return config

OmegaConf.load = patched_load

def assign_and_wrap(module, device):
    """Moves a sub-model to a specific GPU and wraps its forward pass to automatically handle cross-GPU tensors."""
    print(f"📍 Assigning {module.__class__.__name__} to {device}...")
    module.to(device)
    original_forward = module.forward
    
    def new_forward(*args, **kwargs):
        def to_dev(x):
            if isinstance(x, torch.Tensor): return x.to(device)
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            if isinstance(x, list): return [to_dev(v) for v in x]
            return x
            
        d_args = [to_dev(a) for a in args]
        d_kwargs = {k: to_dev(v) for k, v in kwargs.items()}
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            return original_forward(*d_args, **d_kwargs)
            
    module.forward = new_forward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/4] Initializing Pipeline safely in System RAM (CPU)...")
    # Because of our patched_load, this will instantiate everything on CPU
    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
    pipeline = inference._pipeline

    print("\n[2/4] Distributing Pipeline across 4x RTX 2080 Ti GPUs...")
    assign_and_wrap(pipeline.ss_condition_embedder, "cuda:0")
    assign_and_wrap(pipeline.models["ss_generator"], "cuda:1")
    assign_and_wrap(pipeline.models["slat_generator"], "cuda:2")
    assign_and_wrap(pipeline.models["slat_decoder_mesh"], "cuda:3")

    print("\n[3/4] Loading Image and Mask...")
    image = Image.open(args.image_path).convert("RGB")
    mask = Image.open(args.mask_path).convert("L")
    
    print("\n[4/4] Executing Multi-GPU Inference (This may take a minute)...")
    output = inference(image, mask, seed=42)

    print("\n🎉 Inference Complete! Exporting results...")
    if "mesh" in output:
        mesh_path = os.path.join(args.output_dir, "reconstruction.obj")
        try:
            output["mesh"].save_obj(mesh_path)
            print(f"✅ Mesh target saved to {mesh_path}")
        except AttributeError:
            print(f"⚠️ Notice: Mesh object type {type(output['mesh'])} does not support .save_obj().")
    else:
        print("❌ No mesh found in output dict! Available keys:", output.keys())

if __name__ == "__main__":
    main()
