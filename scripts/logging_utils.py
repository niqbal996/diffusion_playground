import torch
import os
import accelerate
import datasets
import numpy as np
import torch
import cv2
import json
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import logging
import wandb 

logger = get_logger(__name__, log_level="INFO")



def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images

def save_model_card(
    args,
    repo_id: str,
    images: list = None,
    repo_folder: str = None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    model_description = f"""
    # Text-to-image finetuning - {repo_id}

    This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
    {img_str}

    """
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
    More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
    """

    model_description += wandb_info

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=args.pretrained_model_name_or_path,
        model_description=model_description,
        inference=True,
    )

    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

def log_images_and_losses(
        depth_prediction, 
        reconstructed_rgb, 
        depth_prediction_loss, 
        reconstructed_rgb_loss, 
        total_loss,
        input_images, 
        vae,
        global_step,
        accelerator
    ):
        # Only log on main process
        if not accelerator.is_main_process:
            return

        # Convert to numpy and correct format (B,C,H,W) -> (B,H,W,C)
        # Take only first 4 images from batch for visualization
        depth_images = (depth_prediction[:4] / 2 + 0.5).clamp(0, 1)
        recon_images = (reconstructed_rgb[:4] / 2 + 0.5).clamp(0, 1)
        input_images = (input_images[:4] / 2 + 0.5).clamp(0, 1) 
        
        # Ensure all images are properly detached and moved to CPU
        depth_images = depth_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        recon_images = recon_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        input_images = input_images.detach().cpu().permute(0, 2, 3, 1).float().numpy()

        # Create a list to store combined images with captions
        image_logs = []
        
        # Combine corresponding images from each batch with appropriate captions
        for idx in range(min(4, len(input_images))):
            image_logs.extend([
                wandb.Image(input_images[idx], caption=f"Input {idx}"),
                wandb.Image(depth_images[idx], caption=f"Depth {idx}"),
                wandb.Image(recon_images[idx], caption=f"Reconstructed {idx}")
            ])

        # Log images and losses
        accelerator.log(
            {
                "training_samples": image_logs,
                "depth_prediction_loss": depth_prediction_loss.item(),
                "reconstructed_rgb_loss": reconstructed_rgb_loss.item(),
                "total_loss": total_loss.item(),
            },
            step=global_step,
        )