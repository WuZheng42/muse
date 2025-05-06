import argparse
import diffusers
import logging # Use standard logging for setup before accelerate logger
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import warnings
import random

from accelerate import Accelerator
from accelerate.utils import LoggerType, set_seed
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger # Import accelerate logger function
from accelerate.utils import DistributedDataParallelKwargs
from datetime import datetime, timedelta

# Check diffusers version if needed (optional)
# from diffusers.utils import check_min_version
# check_min_version("0.17.0") # Example version check

from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# --- Import necessary functions from your project structure ---
try:
    # Utils for general helpers and LoRA-aware saving
    from musetalk.utils.utils import (
        delete_additional_ckpt,
        seed_everything, # Or use accelerate.set_seed
        get_mouth_region,
        process_audio_features, # Ensure this works with accelerator.device
        save_models # IMPORTANT: This function MUST be LoRA-aware
    )
    # Loss functions
    from musetalk.loss.basic_loss import set_requires_grad
    from musetalk.loss.syncnet import get_sync_loss
    # Initialization and validation functions (LoRA-aware)
    from musetalk.utils.training_utils import (
        initialize_models_and_optimizers, # IMPORTANT: Must handle LoRA setup based on cfg
        initialize_dataloaders,
        initialize_loss_functions,
        initialize_syncnet,
        initialize_vgg,
        validation # IMPORTANT: Must correctly generate samples with LoRA weights active
    )
except ImportError as e:
    # Use standard logging here as accelerate logger might not be ready
    logging.error(f"Error importing from musetalk: {e}")
    logging.error("Please ensure the musetalk package is correctly installed and paths are correct.")
    logging.error("Also ensure training_utils.py and utils.py contain the necessary LoRA modifications.")
    exit(1)

# --- Setup standard logging temporarily for early messages ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
warnings.filterwarnings("ignore")


# --- Helper Function for GAN Loss Weight ---
def get_ganloss_weight(step, start_step):
    """Calculate GAN loss weight based on training step"""
    if step < start_step:
        return 0.0
    else:
        return 1.0

# --- Main Training Function ---
def main(cfg):
    # --- Accelerator Setup ---
    exp_name = cfg.exp_name
    # Use os.path.join for better path handling
    save_dir = os.path.join(cfg.output_dir, exp_name)
    # Create directory only on the main process (checked later)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=cfg.get("log_with", ["tensorboard"]),
        project_dir=os.path.join(save_dir, "logs"),
        mixed_precision=cfg.solver.get("mixed_precision", "no"),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=cfg.get("find_unused_parameters", True)), # Configurable
            InitProcessGroupKwargs(timeout=timedelta(seconds=cfg.get("accelerate_timeout", 5400)))
        ]
    )

    # --- Logger Setup (AFTER Accelerator Init) ---
    # Now it's safe to get the accelerate logger
    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"Accelerator initialized: {accelerator.state}")

    # Create output directory on main process
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Experiment outputs will be saved in: {save_dir}")

    # Set verbosity based on process rank
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # --- Seed ---
    if cfg.seed is not None:
        # Use accelerate's set_seed for consistency across processes
        # seed_everything might not be needed if using set_seed
        set_seed(cfg.seed + accelerator.process_index)
        logger.info(f"Seed set to {cfg.seed + accelerator.process_index} for process {accelerator.process_index}")

    # --- Determine Weight Dtype ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"Using mixed precision: {accelerator.mixed_precision}, weight dtype for loading: {weight_dtype}")

    # --- Initialization (Using Imported Functions) ---
    # These functions MUST correctly handle LoRA based on cfg and use accelerator
    logger.info("Initializing models and optimizer...")
    # Pass accelerator object as expected by the function signature
    model_dict = initialize_models_and_optimizers(cfg, accelerator, weight_dtype)

    logger.info("Initializing dataloaders...")
    dataloader_dict = initialize_dataloaders(cfg) # Assumes this doesn't need accelerator

    logger.info("Initializing loss functions...")
    loss_dict = initialize_loss_functions(cfg, accelerator, model_dict['scheduler_max_steps'])

    logger.info("Initializing SyncNet...")
    syncnet = initialize_syncnet(cfg, accelerator, weight_dtype)

    logger.info("Initializing VGG...")
    vgg_IN, pyramid, downsampler = initialize_vgg(cfg, accelerator)

    # --- Prepare with Accelerator ---
    logger.info("Preparing components with Accelerator...")
    try:
        # Ensure all components intended for prepare exist
        components_to_prepare = [
            model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler']
        ]
        if dataloader_dict.get('train_dataloader'):
            components_to_prepare.append(dataloader_dict['train_dataloader'])
        if dataloader_dict.get('val_dataloader'):
             components_to_prepare.append(dataloader_dict['val_dataloader'])
        else:
             logger.warning("Validation dataloader not found in dataloader_dict, skipping prepare.")

        prepared_components = accelerator.prepare(*components_to_prepare)

        # Assign back prepared components carefully based on what was passed
        comp_idx = 0
        net = prepared_components[comp_idx]; comp_idx += 1
        optimizer = prepared_components[comp_idx]; comp_idx += 1
        lr_scheduler = prepared_components[comp_idx]; comp_idx += 1
        if dataloader_dict.get('train_dataloader'):
            train_dataloader = prepared_components[comp_idx]; comp_idx += 1
        else:
             train_dataloader = None
        if dataloader_dict.get('val_dataloader'):
            val_dataloader = prepared_components[comp_idx]; comp_idx += 1
        else:
            val_dataloader = None

        # Prepare discriminators separately if they exist
        if loss_dict.get('discriminator'):
            discriminator, optimizer_D, scheduler_D = accelerator.prepare(
                loss_dict['discriminator'], loss_dict['optimizer_D'], loss_dict['scheduler_D']
            )
            loss_dict['discriminator'] = discriminator
            loss_dict['optimizer_D'] = optimizer_D
            loss_dict['scheduler_D'] = scheduler_D
            logger.info("Discriminator prepared.")
        if loss_dict.get('mouth_discriminator'):
            mouth_discriminator, mouth_optimizer_D, mouth_scheduler_D = accelerator.prepare(
                loss_dict['mouth_discriminator'], loss_dict['mouth_optimizer_D'], loss_dict['mouth_scheduler_D']
            )
            loss_dict['mouth_discriminator'] = mouth_discriminator
            loss_dict['mouth_optimizer_D'] = mouth_optimizer_D
            loss_dict['mouth_scheduler_D'] = mouth_scheduler_D
            logger.info("Mouth Discriminator prepared.")

    except Exception as e:
        logger.error(f"Error during accelerator.prepare: {e}", exc_info=True)
        raise

    # Update model_dict (optional, but helps keep track)
    model_dict['net'] = net
    model_dict['optimizer'] = optimizer
    model_dict['lr_scheduler'] = lr_scheduler
    # Access unprepared (frozen) models directly
    vae = model_dict['vae']
    wav2vec = model_dict['wav2vec']

    if train_dataloader:
        logger.info(f"Prepared Train Dataloader length: {len(train_dataloader)}")
    if val_dataloader:
        logger.info(f"Prepared Validation Dataloader length: {len(val_dataloader)}")

    # --- Training Prep ---
    if not train_dataloader:
        logger.error("Train dataloader is None after prepare. Exiting.")
        return

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.solver.gradient_accumulation_steps) if len(train_dataloader) > 0 else 0
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else 1

    # Initialize trackers on the main process
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        try:
            # Convert OmegaConf to dict for logging
            config_log = OmegaConf.to_container(cfg, resolve=True)
            accelerator.init_trackers(
                project_name=cfg.get("project_name", "musetalk-lora-finetune"), # Configurable
                config=config_log,
                init_kwargs={"tensorboard": {"name": run_time}}
            )
            logger.info("Initialized trackers.")
        except Exception as e:
            logger.warning(f"Failed to initialize trackers: {e}")

    total_batch_size = (cfg.data.train_bs * accelerator.num_processes * cfg.solver.gradient_accumulation_steps)

    logger.info("***** Running Training *****")
    # Use accelerator.print for distributed-safe printing
    accelerator.print(f"  Num examples = {len(train_dataloader.dataset)}") # Approx
    accelerator.print(f"  Num Epochs = {num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    accelerator.print(f"  LoRA Enabled = {cfg.get('lora', {}).get('enable_lora', False)}")

    global_step = 0
    first_epoch = 0

    # --- Checkpoint Loading ---
    if cfg.resume_from_checkpoint:
        # Try to find the latest checkpoint in the specified save_dir
        checkpoint_dirs = []
        if accelerator.is_main_process and os.path.exists(save_dir): # Check existence on main process
             checkpoint_dirs = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
        # Broadcast found checkpoints or lack thereof
        checkpoint_dirs = accelerator.broadcast_object_list(checkpoint_dirs)

        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(save_dir, checkpoint_dirs[-1])
            accelerator.print(f"Resuming from checkpoint: {latest_checkpoint}")
            try:
                # load_state should handle mapping location automatically
                accelerator.load_state(latest_checkpoint)
                global_step = int(latest_checkpoint.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0
                # resume_step = global_step % num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0
                accelerator.print(f"Resumed from global step: {global_step}")
            except Exception as e:
                accelerator.print(f"Failed to load checkpoint state from {latest_checkpoint}: {e}")
                accelerator.print("Starting training from scratch.")
                global_step = 0
                first_epoch = 0
        else:
            accelerator.print(f"resume_from_checkpoint is True, but no valid checkpoint found in {save_dir}. Starting from scratch.")

    # --- Progress Bar ---
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps"
    )

    # --- Get GAN Start Step ---
    gan_start_step = cfg.discriminator_train_params.start_gan

    # --- Training Loop ---
    for epoch in range(first_epoch, num_train_epochs):
        net.train()
        if loss_dict.get('discriminator'): loss_dict['discriminator'].train()
        if loss_dict.get('mouth_discriminator'): loss_dict['mouth_discriminator'].train()

        train_loss_accum = 0.0
        l1_loss_accum = 0.0
        sync_loss_accum = 0.0
        train_loss_D_accum = 0.0
        train_loss_Dm_accum = 0.0
        # Add other loss accumulators if needed

        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()

            # --- Forward Pass (Generator) ---
            with accelerator.accumulate(net):
                # --- Data Preparation ---
                try:
                    pixel_values = batch["pixel_values_vid"].to(accelerator.device, dtype=weight_dtype)
                    ref_pixel_values = batch["pixel_values_ref_img"].to(accelerator.device, dtype=weight_dtype)
                    pixel_values_face_mask = batch['pixel_values_face_mask'].to(accelerator.device)
                    mels = batch['mel'].to(accelerator.device) if syncnet and 'mel' in batch else None
                    bsz, num_frames, c, h, w = pixel_values.shape
                except Exception as e:
                    logger.error(f"Error processing batch keys/moving to device at step {global_step}: {e}")
                    continue

                # Process audio features
                try:
                    # Pass accelerator.device explicitly if the function needs it
                    audio_prompts = process_audio_features(cfg, batch, wav2vec, bsz, num_frames, weight_dtype, accelerator.device)
                except Exception as e:
                    logger.error(f"Error in process_audio_features at step {global_step}: {e}")
                    continue

                # --- Prepare Latents (VAE encoding in no_grad) ---
                with torch.no_grad():
                    # Using all frames for simplicity
                    pixel_values_backward = pixel_values
                    ref_pixel_values_backward = ref_pixel_values
                    pixel_values_face_mask_backward = pixel_values_face_mask
                    audio_prompts_backward = audio_prompts

                    frames = rearrange(pixel_values_backward, 'b f c h w-> (b f) c h w').float()
                    latents = vae.encode(frames).latent_dist.mode() * vae.config.scaling_factor

                    masked_pixel_values = pixel_values_backward.clone()
                    masked_pixel_values[:, :, :, h//2:, :] = -1
                    masked_frames = rearrange(masked_pixel_values, 'b f c h w-> (b f) c h w').float()
                    masked_latents = vae.encode(masked_frames).latent_dist.mode() * vae.config.scaling_factor

                    ref_frames = rearrange(ref_pixel_values_backward, 'b f c h w-> (b f) c h w').float()
                    ref_latents = vae.encode(ref_frames).latent_dist.mode() * vae.config.scaling_factor

                    audio_unet_prompts = rearrange(audio_prompts_backward, 'b f c h w-> (b f) c h w')
                    audio_unet_prompts = rearrange(audio_unet_prompts, '(b f) c h w -> (b f) (c h) w', b=bsz)

                    input_latents = torch.cat([masked_latents.to(weight_dtype), ref_latents.to(weight_dtype)], dim=1)
                    timesteps = torch.tensor([0], device=accelerator.device)

                # --- Actual UNet Forward Pass ---
                # Model is already prepared, use it directly
                latents_pred = net(input_latents, timesteps, audio_unet_prompts.to(weight_dtype))
                latents_pred = (1 / vae.config.scaling_factor) * latents_pred
                image_pred = vae.decode(latents_pred).sample.float()

                # --- Loss Calculation (Generator) ---
                loss_G_total = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                adapted_weight = 1.0 # Simplified

                # L1 Loss
                l1_loss = loss_dict['L1_loss'](frames, image_pred)
                loss_G_total = loss_G_total + cfg.loss_params.l1_loss * l1_loss * adapted_weight
                l1_loss_val = l1_loss.item()

                # Sync Loss
                sync_loss_val = 0.0
                if syncnet and cfg.loss_params.sync_loss > 0 and mels is not None:
                    with torch.no_grad():
                        audio_embed = syncnet.get_audio_embed(mels)
                        gt_frames_sync = frames[:, :, h // 2:, :]
                    pred_frames_sync = image_pred[:, :, h // 2 :, :]
                    sync_loss, _ = get_sync_loss(audio_embed, gt_frames_sync, pred_frames_sync, syncnet, adapted_weight)
                    loss_G_total = loss_G_total + sync_loss * cfg.loss_params.sync_loss
                    sync_loss_val = sync_loss.item()

                # VGG Loss (Add if needed)
                vgg_loss_val = 0.0
                if vgg_IN and cfg.loss_params.vgg_loss > 0:
                    # ... (VGG loss calculation logic) ...
                    # loss_G_total = loss_G_total + loss_IN * cfg.loss_params.vgg_loss * adapted_weight
                    # vgg_loss_val = loss_IN.item()
                    pass

                # GAN Loss (Generator) & Feature Matching
                gan_g_loss_val = 0.0
                fm_loss_val = 0.0
                gan_weight = get_ganloss_weight(global_step, gan_start_step)
                if loss_dict.get('discriminator') and cfg.loss_params.gan_loss > 0:
                    set_requires_grad(loss_dict['discriminator'], False)
                    # ... (GAN G loss and FM loss calculation logic) ...
                    # loss_G_total = loss_G_total + loss_G * cfg.loss_params.gan_loss * gan_weight * adapted_weight
                    # loss_G_total = loss_G_total + L_feature_matching * adapted_weight # Check scaling
                    # gan_g_loss_val = loss_G.item()
                    # fm_loss_val = L_feature_matching.item()
                    pass

                # Mouth GAN Loss (Generator) & Feature Matching
                mouth_gan_g_loss_val = 0.0
                if loss_dict.get('mouth_discriminator') and cfg.loss_params.mouth_gan_loss > 0:
                    set_requires_grad(loss_dict['mouth_discriminator'], False)
                    # ... (Mouth GAN G loss and FM loss calculation logic) ...
                    # loss_G_total = loss_G_total + mouth_loss_G * cfg.loss_params.mouth_gan_loss * gan_weight * adapted_weight
                    # loss_G_total = loss_G_total + L_mouth_fm * adapted_weight # Check scaling
                    # mouth_gan_g_loss_val = mouth_loss_G.item()
                    # fm_loss_val += L_mouth_fm.item() # Add mouth FM loss
                    pass

                # --- Generator Backward ---
                accelerator.backward(loss_G_total) # Handles scaling and accumulation

                # Accumulate unscaled losses for logging this cycle
                train_loss_accum += loss_G_total.item()
                l1_loss_accum += l1_loss_val
                sync_loss_accum += sync_loss_val
                # Accumulate others (vgg, gan_g, fm, etc.)

                # --- Discriminator Training Step ---
                # Needs to be handled carefully with accelerator.accumulate if D params are separate
                loss_D_val = 0.0
                loss_Dm_val = 0.0
                if gan_weight > 0: # Only train D if GAN loss is active
                    if loss_dict.get('discriminator') and cfg.loss_params.gan_loss > 0:
                         with accelerator.accumulate(loss_dict['discriminator']): # Use D model for context
                             set_requires_grad(loss_dict['discriminator'], True)
                             # ... Calculate D loss ...
                             # loss_D = loss_dict['discriminator_full'](...)
                             # loss_D_val = loss_D.item()
                             # accelerator.backward(loss_D * gan_weight * adapted_weight) # Backward for D
                             pass

                    if loss_dict.get('mouth_discriminator') and cfg.loss_params.mouth_gan_loss > 0:
                         with accelerator.accumulate(loss_dict['mouth_discriminator']):
                             set_requires_grad(loss_dict['mouth_discriminator'], True)
                             # ... Calculate Mouth D loss ...
                             # mouth_loss_D = loss_dict['mouth_discriminator_full'](...)
                             # loss_Dm_val = mouth_loss_D.item()
                             # accelerator.backward(mouth_loss_D * gan_weight * adapted_weight) # Backward for Mouth D
                             pass
                # Accumulate D losses for logging
                train_loss_D_accum += loss_D_val
                train_loss_Dm_accum += loss_Dm_val

            # --- Optimizer Step ---
            if accelerator.sync_gradients:
                # Clip gradients (optional but recommended)
                if cfg.solver.max_grad_norm > 0:
                    # Clip generator (LoRA or full) gradients
                    accelerator.clip_grad_norm_(net.parameters(), cfg.solver.max_grad_norm)
                    # Clip discriminator gradients
                    if loss_dict.get('discriminator') and gan_weight > 0:
                         accelerator.clip_grad_norm_(loss_dict['discriminator'].parameters(), cfg.solver.max_grad_norm)
                    if loss_dict.get('mouth_discriminator') and gan_weight > 0:
                         accelerator.clip_grad_norm_(loss_dict['mouth_discriminator'].parameters(), cfg.solver.max_grad_norm)

                # Step optimizers
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if loss_dict.get('optimizer_D') and gan_weight > 0:
                    loss_dict['optimizer_D'].step()
                    if loss_dict.get('scheduler_D'): loss_dict['scheduler_D'].step()
                    loss_dict['optimizer_D'].zero_grad(set_to_none=True)

                if loss_dict.get('mouth_optimizer_D') and gan_weight > 0:
                    loss_dict['mouth_optimizer_D'].step()
                    if loss_dict.get('mouth_scheduler_D'): loss_dict['mouth_scheduler_D'].step()
                    loss_dict['mouth_optimizer_D'].zero_grad(set_to_none=True)

                # --- Logging & Saving ---
                progress_bar.update(1)
                global_step += 1

                # Log average losses for the completed accumulation cycle
                if global_step % cfg.get("logging_steps", 10) == 0: # Log every N steps
                    avg_loss_G = train_loss_accum / (cfg.solver.gradient_accumulation_steps * cfg.get("logging_steps", 10))
                    avg_l1_loss = l1_loss_accum / (cfg.solver.gradient_accumulation_steps * cfg.get("logging_steps", 10))
                    avg_sync_loss = sync_loss_accum / (cfg.solver.gradient_accumulation_steps * cfg.get("logging_steps", 10))
                    avg_loss_D = train_loss_D_accum / (cfg.solver.gradient_accumulation_steps * cfg.get("logging_steps", 10))
                    avg_loss_Dm = train_loss_Dm_accum / (cfg.solver.gradient_accumulation_steps * cfg.get("logging_steps", 10))

                    # Gather losses across processes for accurate logging
                    avg_loss_G = accelerator.gather(torch.tensor(avg_loss_G, device=accelerator.device)).mean().item()
                    avg_l1_loss = accelerator.gather(torch.tensor(avg_l1_loss, device=accelerator.device)).mean().item()
                    avg_sync_loss = accelerator.gather(torch.tensor(avg_sync_loss, device=accelerator.device)).mean().item()
                    avg_loss_D = accelerator.gather(torch.tensor(avg_loss_D, device=accelerator.device)).mean().item()
                    avg_loss_Dm = accelerator.gather(torch.tensor(avg_loss_Dm, device=accelerator.device)).mean().item()

                    if accelerator.is_main_process:
                        logs = {
                            "Loss/Generator_Total": avg_loss_G,
                            "Loss/L1": avg_l1_loss,
                            "LR/unet": lr_scheduler.get_last_lr()[0],
                        }
                        if avg_sync_loss > 0: logs["Loss/SyncNet"] = avg_sync_loss
                        if avg_loss_D > 0: logs["Loss/Discriminator"] = avg_loss_D
                        if avg_loss_Dm > 0: logs["Loss/Mouth_Discriminator"] = avg_loss_Dm
                        # Add other losses (VGG, GAN_G, FM) if tracked

                        accelerator.log(logs, step=global_step)
                        progress_bar.set_postfix({
                            "Loss_G": f"{avg_loss_G:.4f}", "L1": f"{avg_l1_loss:.4f}", "Sync": f"{avg_sync_loss:.4f}",
                            "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

                    # Reset accumulators after logging
                    train_loss_accum = 0.0
                    l1_loss_accum = 0.0
                    sync_loss_accum = 0.0
                    train_loss_D_accum = 0.0
                    train_loss_Dm_accum = 0.0
                    # Reset others

                # --- Validation ---
                if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                    if val_dataloader:
                        logger.info(f"Running validation for step {global_step}...")
                        val_start_time = time.time()
                        try:
                            # Call the imported validation function
                            validation(
                                cfg=cfg, val_dataloader=val_dataloader,
                                net=accelerator.unwrap_model(net), vae=vae, wav2vec=wav2vec,
                                accelerator=accelerator, save_dir=save_dir, global_step=global_step,
                                weight_dtype=weight_dtype,
                                # syncnet_score=avg_sync_loss # Pass score if needed
                            )
                            logger.info(f"Validation took {time.time() - val_start_time:.2f}s")
                        except Exception as e:
                            logger.error(f"Error during validation call: {e}", exc_info=True)
                    else:
                        logger.warning("Validation frequency reached, but no validation dataloader available.")

                # --- Checkpointing ---
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_start_time = time.time()
                        state_save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(state_save_path)
                        logger.info(f"Accelerator state saved to {state_save_path}")

                        # Save Model Weights (LoRA or Full) using the modified save_models
                        save_models(
                            accelerator=accelerator, net=net, save_dir=save_dir,
                            global_step=global_step, cfg=cfg, logger=logger
                        )
                        # Clean up old accelerator state checkpoints
                        delete_additional_ckpt(save_dir, cfg.total_limit, logger=logger)
                        logger.info(f"Checkpointing took {time.time() - save_start_time:.2f}s")
                    # Ensure all processes wait for main process to finish saving
                    accelerator.wait_for_everyone()


            # --- Check Max Steps ---
            if global_step >= cfg.solver.max_train_steps:
                break # Exit inner loop (step loop)

        # --- End of Epoch ---
        logger.info(f"Epoch {epoch+1}/{num_train_epochs} finished.")
        accelerator.wait_for_everyone()

        if global_step >= cfg.solver.max_train_steps:
            logger.info("Maximum training steps reached. Exiting training.")
            break # Exit outer loop (epoch loop)

    # --- End of Training ---
    logger.info("Training finished.")
    accelerator.end_training()

    # Final save on main process
    if accelerator.is_main_process:
        logger.info("Performing final save...")
        final_state_path = os.path.join(save_dir, f"checkpoint-{global_step}")
        accelerator.save_state(final_state_path)
        logger.info(f"Final accelerator state saved to {final_state_path}")
        save_models(accelerator, net, save_dir, global_step, cfg, logger)


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuseTalk LoRA Fine-tuning with Accelerate")
    parser.add_argument(
        "--config",
        type=str,
        default= "./configs/training/stage2_lora.yaml",
        required=False,
        help="Path to the YAML configuration file (e.g., ./configs/training/stage2_lora.yaml)"
    )
    args = parser.parse_args()

    # --- Early Logging Setup (Standard Python Logger) ---
    # This helps catch errors before accelerate logger is ready
    try:
        # Basic config loading check before main function
        if not os.path.exists(args.config):
             logging.error(f"Configuration file not found at: {args.config}")
             exit(1)
        config = OmegaConf.load(args.config)
        logging.info(f"Configuration loaded from: {args.config}")
        # Call the main function
        main(config)
    except FileNotFoundError:
        # This case is handled above, but kept for robustness
        logging.error(f"Configuration file not found at: {args.config}")
    except ImportError as ie:
         logging.error(f"Import Error: {ie}. Check dependencies and musetalk imports.")
    except Exception as e:
        # Use standard logger here as accelerate might not be initialized
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)