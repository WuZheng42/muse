import torch
from tqdm import tqdm
from einops import rearrange
from musetalk.data import LoraDataUtils
from musetalk.utils.utils import load_all_model
from peft import get_peft_model, LoraConfig
import torch.nn as nn
from musetalk.loss.basic_loss import Interpolate
import musetalk.loss.vgg_face as vgg_face
from diffusers.optimization import get_scheduler


def Adaptation(unet_model_path, vae_type, unet_config, Epoch, adam_beta1, adam_beta2, adam_epsilon, adam_weight_decay,
               learning_rate=2.0e-5, l1_loss_weight=1.0, vgg_loss_weight=0.01,
               vgg_layer_weight=[1, 1, 1, 1, 1], pyramid_scale=[1, 0.5, 0.25, 0.125], device='cuda:0'):
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type=vae_type,
        unet_config=unet_config,
        device=device
    )
    pe.eval()
    vae_model = vae.vae
    vae_model.eval()
    vae_model.requires_grad_(False)
    pe.requires_grad_(False)
    unet_model = unet.model
    unet_model.train()
    # 配置 LoRA
    lora_config = LoraConfig(
        r=32,  # LoRA 的秩，即低秩矩阵的维度，决定了参数调整的自由度
        lora_alpha=16,  # 缩放系数，控制 LoRA 权重对模型的影响
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0"  # 指定 UNet 的 LoRA 应用对象（用于调整 UNet 中的注意力机制）
        ],
        lora_dropout=0  # LoRA dropout 概率，0 表示不使用 dropout
    )

    # 将 LoRA 应用到模型中
    lora_unet = get_peft_model(unet, lora_config)

    train_dataloader = LoraDataUtils.GetDataLoader()

    # 损失函数
    if vgg_loss_weight > 0:
        vgg_IN = vgg_face.Vgg19().to(device)
        pyramid = vgg_face.ImagePyramide(
            pyramid_scale, 3).to(device)
        vgg_IN.eval()
        downsampler = Interpolate(
            size=(224, 224), mode='bilinear', align_corners=False).to(device)
    l1_losser = nn.L1Loss(reduction='mean')

    # 准备优化器
    trainable_params = list(filter(lambda p: p.requires_grad, lora_unet.parameters()))
    print('trainable params')
    for n, p in lora_unet.named_parameters():
        if p.requires_grad:
            print(n)

    optimizer_cls = torch.optim.AdamW  # optimizer_cls = bnb.optim.AdamW8bit

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    lr_warmup_steps = 1000
    gradient_accumulation_steps = 1
    scheduler_max_steps = len(train_dataloader) * Epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=scheduler_max_steps,
    )

    for epoch in range(Epoch):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        l1_loss_accum = 0.
        vgg_loss_accum = 0.
        # pixel_values-ref_pixel_values是视频（bsz, num_frames, c, h, w）
        # audio_prompts是process_audio_features(cfg, batch, model_dict['wav2vec'], bsz, num_frames, weight_dtype)来的
        for pixel_values, ref_pixel_values, audio_prompts in pbar:
            with torch.no_grad():
                bsz, num_frames, c, h, w = pixel_values.shape

                # Extract frames for backward pass
                frames_left_index = 0
                frames_right_index = num_frames
                pixel_values_backward = pixel_values[:, frames_left_index:frames_right_index, ...]
                ref_pixel_values_backward = ref_pixel_values[:, frames_left_index:frames_right_index, ...]
                audio_prompts_backward = audio_prompts[:, frames_left_index:frames_right_index, ...]

                # Encode target images
                frames = rearrange(pixel_values_backward, 'b f c h w-> (b f) c h w')
                # latents = vae_model.encode(frames).latent_dist.mode()
                # latents = latents * vae_model.config.scaling_factor
                # latents = latents.float()

                # Create masked images
                masked_pixel_values = pixel_values_backward.clone()
                masked_pixel_values[:, :, :, h // 2:, :] = -1
                masked_frames = rearrange(masked_pixel_values, 'b f c h w -> (b f) c h w')
                masked_latents = vae_model.encode(masked_frames).latent_dist.mode()
                masked_latents = masked_latents * vae_model.config.scaling_factor
                masked_latents = masked_latents.float()

                # Encode reference images
                ref_frames = rearrange(ref_pixel_values_backward, 'b f c h w-> (b f) c h w')
                ref_latents = vae_model.encode(ref_frames).latent_dist.mode()
                ref_latents = ref_latents * vae_model.config.scaling_factor
                ref_latents = ref_latents.float()

                audio_prompts_backward = rearrange(
                    audio_prompts_backward,
                    'b f c h w-> (b f) c h w'
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward,
                    '(b f) c h w -> (b f) (c h) w',
                    b=bsz
                )

            input_latents = torch.cat([masked_latents, ref_latents], dim=1)
            # TimeSteps (usually 0 for this task)
            time_steps = torch.tensor([0], device=input_latents.device)

            # Forward pass
            latents_pred = lora_unet(
                input_latents,
                time_steps,
                audio_prompts_backward,
            )

            latents_pred = (1 / vae_model.config.scaling_factor) * latents_pred
            image_pred = vae_model.decode(latents_pred).sample

            # Calculate L1 loss
            l1_loss = l1_losser(frames, image_pred)
            l1_loss_accum += l1_loss.item()
            loss = l1_loss_weight * l1_loss

            # Process VGG loss if enabled
            if vgg_loss_weight > 0:
                pyramide_real = pyramid(downsampler(frames))
                pyramide_generated = pyramid(downsampler(image_pred))

                loss_IN = 0
                for scale in pyramid_scale:
                    x_vgg = vgg_IN(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = vgg_IN(pyramide_real['prediction_' + str(scale)])
                    for i, weight in enumerate(vgg_layer_weight):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        loss_IN += weight * value
                loss_IN /= sum(vgg_layer_weight)
                loss += loss_IN * vgg_loss_weight
                vgg_loss_accum += loss_IN.item()

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            lr_scheduler.step()


Adaptation(unet_config="./models/musetalkV15/musetalk.json",
           unet_model_path="./models/musetalkV15/unet.pth", vae_type='sd-vae')
