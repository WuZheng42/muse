from musetalk.models.unet import UNet
from musetalk.models.lora import LoRALinear,LoRAConv2d
from torch import nn


def replace_linear(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            old_linear = getattr(model, name)
            new_linear = LoRALinear(in_features=old_linear.in_features, out_features=old_linear.out_features,
                                    r=2, lora_alpha=1, lora_dropout=0, merge_weights=True,
                                    bias=(old_linear.bias is not None))
            # 复制原有的权重和偏置
            new_linear.weight.data = old_linear.weight.data.clone()
            if old_linear.bias is not None:
                new_linear.bias.data = old_linear.bias.data.clone()
            setattr(model, name, new_linear)
        else:
            replace_linear(module)


# def replace_conv2d(model):
#     for name, module in list(model.named_children()):
#         if isinstance(module, nn.Conv2d):
#             old_conv2d = getattr(model, name)
#             new_conv2d = LoRAConv2d(in_features=old_linear.in_features, out_features=old_linear.out_features,
#                                     r=2, lora_alpha=1, lora_dropout=0, merge_weights=True,
#                                     bias=(old_linear.bias is not None))
#             # 复制原有的权重和偏置
#             new_linear.weight.data = old_linear.weight.data.clone()
#             if old_linear.bias is not None:
#                 new_linear.bias.data = old_linear.bias.data.clone()
#             setattr(model, name, new_conv2d)
#         else:
#             replace_linear(module)

unet = UNet(
    unet_config="./models/musetalkV15/musetalk.json",
    model_path="./models/musetalkV15/unet.pth",
    device='cpu'
)
model = unet.model
replace_linear(model)
print(model)
