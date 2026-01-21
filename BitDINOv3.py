
import torch
from bitlayers.dinov3.models.vision_transformer import vit_small, vit_base, vit_large, vit_so400m, vit_huge2, vit_giant2, vit_7b
from common_utils import convert_to_ternary, summ

model = vit_small()
model.init_weights()
info = summ(model)
print(sum([i[2] for i in info])/1024/1024)
torch.save(model.state_dict(), "vit_small.pth")
torch.save(convert_to_ternary(model).state_dict(), "vit_small_tern.pth")

# from bitlayers.dinov3.models.convnext import get_convnext_arch

# ConvNeXtTiny = get_convnext_arch("convnext_tiny")
# model = ConvNeXtTiny(patch_size=16)
# model.init_weights()


