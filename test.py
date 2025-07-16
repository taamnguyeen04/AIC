# import torch
#
# ckpt = torch.load('co_dino_5scale_swin_large_16e_o365tococo.pth', map_location='cpu')
# print(ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))

import torch
print(torch.__version__)
print(torch.version.cuda)

import mmcv
print(mmcv.__version__)
