import torch
ckpt = torch.load('models/detect-20241225.ckpt', map_location='cpu')
# Check for architecture hints
print(ckpt.keys())
print(ckpt.get('model', ckpt.get('arch', 'unknown')))