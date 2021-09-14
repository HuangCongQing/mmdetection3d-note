
# https://github.com/open-mmlab/mmaction2/issues/621#:~:text=RuntimeError%3A%20CUDA%20error%3A%20invalid%20device%20function%20Segmentation%20fault,you%20can%20raise%20an%20issue%20under%20mmcv%20repo.


import torch
from mmcv.ops import RoIAlign
roi_align = RoIAlign(output_size=(7, 7)).cuda()
feat = torch.randn([1, 3, 24, 24]).cuda()
roi = torch.tensor([[0., 0., 0., 1., 1.]], dtype=torch.float32).cuda()
ret = roi_align(feat, roi)
print(ret.shape) # torch.Size([1, 3, 7, 7])
