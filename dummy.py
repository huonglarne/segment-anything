import torch

boxes = torch.load("boxes.pt")
crop_box_torch = torch.load("crop_box_torch.pt")
atol = 1e-3

print("boxes: ", boxes)
print("crop_box_torch: ", crop_box_torch)

near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :])
