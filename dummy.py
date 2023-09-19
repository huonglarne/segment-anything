
# import cv2


# img = cv2.imread("assets/notebook1.png")

# from segment_anything import SamPredictor, sam_model_registry
# sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
# predictor = SamPredictor(sam)
# predictor.set_image(img)
# masks, _, _ = predictor.predict("mask")

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(img)
import torch

boxes = torch.load("boxes.pt")
crop_box_torch = torch.load("crop_box_torch.pt")
atol = 1e-3

print("boxes: ", boxes)
print("crop_box_torch: ", crop_box_torch)

near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
