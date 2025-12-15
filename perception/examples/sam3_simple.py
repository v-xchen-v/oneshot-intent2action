from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# # Segment using bounding box prompt
# box_xyxy = [100, 150, 500, 450]
# input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
# input_boxes_labels = [[1]]  # 1 = positive box

# inputs = processor(
#     images=image,
#     input_boxes=input_boxes,
#     input_boxes_labels=input_boxes_labels,
#     return_tensors="pt"
# ).to(device)

    
# # Segment using text prompt
# inputs = processor(images=image, text="ear", return_tensors="pt").to(device)

# # Segment with bounding box and text prompt
# box_xyxy = [100, 150, 500, 450]
# input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
# input_boxes_labels = [[1]]  # 1 = positive box
# inputs = processor(
#     images=image,
#     input_boxes=input_boxes,
#     input_boxes_labels=input_boxes_labels,
#     text="cat",
#     return_tensors="pt"
# ).to(device)

# Segment with bounding box and text prompt
box_xyxy = [100, 150, 500, 450]
input_boxes = [[box_xyxy]]  # [batch, num_boxes, 4]
input_boxes_labels = [[1]]  # 1 = positive box
inputs = processor(
    images=image,
    input_boxes=input_boxes,
    input_boxes_labels=input_boxes_labels,
    text="cat",
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores


# sort masks by scores descending
if "scores" in results:
    scores = results["scores"]
    sorted_indices = torch.argsort(scores, descending=True)
    results["masks"] = results["masks"][sorted_indices]
    results["boxes"] = results["boxes"][sorted_indices]
    results["scores"] = results["scores"][sorted_indices]
        
# Save masks as images
for i, mask in enumerate(results["masks"]):
    mask_image = Image.fromarray((mask.cpu().numpy() * 255).astype("uint8"))
    mask_image.save(f"mask_{i}.png")
    
import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

# Overlay and save
overlay_image = overlay_masks(image, results["masks"][:1]) # Overlay first mask
overlay_image.save("overlay_masks.png")