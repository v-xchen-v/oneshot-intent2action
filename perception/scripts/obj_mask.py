"""Generate object masks from images using bounding boxes and/or text prompts.

This script supports three SAM3 prompting modes:
1. Bounding box only
2. Text description only  
3. Combined bounding box + text

Usage Examples:
    # Text + bbox (recommended for best accuracy)
    python obj_mask.py \\
        --frame_path ./testcase/color/0.png \\
        --object_name "red mug" \\
        --bbox_xywh "100,150,200,300" \\
        --output_mask_path ./testcase/0_mask.png
    
    # Bbox only
    python obj_mask.py \\
        --frame_path ./testcase/color/0.png \\
        --bbox_xywh "100,150,200,300" \\
        --output_mask_path ./testcase/0_mask.png
    
    # Text only
    python obj_mask.py \\
        --frame_path ./testcase/color/0.png \\
        --object_name "red mug" \\
        --output_mask_path ./testcase/0_mask.png
    
    # Optional: Visualize overlay
    python obj_mask.py \\
        --frame_path ./testcase/color/0.png \\
        --object_name "red mug" \\
        --bbox_xywh "100,150,200,300" \\
        --output_mask_path ./testcase/0_mask.png \\
        --output_overlay_path ./testcase/0_overlay.png
"""

import torch
import numpy as np
from PIL import Image
from typing import List


def inference(inputs, model, processor, threshold=0.5, mask_threshold=0.5):
    """Run SAM3 inference and post-process results."""
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    print(f"Found {len(results['masks'])} objects")
    return results['masks']


def gen_obj_mask_bbox(processor, model, image: Image.Image, box_xyxy: List[int], device: torch.device):
    """Generate mask using bounding box only."""
    inputs = processor(
        images=image,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],  # 1 = positive box
        return_tensors="pt"
    ).to(device)
    
    return inference(inputs, model, processor)


def gen_obj_mask_text(processor, model, image: Image.Image, text: str, device: torch.device):
    """Generate mask using text description only."""
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)
    
    return inference(inputs, model, processor)


def gen_obj_mask_bbox_text(processor, model, image: Image.Image, box_xyxy: List[int], text: str, device: torch.device):
    """Generate mask using both bounding box and text description."""
    inputs = processor(
        images=image,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],
        text=text,
        return_tensors="pt"
    ).to(device)
    
    return inference(inputs, model, processor)


def overlay_masks(image: Image.Image, masks: torch.Tensor):
    """Create overlay visualization of masks on the original image."""
    import matplotlib
    
    image = image.convert("RGBA")
    masks_np = (255 * masks.cpu().numpy()).astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    
    return image

def main():
    """Main function to parse arguments and generate masks."""
    import argparse
    from transformers import Sam3Processor, Sam3Model

    parser = argparse.ArgumentParser(
        description="Generate object mask using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--frame_path",
        type=str,
        required=True,
        help="Path to input RGB image"
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Text description of the object (optional)"
    )
    parser.add_argument(
        "--bbox_xywh",
        type=str,
        default=None,
        help="Bounding box in format 'x,y,w,h' (optional)"
    )
    parser.add_argument(
        "--output_mask_path",
        type=str,
        required=True,
        help="Path to save output binary mask image (PNG)"
    )
    parser.add_argument(
        "--output_overlay_path",
        type=str,
        default=None,
        help="Path to save overlay visualization (optional)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.object_name and not args.bbox_xywh:
        parser.error("Must provide at least --object_name or --bbox_xywh")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading SAM3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Model loaded successfully")

    # Load image
    image = Image.open(args.frame_path).convert("RGB")
    print(f"Loaded image: {args.frame_path} (size: {image.size})")

    # Parse bounding box if provided
    box_xyxy = None
    if args.bbox_xywh:
        try:
            x, y, w, h = map(int, args.bbox_xywh.split(','))
            box_xyxy = [x, y, x + w, y + h]  # Convert xywh to xyxy
            print(f"Using bbox (xyxy): {box_xyxy}")
        except ValueError:
            raise ValueError(f"Invalid bbox format: {args.bbox_xywh}. Expected 'x,y,w,h'")

    # Generate mask based on provided inputs
    print("\nGenerating mask...")
    if args.object_name and box_xyxy:
        print(f"  Mode: Text + Bounding Box")
        print(f"  Text: '{args.object_name}'")
        print(f"  Bbox: {box_xyxy}")
        masks = gen_obj_mask_bbox_text(processor, model, image, box_xyxy, args.object_name, device)
    elif box_xyxy:
        print(f"  Mode: Bounding Box Only")
        print(f"  Bbox: {box_xyxy}")
        masks = gen_obj_mask_bbox(processor, model, image, box_xyxy, device)
    elif args.object_name:
        print(f"  Mode: Text Only")
        print(f"  Text: '{args.object_name}'")
        masks = gen_obj_mask_text(processor, model, image, args.object_name, device)

    # Extract first mask
    if len(masks) == 0:
        print("WARNING: No masks generated!")
        return
    
    mask = masks[0].cpu().numpy()  # Take the first/best mask
    print(f"Selected mask shape: {mask.shape}")
    
    # Save binary mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(args.output_mask_path)
    print(f"\n✓ Saved mask to: {args.output_mask_path}")
    
    # Save overlay visualization if requested
    if args.output_overlay_path:
        overlay_image = overlay_masks(image, masks[:1])  # Overlay first mask only
        overlay_image.save(args.output_overlay_path)
        print(f"✓ Saved overlay to: {args.output_overlay_path}")


if __name__ == "__main__":
    main()