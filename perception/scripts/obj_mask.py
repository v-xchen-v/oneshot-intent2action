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
    
    # Optional: Visualize overlay, add:
        --output_overlay_path ./testcase/0_overlay.png
        
    # Optional: Specify mask_id if first one is not your target object, add:
        --mask_id 1
        starts from 0 for the highest scored mask.
"""

import torch
import numpy as np
from PIL import Image
from typing import List


def inference(inputs, model, processor, threshold=0.5, mask_threshold=0.5):
    """Run SAM3 inference and post-process results.
    
    Returns:
        tuple: (masks, scores) - masks sorted by scores in descending order
    """
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    # Sort masks by scores descending
    if "scores" in results:
        scores = results["scores"]
        sorted_indices = torch.argsort(scores, descending=True)
        results["masks"] = results["masks"][sorted_indices]
        results["boxes"] = results["boxes"][sorted_indices]
        results["scores"] = results["scores"][sorted_indices]
    else:
        # If no scores, create dummy scores
        scores = torch.ones(len(results["masks"]))
    
    print(f"Found {len(results['masks'])} objects with scores: {results['scores'].tolist()}")
    return results['masks'], results['scores']


def gen_obj_mask_bbox(processor, model, image: Image.Image, box_xyxy: List[int], device: torch.device):
    """Generate mask using bounding box only.
    
    Returns:
        tuple: (masks, scores)
    """
    inputs = processor(
        images=image,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],  # 1 = positive box
        return_tensors="pt"
    ).to(device)
    
    return inference(inputs, model, processor)


def gen_obj_mask_text(processor, model, image: Image.Image, text: str, device: torch.device):
    """Generate mask using text description only.
    
    Returns:
        tuple: (masks, scores)
    """
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)
    
    return inference(inputs, model, processor)


def gen_obj_mask_bbox_text(processor, model, image: Image.Image, box_xyxy: List[int], text: str, device: torch.device):
    """Generate mask using both bounding box and text description.
    
    Returns:
        tuple: (masks, scores)
    """
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
        help="Path to save output masks and overlays"
    )

    parser.add_argument(
        "--output_overlay_path",
        type=str,
        default=None,
        help="Path to save overlay visualization (optional)"
    )
    
    parser.add_argument(
        "--mask_id",
        type=int,
        default=0,
        help="Index of the mask to save if multiple masks are generated (default: 0)"
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
        masks, scores = gen_obj_mask_bbox_text(processor, model, image, box_xyxy, args.object_name, device)
    elif box_xyxy:
        print(f"  Mode: Bounding Box Only")
        print(f"  Bbox: {box_xyxy}")
        masks, scores = gen_obj_mask_bbox(processor, model, image, box_xyxy, device)
    elif args.object_name:
        print(f"  Mode: Text Only")
        print(f"  Text: '{args.object_name}'")
        masks, scores = gen_obj_mask_text(processor, model, image, args.object_name, device)

    # Filter masks with score > 0.5
    if len(masks) == 0:
        print("WARNING: No masks generated!")
        return
    
    mask_id = args.mask_id
    mask = masks[mask_id].cpu().numpy()
    score = scores[mask_id].item()
    mask_filename = args.output_mask_path
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(mask_filename)
    print(f"✓ Saved mask (id={mask_id}, score={score:.3f}) to: {mask_filename}")
    
    # Save overlay visualization if requested
    if args.output_overlay_path:
        # Overlay all valid masks
        overlay_image = overlay_masks(image, masks[mask_id:mask_id+1])
        overlay_image.save(args.output_overlay_path)
        print(f"\n✓ Saved overlay with {len(masks)} masks to: {args.output_overlay_path}")

if __name__ == "__main__":
    main()