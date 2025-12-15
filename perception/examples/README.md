# Perception Examples

Simple standalone scripts for testing and playground purposes.

These scripts are meant to be run directly without arguments for quick experimentation with external libraries.

## SAM3 Segmentation Features

SAM3 (Segment Anything Model 3) supports three flexible prompting modes for object segmentation:

1. **Text-only prompts**: Segment objects using natural language descriptions
   - Example: `"a red car"`, `"person wearing blue shirt"`
   - Best for: Semantic understanding, category-based segmentation

2. **Bounding box-only prompts**: Segment objects within specified rectangular regions
   - Example: `[[x1, y1, x2, y2]]` in pixel coordinates
   - Best for: Spatial precision, known object locations

3. **Combined text + bbox prompts**: Combine both text and spatial hints for precise results
   - Example: Text `"laptop"` + bbox `[[100, 150, 400, 350]]`
   - Best for: Disambiguating multiple objects, refining segmentation accuracy

This multimodal prompting capability makes SAM3 highly versatile for various perception tasks.

## Examples

- `sam3_simple.py` - Simple SAM3 segmentation example
- Add more standalone examples here...
