# oneshot-intent2action

This repository uses SAM 3D Objects to generate 3D mesh files from images for intent-to-action tasks.

## Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-compatible GPU (recommended for SAM 3D Objects)

### Installation

1. **Clone the repository with submodules**

```bash
git clone --recurse-submodules https://github.com/v-xchen-v/oneshot-intent2action.git
cd oneshot-intent2action
```

If you already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

2. **Set up Python environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If you need to install SAM 3D Objects dependencies separately:

```bash
cd external/sam-3d-objects
pip install -e .
cd ../..
```

### Purpose of SAM 3D Objects

This repository integrates [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) as a submodule to:

- **Generate 3D mesh files (.stl format)** from 2D images
- Provide 3D object understanding for intent-to-action workflows
- Enable spatial reasoning and manipulation planning

The SAM 3D Objects model takes input images and produces high-quality 3D mesh representations in .stl file format, which is required by FoundationPose for 6D object pose estimation. These mesh files can be used for downstream robotic tasks, scene understanding, and action planning.

## Usage

### Using SAM 3D Objects to Generate Meshes

**Quick Setup (Minimal Dependencies)**

The SAM3D environment is easy to prepare with minimal dependencies:

```bash
# Create a Python 3.10+ environment
conda create -n sam3d python=3.10
conda activate sam3d

# Install SAM3D Objects
cd external/sam-3d-objects
pip install -e .

# Install minimal required dependencies
pip install transformers==5.0.0rc0
pip install torch torchvision
pip install matplotlib
```

**Using SAM3D Scripts**

After installation, you can use the SAM3D scripts to generate 3D mesh files:

```bash
conda activate sam3d
cd external/sam-3d-objects

# Use the provided scripts from the notebook directory
python notebook/inference.py --image /path/to/image.jpg --mask /path/to/mask.png --output model.ply
```

The generated mesh files can then be used in your intent-to-action pipeline for 3D object understanding and manipulation planning.

For detailed setup and usage, refer to the [official SAM 3D Objects setup guide](https://github.com/facebookresearch/sam-3d-objects).

## Project Structure

The repository is organized into three main categories for working with external libraries:

```
oneshot-intent2action/
├── perception/
│   ├── examples/        # Simple standalone playground scripts
│   │                    # - No command-line arguments
│   │                    # - Quick experimentation with libraries
│   │                    # - Self-contained test scripts
│   ├── scripts/         # Complex pipeline scripts with arguments
│   │                    # - Production-ready with argparse/CLI
│   │                    # - Part of the main pipeline
│   │                    # - Configurable via command-line
│   ├── modules/         # Reusable modules and utilities
│   │                    # - Shared code for scripts
│   │                    # - Library wrappers and helpers
│   └── tests/           # Test scripts for validation
│                        # - Test individual scripts and modules
│                        # - Verify functionality with sample data
├── external/
│   └── sam-3d-objects/  # SAM 3D Objects submodule for 3D mesh generation
├── README.md
└── requirements.txt
```

**Folder Design Philosophy:**

- **`perception/examples/`**: Start here when exploring new libraries. Write simple, hardcoded scripts to understand how things work. No arguments needed - just run and see results.

- **`perception/scripts/`**: Once you understand the library, create production scripts here. These accept arguments, handle errors properly, and integrate into the pipeline.

- **`perception/modules/`**: Extract common functionality into reusable modules that both examples and scripts can import.

- **`perception/tests/`**: Write test scripts to validate your scripts and modules work correctly with sample data.

## Updating Submodules

To update the SAM 3D Objects submodule to the latest version:

```bash
git submodule update --remote external/sam-3d-objects
```

## Contributing

When working with this repository, remember to commit submodule updates if you change them:

```bash
git add external/sam-3d-objects
git commit -m "Update SAM 3D Objects submodule"
```

## License

Please refer to the individual licenses of dependencies, particularly SAM 3D Objects.
