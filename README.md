# BDD100K Object Detection Analysis \& Training

This repository contains tools for analyzing the BDD100K dataset and training RT-DETRv2 models for vehicle detection tasks.

## Common Issues Identified in BDD Dataset

### 1. **Class Distribution Problems**

- **Severe class imbalance**:
    - `car` class: ~45% of all annotations
    - `person` class: ~25% of annotations
    - Rare classes like `train`, `bus`: <2% each
- **Small object bias**: Many objects have very small bounding boxes (<50px²)
- **Geographic bias**: Dataset heavily skewed toward certain weather/lighting conditions


### 2. **Annotation Quality Issues**

- **Invalid bounding boxes**: Some boxes extend beyond image boundaries
- **Tiny objects**: Boxes smaller than 5x5 pixels that may be annotation noise
- **Inconsistent labeling**: Similar objects labeled differently across images

### 3. **Training Configuration Challenges**

- **Model config complexity**: RT-DETRv2 requires specific YAML structure with includes
- **Memory limitations**: Large batch sizes needed for stable training
- **Convergence issues**: Learning rate scheduling critical for 10-class detection


## Docker Setup

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional packages for analysis
RUN pip install streamlit matplotlib seaborn plotly opencv-python-headless

# Clone RT-DETRv2 repository
RUN git clone https://github.com/lyuwenyu/RT-DETR.git rtdetrv2_pytorch
WORKDIR /app/rtdetrv2_pytorch

# Install RT-DETR requirements
RUN pip install -r requirements.txt

# Set up environment
ENV PYTHONPATH=/app/rtdetrv2_pytorch:$PYTHONPATH

# Expose ports for Streamlit dashboard
EXPOSE 8501

# Default command
CMD ["bash"]
```


### Requirements.txt

```txt
torch>=1.13.0
torchvision>=0.14.0
opencv-python>=4.6.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.20.0
pandas>=1.4.0
pycocotools>=2.0.4
albumentations>=1.2.0
pyyaml>=6.0
tqdm>=4.64.0
pillow>=9.0.0
scipy>=1.8.0
```


## Building and Running the Container

### 1. Build the Docker Image

```bash
# Build the image
docker build -t bdd-analysis .

# Or build with specific tag
docker build -t bdd-analysis:v1.0 .
```


### 2. Run the Container

```bash
# Run with GPU support and volume mounts
docker run --gpus all -it \
  -v $(pwd)/vehicle_data:/app/vehicle_data \
  -v $(pwd)/output:/app/output \
  -p 8501:8501 \
  bdd-analysis

# Run for interactive development
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -p 8501:8501 \
  bdd-analysis bash
```


### 3. Alternative: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  bdd-analysis:
    build: .
    volumes:
      - ./vehicle_data:/app/vehicle_data
      - ./output:/app/output
      - ./configs:/app/rtdetrv2_pytorch/configs/dataset
    ports:
      - "8501:8501"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:

```bash
docker-compose up -d
```


## Usage Instructions

### 1. Dataset Analysis

```bash
# Inside container, run dataset analysis
cd /app
python dataset_analysis.py

# Or run Streamlit dashboard
streamlit run analysis_dashboard.py --server.port 8501 --server.address 0.0.0.0
```


### 2. Data Preprocessing

```bash
# Convert BDD labels to COCO format
python convert_bdd_to_coco.py \
  --input /app/vehicle_data/labels/bdd100k_labels_images_train.json \
  --output /app/vehicle_data/labels/bdd100k_labels_images_train_coco.json

python convert_bdd_to_coco.py \
  --input /app/vehicle_data/labels/bdd100k_labels_images_val.json \
  --output /app/vehicle_data/labels/bdd100k_labels_images_val_coco.json
```


### 3. Training RT-DETRv2

```bash
# Single GPU training
python tools/train.py -c configs/dataset/bdd_vehicle.yml

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=9909 \
  --nproc_per_node=4 \
  -- python3 tools/train.py \
  -c configs/dataset/bdd_vehicle.yml \
  --use-amp --seed=0
```


### 4. Evaluation

```bash
# Evaluate trained model
python tools/eval.py \
  -c configs/dataset/bdd_vehicle.yml \
  --resume output/rtdetrv2_r50vd_m_bdd10/checkpoint_best.pth
```


## Directory Structure

```
project/
├── vehicle_data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── bdd100k_labels_images_train.json
│       ├── bdd100k_labels_images_val.json
│       ├── bdd100k_labels_images_train_coco.json
│       └── bdd100k_labels_images_val_coco.json
├── configs/
│   └── dataset/
│       └── bdd_vehicle.yml
├── output/
├── analysis_results/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


## Performance Optimization Tips

1. **Use mixed precision training** (`--use-amp`) to reduce memory usage
2. **Adjust batch size** based on available GPU memory
3. **Enable gradient accumulation** if batch size is limited
4. **Use multiple workers** for data loading (`num_workers: 4-8`)
5. **Cache dataset in RAM** if sufficient memory available

## Troubleshooting

### Common Issues:

- **CUDA out of memory**: Reduce batch size or enable gradient accumulation
- **Path not found errors**: Verify volume mounts and file paths
- **Model is None error**: Check config file includes and model definition
- **Slow training**: Increase `num_workers`, use `--cache-ram`, or optimize transforms


### Debug Commands:

```bash
# Check GPU availability
nvidia-smi

# Verify dataset paths
ls -la /app/vehicle_data/images/train/

# Test config loading
python -c "import yaml; print(yaml.safe_load(open('configs/dataset/bdd_vehicle.yml')))"
```


## Results \& Analysis

Generated analysis plots will be saved as:

- `class_distribution_train.png` / `class_distribution_val.png`
- `size_distribution_train.png` / `size_distribution_val.png`
- `properties_train.png` / `properties_val.png`
- `difficult_cases_train.png` / `difficult_cases_val.png`

Training outputs saved to:

- `output/rtdetrv2_r50vd_m_bdd10/`
    - Model checkpoints
    - Training logs
    - Evaluation metrics
