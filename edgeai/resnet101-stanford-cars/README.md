# ResNet101 Stanford Cars Classification - Zededa EdgeAI Example

This example demonstrates how to upload and deploy a ResNet101 model for Stanford Cars classification using Zededa EdgeAI platform. The notebook provides a complete workflow from model analysis to MLflow deployment.

## Overview

- **Model**: ResNet101 (Residual Neural Network with 101 layers)
- **Dataset**: Stanford Cars (196 fine-grained car classes)
- **Task**: Fine-grained car classification
- **Format**: ONNX (optimized for deployment)
- **Platform**: Zededa EdgeAI with MLflow integration

## Model Performance

- **Accuracy**: 91.2%
- **Precision**: 90.8%
- **Recall**: 91.2%
- **F1-Score**: 90.9%
- **Inference Time**: ~55ms
- **Model Size**: 166.39 MB
- **Parameters**: 43.6M parameters

## Files in this Example

### Core Files
- `resnet101-model-upload.ipynb` - Main Jupyter notebook with complete workflow
- `class_names.json` - Stanford Cars dataset class names (196 classes)
- `model.onnx` - ResNet101 ONNX model file (166MB)

### Generated Configuration Files
- `config.pbtxt` - Triton deployment configuration
- `requirements.txt` - Python dependencies

## Prerequisites

### 1. Zededa EdgeAI Account
- Access to Zededa EdgeAI Studio
- Valid credentials for authentication

### 2. Python Environment
- Python 3.8+
- Jupyter Notebook environment
- Required packages (installed automatically in notebook):
  - `mlflow` - Model tracking and deployment
  - `onnx` - ONNX model format support
  - `torch` - PyTorch for model analysis
  - `onnx2torch` - ONNX to PyTorch conversion
  - `torchinfo` - Model architecture analysis
  - `thop` - FLOPs calculation
  - `scikit-learn` - Evaluation metrics
  - `onnxruntime` - Model inference

## Quick Start

### Step 1: Setup Environment
```bash
# Create virtual environment
python -m venv resnet101-env
source resnet101-env/bin/activate  # On Windows: resnet101-env\Scripts\activate

# Install Jupyter
pip install jupyter
```

### Step 2: Authentication
Before running the notebook, authenticate with Zededa EdgeAI:

**Option 1: Terminal Authentication (Recommended)**
```bash
export EDGEAI_SERVICE_URL=https://studio.edgeai.zededa.dev
zededa-edgeai login --email your-email@company.com --prompt-password --catalog zededa
```

**Option 2: Environment Variables**
Set environment variables in the notebook or terminal:
```bash
export MLFLOW_TRACKING_URI=https://studio.edgeai.zededa.dev
export MLFLOW_TRACKING_TOKEN=your-token
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export MLFLOW_S3_ENDPOINT_URL=https://minio.edgeai.zededa.dev
export MINIO_BUCKET=mlflow-zededa
```

### Step 3: Run the Notebook
```bash
jupyter notebook resnet101-model-upload.ipynb
```

## Notebook Workflow

The notebook is organized into the following sections:

### 1. Environment Setup & Authentication
- Install Zededa EdgeAI SDK
- Authenticate with the platform
- Verify environment variables

### 2. Package Installation
- Install all required Python packages
- Setup MLflow connection

### 3. Model Loading & Analysis
- Load ONNX model
- Convert to PyTorch for detailed analysis
- Extract comprehensive model metrics:
  - Parameter counts (43.6M parameters)
  - FLOPs calculation (7.8 GFLOPs)
  - Layer analysis (245 layers)
  - Memory usage
  - Input/output shapes

### 4. Dataset Integration
- Load Stanford Cars class names
- Configure for 196 car classes
- Setup evaluation framework

### 5. Performance Evaluation
- Test inference speed (~55ms)
- Performance metrics calculation
- Model validation

### 6. Configuration Generation
- Generate Triton deployment config
- Create requirements.txt
- Generate model documentation

### 7. MLflow Upload
- Create MLflow experiment
- Log comprehensive model metadata
- Upload model artifacts
- Register model in MLflow registry

### 8. Production Deployment
- Transition model to production stage
- Comprehensive verification
- Deployment readiness check

## Advanced Features

### Automatic Model Analysis
The notebook automatically extracts detailed model information using PyTorch conversion:
- **Architecture Analysis**: Layer types, counts, and structure
- **Computational Metrics**: FLOPs, MACs, and complexity analysis
- **Memory Profiling**: Parameter and buffer memory usage
- **Performance Profiling**: Inference speed and optimization metrics

### Dynamic Configuration
- Automatic generation of deployment configurations
- Model-specific parameter extraction
- Flexible input/output shape handling

### Comprehensive Logging
All model metadata is automatically logged to MLflow:
- Model architecture details
- Performance metrics
- Training parameters
- Deployment configurations

## Stanford Cars Dataset

The model classifies cars into 196 fine-grained categories:

### Categories Include:
- **Manufacturers**: BMW, Mercedes-Benz, Toyota, Honda, Ford, etc.
- **Model Years**: 1991-2012
- **Body Types**: Sedan, Coupe, SUV, Convertible, Hatchback, Wagon
- **Luxury Levels**: Economy, Mid-range, Luxury, Sports cars

### Sample Classes:
- 2012 Bentley Continental Supersports Conv. Convertible
- 2007 BMW X5 SUV
- 2012 Bugatti Veyron 16.4 Convertible
- 2012 Ferrari FF Coupe
- 2007 Honda Accord Sedan
- 2011 Mercedes-Benz SL-Class Convertible

See `class_names.json` for the complete list of 196 classes.

## Model Architecture: ResNet101

### Key Features:
- **Deep Residual Network**: 101 layers with skip connections
- **Residual Blocks**: Solves vanishing gradient problem
- **Batch Normalization**: Improves training stability
- **Global Average Pooling**: Reduces overfitting
- **Fine-tuned**: Optimized for Stanford Cars dataset

### Technical Specifications:
- **Input**: 224×224 RGB images (NCHW format)
- **Output**: 196-dimensional probability vector
- **Parameters**: 43,601,220 total parameters
- **FLOPs**: 7.8 GFLOPs
- **Layers**: 245 total layers
- **Memory**: ~174 MB (model + buffers)

## Deployment Configuration

### Triton Inference Server
The generated `config.pbtxt` provides configuration for Triton deployment:
- Platform: ONNX Runtime
- Dynamic batching support
- CPU inference optimization
- Flexible input shapes

### Performance Optimization
- ONNX format for cross-platform deployment
- Optimized for CPU inference
- Dynamic batch size support
- Efficient memory usage

## Troubleshooting

### Common Issues

**1. Authentication Errors**
```bash
# Solution: Re-authenticate
zededa-edgeai login --email your-email --prompt-password --catalog zededa
```

**2. Model Loading Errors**
- Verify ONNX model file exists and is valid
- Check file path in `MODEL_PATH` variable
- Ensure model is compatible with ONNX Runtime

**3. Package Installation Issues**
- Use virtual environment
- Update pip: `pip install --upgrade pip`
- Install packages individually if batch install fails

**4. Memory Issues**
- Reduce batch size in evaluation
- Close other applications
- Use smaller model if available

### Performance Tips

**1. Speed up inference:**
- Use GPU if available (modify ONNX Runtime provider)
- Optimize model with ONNX tools
- Use TensorRT for NVIDIA GPUs

**2. Reduce memory usage:**
- Process images in smaller batches
- Use mixed precision if supported
- Clear variables between cells

## Support

For issues related to:
- **Zededa EdgeAI Platform**: Contact Zededa support
- **Model Performance**: Check model preprocessing and input format
- **MLflow Integration**: Verify authentication and network connectivity
- **ONNX Runtime**: Check model compatibility and runtime version

## License

This example is provided as part of the Zededa examples repository. Please refer to the main repository license for usage terms.

## Contributing

Contributions to improve this example are welcome! Please follow the repository's contribution guidelines when submitting improvements or bug fixes.