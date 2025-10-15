# ConvNeXt Base Stanford Cars Model

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Format](https://img.shields.io/badge/Format-ONNX-green.svg)](https://onnx.ai/)

## Model Overview

**ConvNeXt Base Stanford Cars** is a state-of-the-art computer vision model for fine-grained vehicle classification, based on the ConvNeXt architecture. This model can accurately classify vehicles into 196 distinct car makes, models, and years from the Stanford Cars dataset.

### Key Specifications
- **Model Name**: `convnext_base_cars`
- **Architecture**: ConvNeXt Base (A ConvNet for the 2020s)
- **Task**: Fine-grained Image Classification
- **Dataset**: Stanford Cars Dataset
- **Classes**: 196 vehicle categories
- **Format**: ONNX (optimized for deployment)
- **Input Resolution**: 224×224 pixels

### Architecture Details

ConvNeXt represents a modernized CNN architecture that incorporates design choices from Vision Transformers while maintaining the efficiency of convolutional networks:

- **Backbone**: ConvNeXt-Base architecture
- **Stem**: 4×4 convolution with stride 4 (patchify)
- **Stages**: 4 feature resolution stages with depths [3, 3, 27, 3]
- **Blocks**: Inverted bottleneck blocks with depthwise convolutions
- **Normalization**: LayerNorm for improved training stability
- **Activation**: GELU activation function

## Quick Start

### Prerequisites
```bash
pip install onnxruntime numpy pillow
# For GPU support
pip install onnxruntime-gpu
```

### Basic Inference
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("model.onnx")

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0,1]
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Run inference
input_tensor = preprocess_image("car_image.jpg")
outputs = session.run(None, {"images": input_tensor})
prediction = np.argmax(outputs[0][0])
print(f"Predicted class: {prediction}")
```

## Triton Inference Server Deployment

This model is optimized for production deployment using NVIDIA Triton Inference Server.

### Configuration Features
- **Multi-Backend Support**: ONNX Runtime with TensorRT optimization
- **Dynamic Batching**: Automatic batching for improved throughput
- **Multi-Instance**: CPU and GPU instance groups
- **Model Warmup**: Pre-compiled inference paths

### Deployment Steps

1. **Setup Triton Model Repository**
```bash
# Create directory structure
mkdir -p model_repository/convnext_base_cars/1

# Copy model files
cp model.onnx model_repository/convnext_base_cars/1/
cp config.pbtxt model_repository/convnext_base_cars/
```

2. **Start Triton Server**
```bash
docker run --gpus=all -it --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v$(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

3. **Client Example**
```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient("localhost:8000")

# Prepare batch input
batch_data = np.random.random((4, 3, 224, 224)).astype(np.float32)
inputs = [httpclient.InferInput("images", batch_data.shape, "FP32")]
inputs[0].set_data_from_numpy(batch_data)

outputs = [httpclient.InferRequestedOutput("predictions")]
results = client.infer("convnext_base_cars", inputs, outputs=outputs)

predictions = results.as_numpy("predictions")
print(f"Batch predictions shape: {predictions.shape}")
```

## Stanford Cars Dataset

The model is trained on the Stanford Cars Dataset, which contains:
- **196 classes** of cars (make, model, year combinations)
- Examples: "2012 Tesla Model S Sedan", "2010 BMW X5 SUV"
- **Training set**: ~8,144 images
- **Test set**: ~8,041 images
- **Resolution**: Variable, preprocessed to 224×224

### Sample Classes
```
0: AM General Hummer SUV 2000
1: Acura RL Sedan 2012
2: Acura TL Sedan 2012
3: Acura TL Type-S 2008
4: Acura TSX Sedan 2012
...
195: Volvo XC90 SUV 2007
```

## Model Optimization

### Available Optimizations
- **ONNX Format**: Optimized computational graph
- **TensorRT Integration**: GPU acceleration on NVIDIA hardware
- **Dynamic Batching**: Improved throughput for multiple requests
- **Memory Optimization**: Efficient memory usage patterns

### Production Configuration
```yaml
Batch Sizes: [1, 2, 4, 8, 16, 32]
Max Queue Delay: 100μs
CPU Cores: 4+ recommended
RAM: 8GB+ recommended
```

## Use Cases

### Primary Applications
- **Automotive Industry**: Vehicle inventory management
- **Insurance**: Automated damage assessment
- **Car Dealerships**: Automated classification
- **Traffic Analysis**: Vehicle type recognition
- **Parking Systems**: Smart parking solutions

### Integration Examples
- **REST APIs**: Direct inference endpoints
- **Mobile Apps**: Edge deployment with ONNX Runtime
- **Cloud Services**: Scalable batch processing
- **Edge Devices**: Optimized inference on embedded systems

## Model Limitations

### Known Limitations
- **Dataset Bias**: Limited to cars from 2000-2012
- **Image Quality**: Dependent on input image quality
- **Viewpoint**: Optimal for side and front-angled views
- **Occlusion**: May have difficulty with heavily occluded vehicles

### Recommendations
- Input images should be well-lit and clearly visible
- Minimum resolution of 224×224 recommended
- Consider ensemble methods for critical applications
- Regular retraining for newer vehicle models

## Citation

If you use this model in your research, please cite:

```bibtex
@article{liu2022convnet,
  title={A ConvNet for the 2020s},
  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@inproceedings{KrauseStarkDengFei-Fei_3DRR2013,
  title = {3D Object Representations for Fine-Grained Categorization},
  booktitle = {4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)},
  year = {2013},
  author = {Jonathan Krause and Michael Stark and Jia Deng and Li Fei-Fei}
}
```

## License

This model is released under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.

## Support

For questions, issues, or contributions, please contact the EdgeAI team or open an issue in the repository.

---

**Model Version**: v1.0  
**Last Updated**: October 2025  
**Maintained by**: EdgeAI Team
