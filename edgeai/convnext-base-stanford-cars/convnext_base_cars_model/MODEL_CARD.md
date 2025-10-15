# Model Card: ConvNeXt Base Stanford Cars

## Model Details

### Model Description
ConvNeXt Base Stanford Cars is a computer vision model for fine-grained vehicle classification, based on the ConvNeXt architecture. This model can accurately classify vehicles into 196 distinct categories from the Stanford Cars dataset.

- **Developed by:** EdgeAI Team
- **Model type:** Image Classification
- **Language(s):** Not applicable (computer vision)
- **License:** AGPL-3.0
- **Parent Model:** ConvNeXt Base
- **Resources for more information:** 
  - [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
  - [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### Model Sources
- **Repository:** https://github.com/facebookresearch/ConvNeXt
- **Paper:** "A ConvNet for the 2020s" - Liu et al., CVPR 2022
- **Demo:** Available through Triton Inference Server deployment

## Uses

### Direct Use
This model can be used directly for:
- Vehicle make/model/year identification
- Automotive inventory management
- Insurance damage assessment systems
- Traffic monitoring and analysis
- Smart parking applications

### Downstream Use
The model can be fine-tuned for:
- Other vehicle classification tasks
- Transfer learning for automotive domains
- Feature extraction for vehicle-related applications

### Out-of-Scope Use
This model should not be used for:
- Classification of vehicles outside the 2000-2012 timeframe
- Real-time critical safety applications without additional validation
- Identifying specific vehicle instances or license plates
- Privacy-sensitive surveillance applications

## Bias, Risks, and Limitations

### Known Limitations
- **Temporal Bias:** Training data limited to vehicles from 2000-2012
- **Geographic Bias:** Dataset primarily contains North American vehicle models
- **Image Quality Dependency:** Performance degrades on low-resolution or heavily occluded images
- **Viewpoint Sensitivity:** Optimal performance on side and front-angled vehicle views

### Recommendations
Users should be aware of the model's limitations and consider:
- Regular retraining with updated vehicle models
- Ensemble methods for critical applications
- Additional validation for deployment in non-US markets
- Quality checks for input images

## How to Get Started with the Model

### Quick Start
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
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)

# Run inference
input_tensor = preprocess_image("car_image.jpg")
outputs = session.run(None, {"images": input_tensor})
prediction = np.argmax(outputs[0][0])
```

## Training Details

### Training Data
- **Dataset:** Stanford Cars Dataset
- **Training Set:** ~8,144 images
- **Test Set:** ~8,041 images
- **Classes:** 196 vehicle categories (make/model/year combinations)
- **Resolution:** Variable, preprocessed to 224×224

### Training Procedure
- **Preprocessing:** Standard ImageNet preprocessing with data augmentation
- **Architecture:** ConvNeXt Base backbone with classification head
- **Optimization:** AdamW optimizer with cosine learning rate schedule
- **Framework:** PyTorch training, converted to ONNX for deployment

#### Training Hyperparameters
- **Batch Size:** 32
- **Learning Rate:** 1e-4 (with cosine decay)
- **Epochs:** 100 (with early stopping)
- **Optimizer:** AdamW
- **Weight Decay:** 0.01

#### Speeds, Sizes, Times
- **Training Time:** ~8 hours on NVIDIA A100
- **Model Size:** 336 MB (ONNX format)
- **Inference Speed:** 153.95ms per image (average)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- **Dataset:** Stanford Cars test set (8,041 images)
- **Coverage:** All 196 vehicle categories
- **Evaluation Protocol:** Standard classification accuracy

#### Metrics
- **Top-1 Accuracy:** 92.82%
- **Precision (weighted):** 93.18%
- **Recall (weighted):** 92.82%
- **F1-Score (weighted):** 92.77%

### Results
The model achieves excellent performance on the Stanford Cars dataset:

| Metric | Value |
|--------|-------|
| Accuracy | 92.82% |
| Precision | 93.18% |
| Recall | 92.82% |
| F1-Score | 92.77% |
| Inference Time | 153.95ms |

#### Summary
ConvNeXt Base Stanford Cars demonstrates excellent performance for fine-grained vehicle classification, achieving over 92% accuracy on the full Stanford Cars test set with 8,041 images across 196 vehicle categories.

## Model Examination

### Architecture Analysis
- **Total Parameters:** ~88 million
- **Model Depth:** 4 stages with depths [3, 3, 27, 3]
- **Computational Complexity:** ~15.4 GFLOPs
- **Memory Usage:** ~336 MB model weights

### Interpretability
The model uses standard convolutional features that can be analyzed through:
- Gradient-based attribution methods (GradCAM)
- Feature visualization techniques
- Attention map analysis

## Environmental Impact

### Carbon Footprint
- **Training:** Estimated 15 kWh on A100 GPU (~6 kg CO2 equivalent)
- **Inference:** Low energy consumption suitable for edge deployment

### Sustainability Considerations
- ONNX format enables efficient deployment across hardware platforms
- Optimized for edge computing to reduce cloud computing requirements
- Model compression techniques applied to minimize resource usage

## Technical Specifications

### Model Architecture
```
ConvNeXt Base:
├── Stem: 4×4 conv, stride 4
├── Stage 1: 3 blocks, dim=128
├── Stage 2: 3 blocks, dim=256  
├── Stage 3: 27 blocks, dim=512
├── Stage 4: 3 blocks, dim=1024
└── Head: Global avg pool + Linear(1024→196)
```

### Input/Output Specifications
- **Input Shape:** [batch_size, 3, 224, 224]
- **Input Type:** Float32, normalized [0,1]
- **Output Shape:** [batch_size, 196]
- **Output Type:** Float32, logits (apply softmax for probabilities)

### Deployment Requirements
- **Minimum RAM:** 2GB for inference
- **Recommended GPU:** 4GB+ VRAM for batch processing
- **CPU Requirements:** 4+ cores for optimal performance
- **Storage:** 400MB for model and dependencies

## Citation

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

## Model Card Authors
EdgeAI Team

## Model Card Contact
For questions about this model card, contact the EdgeAI team or open an issue in the repository.

---
*This model card was generated following the framework proposed in "Model Cards for Model Reporting" (Mitchell et al., 2019).*