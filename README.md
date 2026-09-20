# RGB → Binary: 4-Parameter Neural Image Transform

> A deliberately minimal neural model for converting RGB pixels into binary black/white output — built around **4 trainable parameters, a 16-byte FP32 parameter footprint, and a simple pixel-wise inference path**.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](#requirements)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#requirements)
[![Model](https://img.shields.io/badge/Trainable%20Parameters-4-success.svg)](#model-design)
[![Weights](https://img.shields.io/badge/FP32%20Weights-16%20bytes-informational.svg)](#model-footprint)

---

## 1. Project objective

Binary image conversion is often implemented with a direct thresholding operation.

This project investigates whether the same type of RGB-to-binary transformation can be represented as an **extremely small trainable neural model**.

The current architecture contains only:

- 3 RGB weights
- 1 bias

That gives:

**4 trainable parameters**

and, using FP32 storage:

**4 × 4 bytes = 16 bytes**

The model therefore provides an unusually small learned transformation while retaining a conventional neural-network training workflow.

---

## 2. Core idea

The training pipeline first derives a binary target from RGB pixels.

### Target generation

The notebook computes luminance using:

```text
Y = 0.299R + 0.587G + 0.114B
```

and applies:

```text
Binary = 1 if Y > 127
Binary = 0 otherwise
```

The RGB input is normalized to `[0, 1]`.

The neural model then learns to predict the probability of the white class:

```text
                 ┌────────────────┐
R ──────────────►│                │
G ──────────────►│    Dense(1)    ├──► P(white)
B ──────────────►│    + Sigmoid   │
                 └────────────────┘
                         │
                         ▼
                    Threshold
                       0.5
                         │
                         ▼
                   0 / 255 pixel
```

---

## 3. Model design

### Architecture

| Component | Configuration |
|---|---|
| Input | 3 normalized RGB values |
| Layer | `Dense(1)` |
| Activation | Sigmoid |
| Output | White-class probability |
| Binary decision threshold | 0.5 |
| Trainable parameters | **4** |
| FP32 parameter storage | **16 bytes** |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch size | 256 |

### Parameter calculation

The single dense neuron contains:

```text
RGB weights = 3
Bias         = 1
----------------
Total        = 4
```

With FP32 parameters:

```text
4 × 4 bytes = 16 bytes
```

> **Important:** 16 bytes describes the raw FP32 trainable parameters. It does not represent the complete serialized Keras/TensorFlow model file, which includes framework metadata and other overhead.

---

## 4. Training data

The current notebook builds its training set from images in:

```text
Normal_Images/
```

The recorded run found:

```text
226 images
```

Each image is resized to:

```text
224 × 224
```

The RGB pixels are flattened into individual samples.

The recorded training matrix was:

```text
Input shape  : (11,339,776, 3)
Target shape : (11,339,776, 1)
```

This corresponds to the pixel-level formulation of the task rather than treating each image as one training sample.

---

## 5. Training result

The current notebook run reports:

```text
Epoch 10
Training accuracy   ≈ 99.81%
Validation accuracy ≈ 99.80%
Validation loss     ≈ 0.0253
```

These values are from the current notebook execution and should be treated as **experiment-specific measurements**.

For future comparisons, benchmark datasets, splits, random seeds, hardware, TensorFlow versions, and preprocessing should be recorded alongside the result.

---

## 6. Image conversion pipeline

The inference workflow is:

```text
RGB Image
    │
    ▼
Resize to 224 × 224
    │
    ▼
Extract RGB pixels
    │
    ▼
Normalize RGB values
    │
    ▼
4-parameter neural transform
    │
    ▼
White probability
    │
    ▼
Threshold at 0.5
    │
    ▼
0 / 255
    │
    ▼
Binary image
```

The current implementation supports:

```text
.jpg
.jpeg
.png
```

---

## 7. Minimal inference interface

The notebook provides:

```python
convert_to_binary(model, image_path, output_path)
```

Example:

```python
original_image, binary_image = convert_to_binary(
    model,
    "input.jpg",
    "binary.jpg"
)
```

The output is represented as:

```text
0   → black
255 → white
```

---

## 8. Why the architecture is interesting

The goal is not to claim that a four-parameter model is universally superior to conventional thresholding.

The engineering value is in demonstrating how far model reduction can go when the transformation has a simple structure.

The design emphasizes:

- extremely low parameter count
- tiny raw parameter memory
- simple computation
- interpretable input/output relationship
- pixel-wise parallelism
- compatibility with low-resource deployment concepts

This creates a useful experimental platform for studying **tiny learned image transforms**.

---

## 9. Repository architecture

For a serious research/engineering repository, we recommend separating experimentation from reusable implementation:

```text
rgb-to-binary/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   └── binary_training.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── models/
│   └── binary_model/
│
├── examples/
│   ├── input/
│   └── output/
│
├── benchmarks/
│   └── benchmark.py
│
├── tests/
│   └── test_inference.py
│
└── docs/
    └── methodology.md
```

This structure makes it easier for another researcher or engineer to understand:

```text
Experiment → Model → Inference → Benchmark → Reproduction
```

rather than treating the notebook as the entire project.

---

## 10. Recommended benchmark protocol

Because the model is extremely small, parameter count alone is not enough to establish runtime efficiency.

We recommend measuring:

| Metric | Why it matters |
|---|---|
| Trainable parameters | Architectural complexity |
| Raw parameter bytes | Weight memory |
| Serialized model size | Deployment footprint |
| Training time | Development cost |
| Peak RAM | Resource requirements |
| Single-image latency | User-facing speed |
| Pixel throughput | Core computational efficiency |
| CPU utilization | Runtime overhead |
| Energy/image | Edge deployment relevance |
| Binary disagreement rate | Output fidelity |

For this project, **direct thresholding should be included as a baseline** because it represents the conventional algorithmic solution to the same transformation.

---

## 11. Important distinction: learned transform vs. thresholding

The reference implementation uses a known luminance equation to construct the training labels:

```text
RGB → luminance → threshold → binary target
```

The neural model learns an approximation of that mapping:

```text
RGB → learned 4-parameter transform → probability → binary output
```

Therefore, the scientifically interesting comparison is not simply:

```text
"Does the model work?"
```

but:

```text
How much computational / deployment overhead does
the learned representation introduce or remove compared
with the direct mathematical baseline?
```

That comparison should be part of future benchmarking.

---

## 12. Deployment directions

Because the model contains only four parameters, it is a useful candidate for experiments involving:

- embedded vision
- tinyML
- edge preprocessing
- low-memory systems
- CPU-only inference
- SIMD/vectorized implementations
- C/C++ inference
- microcontrollers
- FPGA/ASIC-oriented exploration
- sensor-side preprocessing

Future work can investigate whether the model can be reduced beyond the framework itself.

For example:

```text
TensorFlow model
      ↓
Export parameters
      ↓
4 numerical constants
      ↓
Native implementation
      ↓
No ML framework at inference
```

That direction could make the distinction between **model size** and **runtime stack size** especially important.

---

## 13. Research roadmap

### Phase 1 — Reproducibility
- [x] 4-parameter model
- [x] Pixel-level training
- [x] Binary image conversion
- [x] Training/validation evaluation

### Phase 2 — Engineering benchmarks
- [ ] CPU inference latency
- [ ] Batch throughput
- [ ] Peak memory
- [ ] Serialized model size
- [ ] Direct-threshold baseline
- [ ] Multiple image resolutions

### Phase 3 — Hardware efficiency
- [ ] NumPy reference implementation
- [ ] C/C++ implementation
- [ ] INT8 representation
- [ ] SIMD implementation
- [ ] Microcontroller experiment
- [ ] Energy-per-image measurement

### Phase 4 — Research
- [ ] Cross-dataset evaluation
- [ ] Robustness analysis
- [ ] Learned vs. analytical transforms
- [ ] Hardware-aware parameterization
- [ ] Generalization to other preprocessing operations

---

## 14. Collaboration

This repository is intended to be more than a demonstration notebook.

We are interested in collaborators working on:

- tinyML
- efficient AI
- edge computing
- model compression
- hardware-aware machine learning
- image processing
- embedded vision
- neural representations of classical algorithms

If you work on **making machine learning smaller, cheaper, and closer to the data source**, this project provides a compact experimental foundation for collaboration.

### Contribution principle

We prioritize contributions that improve:

```text
Reproducibility
      ↓
Measurement
      ↓
Efficiency
      ↓
Portability
      ↓
Scientific understanding
```

Please open an issue before major architectural changes so benchmark comparisons remain meaningful.

---

## 15. Requirements

```bash
pip install numpy tensorflow pillow matplotlib
```

Open:

```text
notebooks/binary_training.ipynb
```

and execute the notebook sequentially.

---

## 16. Current status

**Research / engineering prototype**

The current implementation demonstrates a 4-parameter RGB-to-binary neural transformation and reports strong validation accuracy on the notebook's generated dataset.

Runtime latency, energy consumption, and cross-hardware performance are **not yet benchmarked in the current notebook** and should be measured before making deployment-level efficiency claims.

---

## License

Add your chosen open-source license here before publishing the repository.
