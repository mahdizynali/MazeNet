# MazeNet

A simple **neural network from scratch in C++** for MNIST digit classification, built with **OpenCV matrices**.

MazeNet implements a small multi-layer perceptron without deep learning frameworks. It is designed for learning how neural networks work internally: forward propagation, softmax, cross-entropy loss, backpropagation, gradient descent, and model parameter saving.

---

## Overview

MazeNet trains a fully connected neural network on the MNIST handwritten digit dataset.

```text
Input image  →  Flatten 784 values  →  Hidden layer  →  ReLU  →  Output layer  →  Softmax
28 x 28              784                   128                         10 classes
```

The model learns to classify digits from `0` to `9`.

---

## Features

- Neural network implemented from scratch in C++
- OpenCV `cv::Mat` used for matrix operations
- MNIST `.idx` dataset loader
- Fully connected MLP architecture
- ReLU activation
- Softmax output
- Categorical cross-entropy loss
- Manual backpropagation
- Mini-batch gradient descent
- Accuracy evaluation
- Model parameter export to `.yml`
- CMake build support
- Simple `run.sh` script for quick training

---

## Model Architecture

Default architecture:

```text
784 → 128 → 10
```

| Layer | Size | Activation |
|---|---:|---|
| Input | 784 | - |
| Hidden | 128 | ReLU |
| Output | 10 | Softmax |

Default hyperparameters are defined in `include/config.hpp`:

```cpp
# define in_size 784
# define hide_size 128
# define out_size 10

# define l_rate 0.005
# define total_epochs 10
# define batch_size 64
```

---

## Project Structure

```text
MazeNet/
├── include/
│   ├── config.hpp      # Hyperparameters and model settings
│   ├── mnist.hpp       # MNIST dataset reader declarations
│   ├── model.hpp       # MazeNet model class
│   └── utils.hpp       # Math/helper function declarations
├── src/
│   ├── main.cpp        # Training loop and evaluation
│   ├── mnist.cpp       # MNIST IDX file reader
│   ├── model.cpp       # Forward/backward propagation
│   └── utils.cpp       # Activation, loss, matrix helpers
├── dataset.tar.xz      # Compressed MNIST dataset files
├── CMakeLists.txt      # CMake build configuration
├── run.sh              # Quick build/run script
└── README.md
```

---

## Dataset

The project expects MNIST IDX files inside a `dataset/` directory.

If you use the included archive, extract it first:

```bash
tar -xf dataset.tar.xz
```

Expected paths:

```text
dataset/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

If the official test files are not available, the code can fall back to an 80/20 split from the training dataset.

---

## Build and Run

### Option 1: Use the run script

```bash
chmod +x run.sh
./run.sh
```

---

### Option 2: Build manually with CMake

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./maze
```

On macOS, use:

```bash
make -j$(sysctl -n hw.ncpu)
```

---

## Training Output

During training, MazeNet prints progress, loss, and accuracy:

```text
Start training loop ...

Epoch 1 [==============================] 100%
Loss     : 0.7421
Accuracy : 81.35%
----------------------------------------

Epoch 2 [==============================] 100%
Loss     : 0.4218
Accuracy : 88.72%
----------------------------------------
```

After training, learned parameters are saved to:

```text
trained_model.yml
```

The file contains:

```text
w1
b1
w2
b2
```

---

## How It Works

MazeNet implements the basic neural-network training pipeline manually.

### Forward pass

```text
z1 = X · W1 + b1
a1 = ReLU(z1)
z2 = a1 · W2 + b2
ŷ  = Softmax(z2)
```

### Loss

```text
Categorical Cross-Entropy
```

### Backward pass

```text
dZ2 = ŷ - y
dW2 = a1ᵀ · dZ2
db2 = mean(dZ2)

dA1 = dZ2 · W2ᵀ
dZ1 = dA1 * ReLU'(z1)
dW1 = Xᵀ · dZ1
db1 = mean(dZ1)
```

### Optimization

```text
W = W - learning_rate * gradient
b = b - learning_rate * gradient
```

---

## Requirements

- C++17 compatible compiler
- CMake
- OpenCV

Ubuntu / Debian example:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

---

## Configuration

You can change model and training settings in:

```text
include/config.hpp
```

Example:

```cpp
# define l_rate 0.005
# define total_epochs 10
# define batch_size 64
```

Increase epochs for longer training:

```cpp
# define total_epochs 30
```

Change hidden layer size:

```cpp
# define hide_size 256
```

---
