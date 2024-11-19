# Neural Frameworks

## Overview
To train neural networks effectively, several fundamental operations are required:

- **Quickly multiply matrices (tensors):** A key operation in many neural network computations.
- **Compute gradients:** Essential for performing gradient descent optimization.

## What Neural Frameworks Provide
Modern neural frameworks allow you to:
- Operate with tensors on any compute platform (CPU, GPU, or TPU).
- Automatically compute gradients (built-in support for tensor functions).

### Optional Features
- High-level APIs for constructing neural networks (e.g., describing a network as a sequence of layers).
- Simple training functions (like `fit`, as seen in Scikit Learn).
- A variety of optimization algorithms beyond gradient descent.
- Data handling abstractions optimized for compute devices like GPUs.

---

This concise guide outlines the core functionalities of neural frameworks, emphasizing their importance in simplifying and optimizing the process of building and training neural networks.

--
# Most Popular Frameworks

Here are some of the most widely used frameworks for developing neural networks:

1. **TensorFlow 1.x**
   - Developed by Google.
   - Supports defining static computation graphs, GPU acceleration, and explicit evaluation.

2. **PyTorch**
   - Created by Facebook.
   - Known for its dynamic computation graph and growing popularity in the machine learning community.

3. **Keras**
   - A high-level API built on top of TensorFlow and PyTorch.
   - Simplifies the development of neural networks.
   - Created by François Chollet.

4. **TensorFlow 2.x + Keras**
   - A newer version of TensorFlow that integrates Keras functionality.
   - Supports dynamic computation graphs, enabling operations similar to `numpy` and PyTorch.
   - Allows intuitive and flexible development of tensor-based operations.

---

This list highlights the key frameworks empowering modern neural network development, each with its unique strengths and features.
--
# Basic Concepts: Tensor

A **Tensor** is a multi-dimensional array. It is a fundamental data structure in deep learning and is used to represent various types of data.

### Examples of Tensors:
- `400x400` - Black-and-white picture
- `400x400x3` - Color picture
- `16x400x400x3` - Minibatch of 16 color pictures
- `25x400x400x3` - One second of 25-FPS video
- `8x25x400x400x3` - Minibatch of 8 one-second videos

---

## Simple Tensors
You can easily create simple tensors from:
- Lists of `np.arrays`
- Randomly generated data

---

This section provides a quick introduction to the concept of tensors and their practical applications in representing multi-dimensional data.
--
# Variables

**Variables** are essential for representing tensor values that can be modified during the execution of a program. They are particularly useful for tasks such as representing neural network weights.

### Key Features:
- Variables can be updated using methods like `assign` and `assign_add`.
- They allow dynamic modification of values during computation

--
# Computing Gradients

For backpropagation in neural networks, computing gradients is a crucial step. This is typically done using the `tf.GradientTape()` idiom in TensorFlow.

### Steps for Computing Gradients:
1. **Add a `tf.GradientTape` block:**
   Surround your computations with a `with tf.GradientTape` block.
   
2. **Mark tensors for gradient computation:**
   Use `tape.watch` to specify tensors for which gradients need to be computed (though all variables are watched automatically).

3. **Perform computations:**
   Build the computational graph by performing the necessary operations.

4. **Obtain gradients:**
   Use the `tape.gradient` method to compute the gradients.



This guide introduces the use of TensorFlow's `GradientTape` for efficient gradient computation during backpropagation, a foundational concept in neural networks.

# Example 1: Linear Regression

Now we have sufficient knowledge to solve the classical problem of **Linear Regression**. 

### Task:
The goal is to create a small synthetic dataset and use it to perform linear regression.

---

This example demonstrates how foundational concepts in machine learning, such as gradient computation and tensor operations, can be applied to solve real-world problems like linear regression.
--
# Computational Graph and GPU Computations

Whenever we compute tensor expressions, TensorFlow builds a **computational graph** that can be executed on available computing devices, such as CPUs or GPUs.

### Key Concepts:
1. **Computational Graphs:**
   - TensorFlow creates a computational graph for tensor operations.
   - This allows efficient computation on devices like CPUs or GPUs.

2. **Limitations of Arbitrary Python Functions:**
   - Arbitrary Python functions cannot be included in the computational graph.
   - When using a GPU, data might need to transfer between the CPU and GPU for custom Python functions, which reduces efficiency.

3. **Optimizing with `@tf.function`:**
   - TensorFlow provides the `@tf.function` decorator.
   - This decorator converts a Python function into a part of the computational graph.
   - Functions using standard TensorFlow tensor operations can leverage this optimization.

---

This section highlights how TensorFlow manages tensor operations efficiently using computational graphs and the `@tf.function` decorator for performance optimization.
--
# Dataset API

TensorFlow provides a convenient **Dataset API** for efficiently working with data. This API allows seamless integration of data preprocessing and model training.

### Key Features:
- Simplifies data loading and manipulation.
- Optimized for large-scale data pipelines.
- Enables preprocessing, batching, shuffling, and more.

### Objective:
Using the Dataset API, we will:
1. Load and preprocess data.
2. Train a model from scratch using this data.

---

This section introduces TensorFlow's Dataset API, a powerful tool for handling data in machine learning workflows.

--
# Example 2: Classification

In this example, we tackle a **binary classification problem**. 

### Problem Statement:
A common use case is tumor classification — determining whether a tumor is malignant or benign based on its size and age.

### Key Concepts:
- The core model architecture is similar to regression.
- A different **loss function** is required for classification tasks.

### Objective:
We will start by generating sample data and proceed to train a classification model to solve the problem.

---

This example introduces the basics of binary classification and its differences from regression, showcasing how models adapt to various tasks.
--
# Normalizing Data

Before training a neural network, it is common practice to normalize the input features to a standard range, typically `[0, 1]` or `[-1, 1]`.

### Why Normalize?
- Prevent values from becoming too large or too small as they flow through the network.
- Maintain values close to 0, which stabilizes the training process.
- Ensure signals remain within the same range, as weights are initialized with small random values.

### How to Normalize:
1. Subtract the **minimum value** from the dataset.
2. Divide by the **range** (maximum value minus minimum value).

### Practical Considerations:
- Use the **training dataset** to compute the minimum value and range.
- Apply the same normalization parameters to the test and validation datasets.
- Occasionally, new values during prediction may fall outside the `[0, 1]` range, but this is generally not critical.

---

This section emphasizes the importance of data normalization in training stable and efficient neural networks.
--
# Training One-Layer Perceptron

In this section, we use TensorFlow's gradient computation capabilities to train a one-layer perceptron.

### Model Architecture:
- The neural network consists of **2 inputs** and **1 output**.
- The weight matrix \( W \) has a size of \( 2 \times 1 \), and the bias vector \( b \) is a scalar.

### Loss Function:
The loss function used is **logistic loss**, which requires the output to be a probability value between 0 and 1. This is achieved using the **sigmoid activation function**:
\[
p = \sigma(z)
\]

#### Logistic Loss:
Given the probability \( p_i \) for the \( i \)-th input value and the actual class \( y_i \in \{0, 1\} \), the loss is computed as:
\[
L_i = - (y_i \log p_i + (1 - y_i) \log (1 - p_i))
\]

### Implementation in TensorFlow:
- Both the sigmoid activation and logistic loss can be calculated using `sigmoid_cross_entropy_with_logits`.
- Since training is done in minibatches, the total loss is averaged using `reduce_mean`.

---

## Computing Accuracy

To evaluate the model's accuracy on validation data, we can use TensorFlow to cast boolean values to floats and compute the mean:

--
# Using TensorFlow/Keras Optimizers

TensorFlow is closely integrated with Keras, which provides a wide range of useful functionalities. One of these is the ability to use different **optimization algorithms**.

### Key Features:
- TensorFlow/Keras optimizers allow for efficient training by adjusting weights during backpropagation.
- Common optimization algorithms include:
  - **Stochastic Gradient Descent (SGD)**
  - **Adam**
  - **RMSprop**

### Objective:
We will use these optimization algorithms to:
1. Train our model effectively.
2. Print the accuracy obtained during training at each epoch.

---

This section highlights the flexibility and power of TensorFlow/Keras optimizers in enhancing the training process for machine learning models.
--
# Keras

## Deep Learning for Humans

- **Keras** is a library originally developed by François Chollet to work on top of TensorFlow, CNTK, and Theano, unifying lower-level frameworks.
- Although Keras can still be installed as a separate library, it is now included as part of the TensorFlow library and is not recommended to be used separately.
- Allows you to easily construct neural networks layer by layer.
- Includes the `fit` function for streamlined training, along with other functions to work with common data types (e.g., pictures, text, etc.).
- Offers a wide variety of pre-built samples and examples.
- Supports both the **Functional API** and **Sequential API** for model development.

### Why Use Keras?
Keras provides higher-level abstractions for neural networks, enabling developers to focus on layers, models, and optimizers rather than lower-level operations involving tensors and gradients.

---

For more details, check the classical deep learning book by the creator of Keras: [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).
--
# Functional API

When using the **Functional API** in Keras, we define the **input** to the network using `keras.Input`. The **output** is then computed by passing the input through a series of computations. Finally, we define a **model** as an object that maps inputs to outputs.

### Steps to Use the Functional API:
1. **Define Input:**  
   Use `keras.Input` to specify the input tensor for the model.
   
2. **Compute Output:**  
   Pass the input through layers or computations to produce the output.

3. **Create Model:**  
   Use the input and output tensors to define a model object.

### After Defining the Model:
- **Compile the Model:**  
  Specify the loss function and the optimizer to use during training.

- **Train the Model:**  
  Use the `fit` function with the training (and possibly validation) data to train the model.

---

The Functional API is a powerful and flexible way to create complex models in Keras, enabling precise control over model architecture.
--
# Sequential API

The **Sequential API** in Keras offers a simpler way to define models by organizing them as a **sequence of layers**.

### Key Concept:
- A model is thought of as a linear stack of layers, where each layer is added sequentially to a `model` object.

---

The Sequential API is ideal for creating straightforward neural networks, where each layer has one input tensor and one output tensor.
--
# Classification Loss Functions

It is important to correctly select the **loss function** and **activation function** for the last layer of the network. The main rules are as follows:

### Guidelines:
1. **Binary Classification:**
   - Activation function: `sigmoid`.
   - Loss function: **Binary Cross-Entropy** (equivalent to Log Loss).

2. **Multiclass Classification:**
   - Activation function: `softmax`.
   - Loss function:
     - **Categorical Cross-Entropy** for one-hot encoded labels.
     - **Sparse Categorical Cross-Entropy** for outputs with class numbers.

3. **Multi-Label Classification:**
   - Activation function: `sigmoid` (enables assigning probabilities for multiple labels).
   - Encode the labels using one-hot encoding.
   - Loss function: **Categorical Cross-Entropy**.

---

### Summary Table:

| Classification          | Label Format                 | Activation Function | Loss Function                    |
|--------------------------|------------------------------|---------------------|-----------------------------------|
| Binary                  | Probability of 1st class     | `sigmoid`           | Binary Cross-Entropy             |
| Binary                  | One-hot encoding (2 outputs) | `softmax`           | Categorical Cross-Entropy        |
| Multiclass              | One-hot encoding             | `softmax`           | Categorical Cross-Entropy        |
| Multiclass              | Class Number                 | `softmax`           | Sparse Categorical Cross-Entropy |
| Multi-label             | One-hot encoding             | `sigmoid`           | Categorical Cross-Entropy        |

**Note:**  
Binary classification can also be treated as a special case of multiclass classification with two outputs, where the activation function is `softmax` and the loss function is **Categorical Cross-Entropy**.

---

# Task 3: Use Keras to Train MNIST Classifier

- **Dataset Availability:**  
  Keras provides access to standard datasets, including MNIST. Using MNIST with Keras requires just a few lines of code.  
  [More information here](https://keras.io/api/datasets/mnist/)

- **Experimentation:**  
  Try different network configurations:
  - Variations in the number of layers and neurons.
  - Test different activation functions.
--
# Takeaways

- TensorFlow allows you to operate on tensors at a low level, giving you maximum flexibility.
- There are convenient tools for working with data (`tf.Data`) and layers (`tf.layers`).
- For beginners or typical tasks, it is recommended to use **Keras**, which simplifies the process of constructing networks from layers.
- If non-standard architectures are required, you can implement your own custom Keras layer and integrate it into Keras models.
- It is also a good idea to explore **PyTorch** and compare its approaches with TensorFlow/Keras to find the best fit for your tasks.

