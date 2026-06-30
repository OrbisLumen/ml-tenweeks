# Ups and downs of Deep Learning

- 1958: Perception (linear model)
- 1969: Perception has limitation
- 1980s: Multi-layer perception
  - Do not have significant difference from DNN today
- 1986: Backpropagation
  - Usually more than 3 hidden layers is not helpful
- 1989: 1 hidden layer is "good enough", why deep?
- 2006: RBM initialization (breakthrough)
- 2009: GPU
- 2011: Start to be popular in speech recognition
- 2012: win ILSVRC image competition

# Three Steps for Deep Learning

$$
\boxed{\text{Step 1: define a set of function}} 
\\
\downarrow
\\
\boxed{\text{Step 2: goodness of function}}
\\
\downarrow
\\
\boxed{\text{Step 3: pick the best function}}
$$

## Step 1

*Neural Network*: Different connection leads to different network structures

### "Neuron"

![alt text](Neuron.svg)

- Network parameter $\theta$: all the weights and biases in the "neurons"

### Connection ways

**Fully Connect Feedforward Network**

![alt text](<Fully Connect.svg>)

- This is a function 
- Input vector, output vector
- Given network structure, define a function set

**Deep** = **Many Hidden Layers**

- AlexNet, VGG, GoogleNet, Residual Net

### Structure

**Input Layer** 
**Hidden Layers** - Feature extractor replacing feature engineering
**Output Layer** - Muti-class Classifier 

## Step 2

### Total Loss 

$$
L = \sum_{n=1}^N C_n
$$

- Find *the network parameters* that minimize total loss L

## Step 3

### Gradient Descent
To solve the following optimization problem:
$$
\theta^* = arg \min_{\theta} L(\theta) 
$$

### Backpropagation
An efficient way to compute $\partial L/\partial w$ in neural network