# CNN
Convolutional Neural Network - designed for image classification

## Data and Processing

Image $\rightarrow$ 3-D tensor (length, width, channel(RGB))

First, flatten the 3-D tensor to a vector

## Network Structure Design

### Neuron
#### Receptive Field
- some patterns are much smaller than the whole image
- a neuron does not have to see the whole image

Features
- the specific region of the sensory space
- the receptive fields cover the whole image

Typical Settings
- all channels
- `kernel size` (3 $\times$ 3)
- `stride` (better make the field overlapped)
- `padding` (when field exceeds)

#### Parameter Sharing
- the same patterns appear in different regions

Features
- two neurons with the same receptive field would not share parameters

Typical Setting
- each receptive field has a set of neurons
- each receptive field has the neurons with the same set of parameters (called `filter`, e.g. filter 1, fliter 2, etc.)

### Convolutional Layer
- Composed of multiple filters 
- Generate `feature map` with some channels the same number of filters 
- Connected with the next convolutional layer

### Pooling
- Subsampling the pixels will not change the object
- With no parameter to learn
- Aim to reduce operation

#### Max Pooling
- to choose the max from a field by a filter

## The Whole CNN

$$ 
\boxed{\text{Convolution}}
\\
\downarrow
\\
\boxed{\text{Pooling}}
\\
\downarrow
\\
\cdots
\\
\downarrow
\\
\boxed{\text{Flatten}}
\\
\downarrow
\\
\boxed{\text{Fully Connected}}
\\
\downarrow
\\
\boxed{\text{Softmax}}
$$
