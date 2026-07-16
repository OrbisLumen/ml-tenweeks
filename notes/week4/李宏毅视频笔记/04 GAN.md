# GAN
 
- Generative Adversarial Network

## Basic Idea

Generator
- neural network
- input: vector
- output: high dimensional vector (image)

Discriminator
- neural network
- input: image
- output: scalar
  - Larger value means real, smaller value means fake

Adversarial
- Generator vs Discriminator

## Algorithm

- Initialize generator and discriminator 
- In each training iteration
  - Step 1: Fix generator G, and update discriminator D
    - Discriminator learns to assign high scores to real objects and low scores to generated objects
  - Step 2: Fix discriminator D, and update generator G
    - Generator learns to "fool" the discriminator