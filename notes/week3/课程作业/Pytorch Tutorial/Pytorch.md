# What is PyTorch
- An machine learning framework in Py
- Main features
  - N-dimensional Tensor computation on GPUs
  - Automatic differentiation for training deep neural networks

# Training & Testing
- Load Date
- Training 
- Validation 
- Testing

## Dataset & Dataloader
- Dataset: stores data samples and expected values
- Dataloader: groups data in batches, enables multiprocessing

```py
dataset = MyDataset(file)
dalaloader = DataLoader(dataset, batch_size, shffle=True)
```

## Tensor
- High-dimensional matrices

## Torch.nn
- Network Layers

## Torch.optim
- Gradient-based optimization algorithms
- For every batch of data
  - .zero_grad()
  - .backward()
  - .step()

## Notice
- model.eval()
  - changes behaviour of some model layers, such as dropout and batch normalization
- with torch.no_grad()
  - prevents calculations from being added into gradient computation graph
  - usually used to prevent accidental training on validation/testing data

## Save/Load
```py
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```
