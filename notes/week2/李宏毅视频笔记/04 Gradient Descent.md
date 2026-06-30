# Gradient Descent

## Tip 1
Set the learning rate $\eta$ carefully

### Adaptive Learning Rate
- Popular & Simple Idea: Reduce the learning rate by some factor every few epochs
- Learning rate cannot be one-size-fits-all

## Tip 2
### Stochastic Gradient Descent
- Loss for only an example or a few examples

## Tip 3
### Feature Scaling
- For each dimension of input feature:
$$
x_i \leftarrow \frac{x_i - m_i}{\sigma_i}
$$
- $m_i$: mean, $\sigma_i$: standard deviation
- And now the means of all dimensions are 0, and the variances are all 1

## Tip 4
### Limitation of Gradient Descent
- Stuck at local minima
- Stuck at saddle point 
- Very slow at the plateau