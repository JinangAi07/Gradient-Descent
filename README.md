# GradientDescent Package
This package includes two functions to implement linear regression using gradient descent for both one-dimensional and multi-dimensional data.

## 1. Introduction to Gradient Descent
Gradient descent is one of the most commonly used optimization algorithms in machine learning and neural networks. Its primary goal is to minimize the error between the predicted results and the actual results by iteratively adjusting the model's parameters. 

The core idea of gradient descent is to calculate the derivative (gradient) of the loss function with respect to the model's parameters, then update the parameters in the direction opposite to the gradient. This process is repeated iteratively until the algorithm converges to the optimal solution.

The main **formula** for gradient descent is as follows:

$\[
\theta := \theta - \alpha \cdot \nabla_{\theta} J(\theta)
\]$

Where:
- $\theta\$ represents the model parameters.
- $\alpha\$ is the learning rate, controlling the step size of each update.
- $J(\theta)\$ is the loss function, used to measure prediction error.
- $\nabla_{\theta} J(\theta)\$ denotes the gradient of the loss function with respect to the parameters.

Through multiple iterations, gradient descent gradually finds the parameter values that minimize the loss function, thus improving the accuracy of the model's predictions.

![Image text](/graphics/gd1.png)

The diagram illustrates this process. The blue curve represents the contour lines (level sets) and the arrows indicate the direction opposite to the gradient at that point. (Note: The gradient direction at a point is perpendicular to the contour lines passing through that point). By following the gradient descent direction, we will eventually reach to the bottom, which is the point where the loss function J is minimized.

## 2. Linear regression using gradient descent

In this R package, we focus on the application of gradient descent in both univariate and multivariate functions. To this end, we provide two functions, **GD_one_dim** and **GD_multi_dim**, to implement linear regression and also offer visualizations.

### 2.1 Convex Function

Here, we introduce the concept of convex functions to simplify the optimization process, because the minimization problem of a convex function has uniqueness, meaning that a strictly convex function over a convex open set can have at most one local minimum and finding a local minimum on a convex function will also yield a global minimum. 

The essence of gradient descent is to find the extremum and numerical solution of a function. For machine learning algorithms, as long as the loss function can be expressed as a convex function, gradient descent can be used to update the weight vector w at the fastest rate, thereby finding the point where the loss function reaches its minimum.

The main **formula** for convex functions of one variable is as follows:

The main **formula** for convex functions of several variable is as follows:

### 2.2 GD_one_dim Function
This function performs one-dimensional linear regression using gradient descent. It iteratively adjusts the weight w and bias b to minimize the loss function (mean squared error). Users can specify the learning rate, the number of iterations, and the display interval. In each iteration, the function computes the model's predictions and loss, and at the end, it visualizes the regression line, the change in loss over iterations, and the difference between the predicted and actual values.

### 2.3 GD_multi_dim Funtion
This function implements multi-dimensional linear regression for multiple features (variables). After normalizing the data, it uses matrix operations to perform gradient descent, updating the weight matrix W. Users can set the learning rate, the number of iterations, and the display interval. The function outputs the change in loss during training and plots the comparison between predicted and actual values.

Both functions use gradient descent to train the model, making them useful for simple regression tasks with intuitive visualizations to analyze convergence and predictive performance.

## 3. Others
This package also contains a vignette, which describe the introduction of two functions and a test which compare the output of this package with another widely-used package 'lm'. 
