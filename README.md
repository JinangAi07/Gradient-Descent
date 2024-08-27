This package includes two functions to implement linear regression using gradient descent for both one-dimensional and multi-dimensional data.

1. GD_one_dim:
This function performs one-dimensional linear regression using gradient descent. It iteratively adjusts the weight w and bias b to minimize the loss function (mean squared error). Users can specify the learning rate, the number of iterations, and the display interval. In each iteration, the function computes the model's predictions and loss, and at the end, it visualizes the regression line, the change in loss over iterations, and the difference between the predicted and actual values.

2. GD_multi_dim:
This function implements multi-dimensional linear regression for multiple features (variables). After normalizing the data, it uses matrix operations to perform gradient descent, updating the weight matrix W. Users can set the learning rate, the number of iterations, and the display interval. The function outputs the change in loss during training and plots the comparison between predicted and actual values.

Both functions use gradient descent to train the model, making them useful for simple regression tasks with intuitive visualizations to analyze convergence and predictive performance.

3. This package also contains a vignette, which describe the introduction of two functions and a test which compare the output of this package with another widely-used package 'lm'. 
