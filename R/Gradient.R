#' one-dimensional linear regression
#' @param x data vector
#' @param y data vector
#' @param learn_rate number
#' @param iter number
#' @param display_step number
#' @return gradient descent output
#' @examples
#' x <- c(137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00)
#' y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 81.00)
#' GD_one_dim(x, y)
#' GD_one_dim(x, y, iter = 50)
#' GD_one_dim(x, y, iter = 30)
#' GD_one_dim(x, y, display_step = 5)
#' GD_one_dim(x, y, iter = 50, display_step = 5)
#' @export

##Function Definition and Parameters:
GD_one_dim <- function(x, y, learn_rate = 0.00001, iter = 100, display_step = 10){
  #'x': Independent variable (input data).
  #'y': Dependent variable (target values).
  #'learn_rate': The learning rate for gradient descent, default value '0.00001'.
  #'iter': The number of iterations for the gradient descent algorithm, default '100'.
  #'display_step': Defines how often the loss and parameters should be displayed (every '10' iterations by default).

  ##Seed Initialization:
  set.seed(123)
  #Sets the seed for reproducibility of results. This ensures that the random processes used in training will produce the same result every time.

  ##Initialization of Parameters:
  w <- 0.5
  b <- 0.5
  #'w': Weight for the linear model (starting at '0.5').
  #'b': Bias term (starting at '0.5').

  ##Mean Squared Error (MSE) Storage:
  mse <- numeric(iter + 1)
  #'mse': Initializes a numeric vector to store the loss at each iteration. The length of the vector is 'iter + 1' because it includes the initial loss (at iteration '0').

  ##Training Loop:
  for (i in 0:iter) {
  #Starts a loop that runs from iteration '0' to 'iter' (inclusive), which updates the parameters 'w' and 'b' using gradient descent.

    ##Gradient Calculation:
    dL_dw <- mean(x * (w * x + b - y))
    dL_db <- mean(w * x + b - y)
    #'dL_dw': The gradient of the loss function with respect to the weight 'w'. It measures how much 'w' should be adjusted.
    #'dL_db': The gradient of the loss function with respect to the bias 'b'. It measures how much 'b' should be adjusted.

    ##Update Parameters:
    w <- w - learn_rate * dL_dw
    b <- b - learn_rate * dL_db
    #The weight 'w' and bias 'b' are updated using the calculated gradients multiplied by the learning rate.

    ##Prediction and Loss Calculation:
    pred <- w * x + b
    Loss <- mean((y - pred)^2) / 2
    mse[i + 1] <- Loss
    #'pred': Predictions made by the model using the current values of 'w' and 'b'.
    #'Loss': Calculates the loss using the mean squared error divided by 2 (standard for squared loss in gradient descent).
    #'mse[i + 1]': Stores the calculated loss for the current iteration.

    ##Display Information:
    if (i %% display_step == 0) {
      cat(sprintf("i:%i, Loss:%f, w:%f, b:%f\n", i, mse[i + 1], w, b))
    }
    #Every 'display_step' iterations, the function prints out the iteration number, the current loss, and the values of 'w' and 'b'.
  }


  ##Visualization:
  graphics::par(mfrow = c(1, 3), mar = c(5, 4, 4, 2) + 0.1)
  #Sets up the plotting layout to have 1 row and 3 columns. Adjusts the margin sizes.

  ##Scatter Plot of Data and Predictions:
  plot(x, y, col = "indianred", pch = 19)
  graphics::points(x, pred, col = "steelblue", pch = 19)
  graphics::lines(x, pred, col = "steelblue")
  #Plots the original data points ('x', 'y') in red.
  #Plots the predicted values ('x', 'pred') in blue.
  #Adds a line connecting the predicted points.

  ##Loss plot:
  plot(mse, type = "l", xlab = "iteration numbers", ylab = "Loss")
  #Plots the loss over the iterations to visualize the convergence of the gradient descent algorithm.

  ##Differences Plot:
  plot(y, type = "o", col = "indianred", pch = 19)
  graphics::lines(pred, type = "o", col = "steelblue", pch = 19)
  #Plots the actual 'y' values and the predicted 'pred' values, showing how close the model predictions are to the actual values.

  ##Return final loss for comparison:
  return(list(final_loss = mse[iter + 1], mse = mse))
  #Returns a list containing the final loss after all iterations and the vector of MSE values throughout training.
}


#' multiple linear regression
#' @param x data matrix
#' @param y data matrix
#' @param learn_rate number
#' @param iter number
#' @param display_step number
#' @return gradient descent output
#' @examples
#' x1 <- c(137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00)
#' x2 <- c(3, 2, 2, 3, 1, 2, 3, 2)
#' x <- cbind(x1, x2)
#' y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 81.00)
#' GD_multi_dim(x, y)
#' GD_multi_dim(x, y, iter = 200)
#' GD_multi_dim(x, y, iter = 300, display_step = 25)
#' @export
##Function Definition and Parameters:
GD_multi_dim <- function(x, y, learn_rate = 0.001, iter = 500, display_step = 50){
  #'x': Matrix of independent variables (multi-dimensional input data).
  #'y': Dependent variable (target values).
  #'learn_rate': Learning rate for the gradient descent algorithm, default '0.001'.
  #'iter': Number of iterations for the gradient descent algorithm, default '500'.
  #'display_step': Determines how often the loss is displayed, default every '50' iterations.

  ##Adding Bias Column:
  x0 <- rep(1, nrow(x))
  #Creates a column of ones to represent the bias term 'x0'.

  ##Normalizing Data:
  for (i in 1:ncol(x)) {
    x[, i] <- (x[, i] - min(x[, i])) / (max(x[, i]) - min(x[, i]))
  }
  #Normalizes each column of 'x' to scale the data between '0' and '1'. This helps the gradient descent converge faster.

  ##Constructing Matrix X by Adding Bias:
  X <- cbind(x0, x)
  #Combines the bias column 'x0' and the normalized independent variables 'x' to form the matrix 'X'. The first column is all 1s, representing the bias term.

  ##Printing Variables for Debugging:
  print("independent variables")
  print(X)
  cat("



")
  Y <- matrix(y, ncol = 1)
  print("response variable")
  print(Y)
  cat("



")
  print("Loss and iteration numbers")
  #Prints the transformed independent variables matrix 'X' and the dependent variable 'Y' for debugging purposes.

  ##Initializing Weights:
  set.seed(123)
  W <- matrix(rep(0.5, ncol(X)))
  mse <- numeric()
  #Sets a random seed for reproducibility and initializes the weight matrix 'W' with values of '0.5'. The number of weights corresponds to the number of columns in 'X' (features).

  ##Training Loop:
  for (i in 0:iter) {
  #Starts a loop that runs from iteration '0' to 'iter', which will update the weights 'W'.

    ##Gradient Calculation:
    dL_dW <- t(X) %*% (X %*% W - Y)
    #Computes the gradient 'dL_dW', which is the derivative of the loss function with respect to the weights 'W'. The formula used is 'X^T * (X * W - Y)'.

    ##Weight Update:
    W <- W - learn_rate * dL_dW
    #Updates the weight matrix 'W' using the gradient and the learning rate.

    ##Prediction and Loss Calculation:
    PRED <- X %*% W
    Loss <- mean((Y - PRED)^2) / 2
    mse <- c(mse, Loss)
    #'PRED': Calculates the predictions using the current weight matrix 'W'.
    #'Loss': Computes the mean squared error loss and divides by 2 (standard in gradient descent for squared loss).
    #Stores the current iteration's loss in the 'mse' vector.

    ##Display Loss:
    if (i %% display_step == 0) {
      cat(sprintf("i:%i, Loss:%f\n", i, mse[i + 1]))
    }
    #Every 'display_step' iterations, prints the current iteration and loss.
  }

  ##visualization:
  graphics::par(mfrow = c(1, 2))
  #Sets up the plotting area to have 1 row and 2 columns for visualization.

  ##Loss Plot:
  plot(mse, type = "l", xlab = "iteration numbers", ylab = "Loss")
  #Plots the loss curve over the iterations, showing how the loss decreases as the gradient descent progresses.

  ##Differences Plot:
  PRED <- as.vector(PRED)
  plot(y, col = "indianred", pch = 19)
  graphics::points(PRED, col = "steelblue", pch = 19)
  graphics::lines(PRED, pch = 19)
  graphics::lines(y, pch = 19)
  #Plots the actual 'y' values and predicted 'PRED' values in red and blue, respectively, showing the differences between model predictions and actual values.

  ##Return final loss for comparison:
  return(list(final_loss = mse[iter + 1], mse = mse))
  #Returns a list containing the final loss after all iterations and the vector of loss values ('mse') throughout the training process.
}

