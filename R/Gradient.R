#' one-dimensional linear regression
#' @param x data vector
#' @param y data vector
#' @param learn_rate number
#' @param iter number
#' @param display_step number
#' @return gradient descent output
#' @examples
#' x <- c(137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00)
#' y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00)
#' GD_one_dim(x, y)
#' GD_one_dim(x, y, iter = 50)
#' GD_one_dim(x, y, iter = 30)
#' GD_one_dim(x, y, display_step = 5)
#' GD_one_dim(x, y, iter = 50, display_step = 5)
#' @export
GD_one_dim <- function(x, y, learn_rate = 0.00001, iter = 100, display_step = 10){

  set.seed(123)
  ##set the parameter

  #learn_rate
  #learn_rate <- 0.00001



  #iteration numbers
  #iter <- 100

  #Show the effect every 10 iterations
  #display_step <- 10

  #initialization
  w <- 0.5
  b <- 0.5

  ##train the model
  mse <- numeric(iter + 1)

  for (i in 0:iter) {
    dL_dw <- mean(x * (w * x + b - y))
    dL_db <- mean(w * x + b - y)
    w <- w - learn_rate * dL_dw
    b <- b - learn_rate * dL_db
    pred <- w * x + b
    Loss <- mean((y - pred)^2) / 2
    mse[i + 1] <- Loss
    if (i %% display_step == 0) {
      cat(sprintf("i:%i, Loss:%f, w:%f, b:%f\n", i, mse[i + 1], w, b))
    }
  }

  ##visualization
  graphics::par(mfrow = c(1, 3), mar = c(5, 4, 4, 2) + 0.1)

  #scatterplot
  plot(x, y, col = "indianred", pch = 19)
  graphics::points(x, pred, col = "steelblue", pch = 19)
  graphics::lines(x, pred, col = "steelblue")

  #loss
  plot(mse, type = "l", xlab = "iteration numbers", ylab = "Loss")

  #differences
  plot(y, type = "o", col = "indianred", pch = 19)
  graphics::lines(pred, type = "o", col = "steelblue", pch = 19)

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
#' y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00)
#' GD_multi_dim(x, y)
#' GD_multi_dim(x, y, iter = 200)
#' GD_multi_dim(x, y, iter = 300, display_step = 25)
#' @export
GD_multi_dim <- function(x, y, learn_rate = 0.001, iter = 500, display_step = 50){

  x0 <- rep(1, nrow(x))
  #normalization
  for (i in 1:ncol(x)) {
    x[, i] <- (x[, i] - min(x[, i])) / (max(x[, i]) - min(x[, i]))
  }

  X <- cbind(x0, x)
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

  set.seed(123)
  #learn_rate <- 0.001
  #iter <- 500
  #display_step <- 50
  W <- matrix(rep(0.5, ncol(X)))

  ##train the model
  mse <- numeric()
  for (i in 0:iter) {
    dL_dW <- t(X) %*% (X %*% W - Y)   # XT(XW-Y)
    W <- W - learn_rate * dL_dW
    PRED <- X %*% W
    Loss <- mean((Y - PRED)^2) / 2
    mse <- c(mse, Loss)
    if (i %% display_step == 0) {
      cat(sprintf("i:%i, Loss:%f\n", i, mse[i + 1]))
    }
  }

  ##visualization
  graphics::par(mfrow = c(1, 2))
  #Loss
  plot(mse, type = "l", xlab = "iteration numbers", ylab = "Loss")

  #differences
  PRED <- as.vector(PRED)
  plot(y, col = "indianred", pch = 19)
  graphics::points(PRED, col = "steelblue", pch = 19)
  graphics::lines(PRED, pch = 19)
  graphics::lines(y, pch = 19)

}

