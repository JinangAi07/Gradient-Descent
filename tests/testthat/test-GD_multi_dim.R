test_that("GD_multi_dim produces reasonable loss value", {
  # Test input data
  x1 <- c(137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00)
  x2 <- c(3, 2, 2, 3, 1, 2, 3, 2)
  x <- cbind(x1, x2)
  y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 81.00)

  # Run GD_multi_dim
  gd_result <- GD_multi_dim(x, y, iter = 500, display_step = 100)
  final_loss_gd <- gd_result$final_loss

  # Compare with linear model (lm)
  model_lm <- lm(y ~ x1 + x2)
  predictions_lm <- predict(model_lm, newdata = data.frame(x1, x2))
  loss_lm <- mean((y - predictions_lm)^2) / 2

  # Compare losses
  expect_equal(final_loss_gd, loss_lm, tolerance = 0.1)
})

