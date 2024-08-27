test_that("GD_one_dim produces reasonable loss value", {
  # Test input data
  x <- c(137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00)
  y <- c(145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 81.00)

  # Run GD_one_dim
  gd_result <- GD_one_dim(x, y, iter = 100, display_step = 100)
  final_loss_gd <- gd_result$final_loss

  # Compare with linear model (lm)
  model_lm <- lm(y ~ x)
  predictions_lm <- predict(model_lm)
  loss_lm <- mean((y - predictions_lm)^2) / 2

  # Compare losses
  expect_equal(final_loss_gd, loss_lm, tolerance = 0.1)
})
