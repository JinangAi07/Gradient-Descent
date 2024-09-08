test_that("GD_one_dim produces reasonable loss value", {
  # Test input data
  insurance <- read.csv('insurance.csv')
  x <- insurance$age
  y <- insurance$charges

  # Run GD_one_dim and return output
  gd_result <- GD_one_dim(x, y, iter = 100, display_step = 100)
  final_loss_gd <- gd_result$final_loss

  # Compare with another widely-used package linear model (lm)
  model_lm <- lm(y ~ x)
  predictions_lm <- predict(model_lm)
  loss_lm <- mean((y - predictions_lm)^2) / 2

  # Compare losses with tolerance = 0.1
  expect_equal(final_loss_gd, loss_lm, tolerance = 0.1)
})
