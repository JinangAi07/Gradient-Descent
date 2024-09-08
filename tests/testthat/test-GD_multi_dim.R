test_that("GD_multi_dim produces reasonable loss value", {
  # Test input data
  insurance <- read.csv('insurance.csv')
  x1 <- insurance$age
  x2 <- insurance$bmi
  x3 <- insurance$children
  x <- cbind(x1, x2, x3)
  y <- insurance$charges

  # Run GD_multi_dim and return output
  gd_result <- GD_multi_dim(x, y, iter = 500, display_step = 100)
  final_loss_gd <- gd_result$final_loss

  # Compare with another widely-used package linear model (lm)
  model_lm <- lm(y ~ x1 + x2 + x3)
  predictions_lm <- predict(model_lm, newdata = data.frame(x1, x2, x3))
  loss_lm <- mean((y - predictions_lm)^2) / 2

  # Compare losses with tolerance = 0.1
  expect_equal(final_loss_gd, loss_lm, tolerance = 0.1)
})

