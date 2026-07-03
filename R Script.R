# ==============================================================
# Stability and Interpretability of Penalized Logistic Regression Models for Breast Cancer Risk Prediction 
# Data directory: C:/Users/kokye/OneDrive/Desktop/Research1
# Input file:     data.csv  (place it in the same folder)
#
# Outputs:
#   - results/tables/*.csv
#   - results/figures/*.png (+ merged Figure 2 PDF + Figure 3 combined)
#   - results/rds/*.rds
#   - results/sessionInfo.txt
#
# Notes:
#   - ALL plots display in RStudio AND are saved to disk.
#   - ALL plots use a consistent color palette.
#   - PLOS ONE: Figure 3 is combined into one composite (Ridge/Lasso/Elastic-Net).
# ==============================================================

rm(list = ls())
set.seed(42)

# ---------------------------
# 0) Packages
# ---------------------------
needed <- c(
  "tidyverse", "glmnet", "pROC",
  "rsample", "yardstick", "broom",
  "fs", "glue", "forcats",
  "patchwork"
)

to_install <- setdiff(needed, rownames(installed.packages()))
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(glmnet)
library(pROC)
library(rsample)
library(yardstick)
library(broom)
library(fs)
library(glue)
library(forcats)
library(patchwork)

# ---------------------------
# Global color palette (colorblind-safe, journal-friendly)
# ---------------------------
COLORS <- list(
  glm   = "#1b9e77",
  ridge = "#7570b3",
  lasso = "#d95f02",
  enet  = "#e7298a",
  cal   = "#1f78b4",
  ref   = "#444444"
)

# Plot theme (consistent look)
THEME_BASE <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# ---------------------------
# 1) Paths & folders
# ---------------------------
ROOT <- "C:/Users/kokye/OneDrive/Desktop/Research1"
DATA_PATH <- file.path(ROOT, "data.csv")

OUT_RESULTS <- file.path(ROOT, "results")
OUT_FIG <- file.path(OUT_RESULTS, "figures")
OUT_TAB <- file.path(OUT_RESULTS, "tables")
OUT_RDS <- file.path(OUT_RESULTS, "rds")

dir_create(OUT_RESULTS, recurse = TRUE)
dir_create(OUT_FIG, recurse = TRUE)
dir_create(OUT_TAB, recurse = TRUE)
dir_create(OUT_RDS, recurse = TRUE)

if (!file.exists(DATA_PATH)) {
  stop("I cannot find data.csv at: ", DATA_PATH,
       "\nPlace data.csv in the folder and run again.")
}

# ---------------------------
# 2) Load & clean data
# ---------------------------
dat_raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE, check.names = TRUE)

dat <- dat_raw %>%
  select(where(~ !all(is.na(.))))

if ("id" %in% names(dat)) dat <- dat %>% select(-id)

if (!("diagnosis" %in% names(dat))) stop("Column 'diagnosis' not found in data.csv.")

dat <- dat %>%
  mutate(
    diagnosis = as.factor(diagnosis),
    y = ifelse(diagnosis == "M", 1L, 0L)
  ) %>%
  select(-diagnosis)

stopifnot(all(dat$y %in% c(0, 1)))

y <- dat$y
X <- dat %>% select(-y)

# ---------------------------
# 3) Train/Test split (stratified)
# ---------------------------
set.seed(42)
split_obj <- initial_split(dat, prop = 0.80, strata = y)
train_dat <- training(split_obj)
test_dat  <- testing(split_obj)

y_train <- train_dat$y
X_train <- train_dat %>% select(-y) %>% as.matrix()

y_test <- test_dat$y
X_test <- test_dat %>% select(-y) %>% as.matrix()

mu <- colMeans(X_train)
sdv <- apply(X_train, 2, sd)
sdv[sdv == 0] <- 1

X_train_sc <- scale(X_train, center = mu, scale = sdv)
X_test_sc  <- scale(X_test,  center = mu, scale = sdv)

saveRDS(list(mu = mu, sd = sdv), file.path(OUT_RDS, "scaling_train_only.rds"))

# ---------------------------
# 4) Helpers: evaluation metrics
# ---------------------------
eval_binary <- function(y_true, p_hat, threshold = 0.5) {
  pred_class <- ifelse(p_hat >= threshold, 1, 0)
  
  tp <- sum(pred_class == 1 & y_true == 1)
  tn <- sum(pred_class == 0 & y_true == 0)
  fp <- sum(pred_class == 1 & y_true == 0)
  fn <- sum(pred_class == 0 & y_true == 1)
  
  acc <- (tp + tn) / length(y_true)
  sens <- ifelse((tp + fn) == 0, NA, tp / (tp + fn))
  spec <- ifelse((tn + fp) == 0, NA, tn / (tn + fp))
  
  auc <- as.numeric(pROC::auc(pROC::roc(y_true, p_hat, quiet = TRUE)))
  brier <- mean((p_hat - y_true)^2)
  
  tibble(
    threshold = threshold,
    accuracy = acc,
    sensitivity = sens,
    specificity = spec,
    auc = auc,
    brier = brier,
    tp = tp, tn = tn, fp = fp, fn = fn
  )
}

best_threshold_youden <- function(y_true, p_hat) {
  roc_obj <- pROC::roc(y_true, p_hat, quiet = TRUE)
  coords <- pROC::coords(
    roc_obj, x = "best", best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity")
  )
  as.numeric(coords["threshold"])
}

# ---------------------------
# 5) Baseline: Unpenalized logistic regression
# ---------------------------
glm_fit <- glm(y ~ ., data = as.data.frame(train_dat), family = binomial())
p_glm <- predict(glm_fit, newdata = as.data.frame(test_dat), type = "response")

thr_glm <- best_threshold_youden(y_test, p_glm)

glm_metrics_05  <- eval_binary(y_test, p_glm, threshold = 0.5)
glm_metrics_opt <- eval_binary(y_test, p_glm, threshold = thr_glm)

write_csv(glm_metrics_05,  file.path(OUT_TAB, "metrics_glm_threshold_0_5.csv"))
write_csv(glm_metrics_opt, file.path(OUT_TAB, "metrics_glm_threshold_opt_youden.csv"))
saveRDS(glm_fit, file.path(OUT_RDS, "glm_unpenalized.rds"))

# ---------------------------
# 6) Penalized models with CV tuning (glmnet)
# ---------------------------
fit_glmnet_cv <- function(alpha, Xtr, ytr) {
  cv.glmnet(
    x = Xtr, y = ytr,
    family = "binomial",
    alpha = alpha,
    nfolds = 10,
    standardize = FALSE
  )
}

pred_prob_glmnet <- function(cvfit, Xnew) {
  as.numeric(predict(cvfit, newx = Xnew, s = "lambda.min", type = "response"))
}

alpha_grid <- c(0, 1, 0.25, 0.5, 0.75)

cv_fits <- map(alpha_grid, ~fit_glmnet_cv(.x, X_train_sc, y_train))
names(cv_fits) <- paste0("alpha_", alpha_grid)

cv_ridge <- cv_fits[["alpha_0"]]
cv_lasso <- cv_fits[["alpha_1"]]

enet_candidates <- alpha_grid[alpha_grid > 0 & alpha_grid < 1]
if (length(enet_candidates) > 0) {
  cv_min_enet <- map_dbl(cv_fits[paste0("alpha_", enet_candidates)], ~min(.x$cvm))
  enet_best_alpha <- enet_candidates[which.min(cv_min_enet)]
  cv_enet <- cv_fits[[paste0("alpha_", enet_best_alpha)]]
} else {
  enet_best_alpha <- NA_real_
  cv_enet <- NULL
}

saveRDS(cv_ridge, file.path(OUT_RDS, "cv_glmnet_ridge_alpha0.rds"))
saveRDS(cv_lasso, file.path(OUT_RDS, "cv_glmnet_lasso_alpha1.rds"))
if (!is.null(cv_enet)) saveRDS(cv_enet, file.path(OUT_RDS, glue("cv_glmnet_enet_alpha{enet_best_alpha}.rds")))

p_ridge <- pred_prob_glmnet(cv_ridge, X_test_sc)
p_lasso <- pred_prob_glmnet(cv_lasso, X_test_sc)
p_enet  <- if (!is.null(cv_enet)) pred_prob_glmnet(cv_enet, X_test_sc) else NULL

thr_ridge <- best_threshold_youden(y_test, p_ridge)
thr_lasso <- best_threshold_youden(y_test, p_lasso)
thr_enet  <- if (!is.null(p_enet)) best_threshold_youden(y_test, p_enet) else NA_real_

ridge_metrics_05  <- eval_binary(y_test, p_ridge, threshold = 0.5)
ridge_metrics_opt <- eval_binary(y_test, p_ridge, threshold = thr_ridge)

lasso_metrics_05  <- eval_binary(y_test, p_lasso, threshold = 0.5)
lasso_metrics_opt <- eval_binary(y_test, p_lasso, threshold = thr_lasso)

write_csv(ridge_metrics_05,  file.path(OUT_TAB, "metrics_ridge_threshold_0_5.csv"))
write_csv(ridge_metrics_opt, file.path(OUT_TAB, "metrics_ridge_threshold_opt_youden.csv"))

write_csv(lasso_metrics_05,  file.path(OUT_TAB, "metrics_lasso_threshold_0_5.csv"))
write_csv(lasso_metrics_opt, file.path(OUT_TAB, "metrics_lasso_threshold_opt_youden.csv"))

if (!is.null(p_enet)) {
  enet_metrics_05  <- eval_binary(y_test, p_enet, threshold = 0.5)
  enet_metrics_opt <- eval_binary(y_test, p_enet, threshold = thr_enet)
  
  write_csv(enet_metrics_05,  file.path(OUT_TAB, "metrics_enet_threshold_0_5.csv"))
  write_csv(enet_metrics_opt, file.path(OUT_TAB, "metrics_enet_threshold_opt_youden.csv"))
}

perf_all <- bind_rows(
  glm_metrics_opt   %>% mutate(model = "GLM"),
  ridge_metrics_opt %>% mutate(model = "Ridge"),
  lasso_metrics_opt %>% mutate(model = "Lasso"),
  if (!is.null(p_enet)) enet_metrics_opt %>% mutate(model = glue("ElasticNet(alpha={enet_best_alpha})")) else NULL
) %>%
  select(model, threshold, accuracy, sensitivity, specificity, auc, brier, tp, tn, fp, fn) %>%
  arrange(desc(auc))

write_csv(perf_all, file.path(OUT_TAB, "performance_summary_opt_threshold.csv"))

# ---------------------------
# 7) FIGURE 2 (Merged): Coefficient paths for Lasso & Elastic-Net
# ---------------------------
make_coef_path_plot <- function(cvfit, model_name, line_color) {
  
  fit <- cvfit$glmnet.fit
  beta_mat <- as.matrix(fit$beta)
  lam <- fit$lambda
  
  df <- as.data.frame(beta_mat)
  df$feature <- rownames(df)
  
  long <- df %>%
    pivot_longer(cols = -feature, names_to = "k", values_to = "beta") %>%
    mutate(lambda = rep(lam, each = nrow(beta_mat))) %>%
    filter(feature != "(Intercept)")
  
  ggplot(long, aes(x = log(lambda), y = beta, group = feature)) +
    geom_line(color = line_color, alpha = 0.60, linewidth = 0.6) +
    labs(
      title = model_name,
      x = expression(log(lambda)),
      y = "Coefficient"
    ) +
    THEME_BASE
}

p_lasso_path <- make_coef_path_plot(cv_lasso, "A) Lasso", COLORS$lasso)

if (is.null(cv_enet)) {
  
  warning("Elastic-net CV fit is NULL. Only Lasso plot will be produced.")
  p_merged <- p_lasso_path +
    plot_annotation(
      title = "Figure 2. Coefficient Shrinkage Paths for Penalized Logistic Regression Models",
      theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    )
  
} else {
  
  p_enet_path <- make_coef_path_plot(
    cv_enet,
    glue("B) Elastic Net (alpha = {enet_best_alpha})"),
    COLORS$enet
  )
  
  p_merged <- (p_lasso_path | p_enet_path) +
    plot_annotation(
      title = "Figure 2. Coefficient Shrinkage Paths for Penalized Logistic Regression Models",
      theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    )
}

print(p_merged)

ggsave(file.path(OUT_FIG, "coef_paths_merged_lasso_enet.png"), p_merged, width = 12, height = 5, dpi = 300)
ggsave(file.path(OUT_FIG, "coef_paths_merged_lasso_enet.pdf"), p_merged, width = 12, height = 5)

# ---------------------------
# 8) Feature Stability (Bootstrap selection frequency)
# ---------------------------
bootstrap_stability <- function(alpha, lambda_choice = c("lambda.min", "lambda.1se"),
                                X, y, B = 1000, seed = 42) {
  
  lambda_choice <- match.arg(lambda_choice)
  set.seed(seed)
  
  feats <- colnames(X)
  sel_count <- setNames(rep(0, length(feats)), feats)
  
  n <- nrow(X)
  
  cvfit0 <- cv.glmnet(X, y, family = "binomial", alpha = alpha, nfolds = 10, standardize = FALSE)
  lam_use <- cvfit0[[lambda_choice]]
  
  for (b in 1:B) {
    idx <- sample.int(n, size = n, replace = TRUE)
    Xb <- X[idx, , drop = FALSE]
    yb <- y[idx]
    
    fitb <- glmnet(Xb, yb, family = "binomial", alpha = alpha, lambda = lam_use, standardize = FALSE)
    
    bb <- as.matrix(coef(fitb))
    bb <- bb[rownames(bb) != "(Intercept)", , drop = FALSE]
    
    selected <- rownames(bb)[as.numeric(bb) != 0]
    if (length(selected) > 0) sel_count[selected] <- sel_count[selected] + 1
  }
  
  tibble(
    feature = names(sel_count),
    selection_frequency = as.numeric(sel_count) / B,
    alpha = alpha,
    lambda_used = lambda_choice,
    B = B
  ) %>%
    arrange(desc(selection_frequency))
}

B_boot <- 1000
pi_star <- 0.70

stab_ridge <- bootstrap_stability(alpha = 0, X = X_train_sc, y = y_train, B = B_boot, seed = 42)
stab_lasso <- bootstrap_stability(alpha = 1, X = X_train_sc, y = y_train, B = B_boot, seed = 42)
stab_enet  <- if (!is.null(cv_enet)) bootstrap_stability(alpha = enet_best_alpha, X = X_train_sc, y = y_train, B = B_boot, seed = 42) else NULL

write_csv(stab_ridge, file.path(OUT_TAB, "stability_ridge.csv"))
write_csv(stab_lasso, file.path(OUT_TAB, "stability_lasso.csv"))
if (!is.null(stab_enet)) write_csv(stab_enet, file.path(OUT_TAB, "stability_enet.csv"))

# ---------------------------
# 8A) FIGURE 3 (Combined): Stability comparison (A/B/C)
# ---------------------------
# Set to TRUE if you still want separate stability plots saved too
SAVE_INDIVIDUAL_STABILITY_PLOTS <- FALSE

make_stability_plot <- function(stab_df, panel_title, fill_color, top_k = 15, pi_line = 0.70) {
  
  df <- stab_df %>%
    slice_head(n = top_k) %>%
    mutate(feature = fct_reorder(feature, selection_frequency))
  
  ggplot(df, aes(x = selection_frequency, y = feature)) +
    geom_col(fill = fill_color, alpha = 0.85) +
    geom_vline(xintercept = pi_line, linetype = 2, color = COLORS$ref) +
    labs(
      title = panel_title,
      x = "Selection frequency (bootstrap)",
      y = NULL
    ) +
    THEME_BASE +
    theme(plot.title = element_text(size = 12, face = "bold"))
}

p_stab_ridge <- make_stability_plot(stab_ridge, "A) Ridge", COLORS$ridge, top_k = 15, pi_line = pi_star)
p_stab_lasso <- make_stability_plot(stab_lasso, "B) Lasso", COLORS$lasso, top_k = 15, pi_line = pi_star)

if (is.null(stab_enet)) {
  warning("Elastic-net stability is NULL; Figure 3 will include Ridge and Lasso only.")
  fig3 <- (p_stab_ridge / p_stab_lasso) +
    plot_annotation(
      title = "Figure 3. Bootstrap Feature Stability Across Penalized Logistic Regression Models",
      theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    )
  fig3_w <- 10
  fig3_h <- 10
} else {
  p_stab_enet <- make_stability_plot(
    stab_enet,
    glue("C) Elastic Net (alpha = {enet_best_alpha})"),
    COLORS$enet,
    top_k = 15,
    pi_line = pi_star
  )
  
  # Vertical layout (recommended for readability)
  fig3 <- (p_stab_ridge / p_stab_lasso / p_stab_enet) +
    plot_annotation(
      title = "Figure 3. Bootstrap Feature Stability Across Penalized Logistic Regression Models",
      theme = theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5))
    )
  
  fig3_w <- 10
  fig3_h <- 14
}

print(fig3)

ggsave(file.path(OUT_FIG, "Figure3_Stability_Combined.png"), fig3, width = fig3_w, height = fig3_h, dpi = 300)
ggsave(file.path(OUT_FIG, "Figure3_Stability_Combined.pdf"), fig3, width = fig3_w, height = fig3_h)

# Optional: save individual stability plots (OFF by default)
if (isTRUE(SAVE_INDIVIDUAL_STABILITY_PLOTS)) {
  ggsave(file.path(OUT_FIG, "stability_ridge_top15.png"), p_stab_ridge, width = 8, height = 5, dpi = 300)
  ggsave(file.path(OUT_FIG, "stability_lasso_top15.png"), p_stab_lasso, width = 8, height = 5, dpi = 300)
  if (!is.null(stab_enet)) {
    ggsave(file.path(OUT_FIG, "stability_enet_top15.png"), p_stab_enet, width = 8, height = 5, dpi = 300)
  }
}

# Stable sets table (unchanged)
stable_sets <- list(
  Ridge = stab_ridge %>% filter(selection_frequency >= pi_star) %>% pull(feature),
  Lasso = stab_lasso %>% filter(selection_frequency >= pi_star) %>% pull(feature),
  ElasticNet = if (!is.null(stab_enet)) stab_enet %>% filter(selection_frequency >= pi_star) %>% pull(feature) else character(0)
)

stable_table <- tibble(
  model = names(stable_sets),
  n_stable = map_int(stable_sets, length),
  stable_features = map_chr(stable_sets, ~ paste(.x, collapse = ", "))
)

write_csv(stable_table, file.path(OUT_TAB, glue("stable_features_pi_{pi_star}.csv")))

# ---------------------------
# 9) Choose "best model" (by AUC from perf_all) + Calibration + ROC
# ---------------------------
best_model <- perf_all %>% slice(1) %>% pull(model)

get_probs_by_name <- function(name) {
  if (name == "GLM") return(p_glm)
  if (name == "Ridge") return(p_ridge)
  if (name == "Lasso") return(p_lasso)
  if (str_detect(name, "ElasticNet")) return(p_enet)
  stop("Unknown model name: ", name)
}

best_color <- function(name) {
  if (name == "GLM") return(COLORS$glm)
  if (name == "Ridge") return(COLORS$ridge)
  if (name == "Lasso") return(COLORS$lasso)
  if (str_detect(name, "ElasticNet")) return(COLORS$enet)
  return(COLORS$ref)
}

p_best <- get_probs_by_name(best_model)
col_best <- best_color(best_model)

cal_df <- tibble(y = y_test, p = p_best) %>%
  mutate(bin = ntile(p, 10)) %>%
  group_by(bin) %>%
  summarise(
    p_mean = mean(p),
    y_rate = mean(y),
    n = n(),
    .groups = "drop"
  )

p_cal <- ggplot(cal_df, aes(x = p_mean, y = y_rate)) +
  geom_point(color = col_best, size = 2) +
  geom_line(color = col_best, linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = COLORS$ref) +
  labs(
    title = glue("Calibration Plot (Deciles) – Best by AUC: {best_model}"),
    x = "Mean predicted probability (bin)",
    y = "Observed event rate (bin)"
  ) +
  THEME_BASE

print(p_cal)
ggsave(file.path(OUT_FIG, "calibration_best_model.png"), p_cal, width = 7, height = 5, dpi = 300)

roc_obj <- pROC::roc(y_test, p_best, quiet = TRUE)
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = col_best, linewidth = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = COLORS$ref) +
  labs(
    title = glue("ROC Curve – {best_model} (AUC = {round(as.numeric(pROC::auc(roc_obj)), 4)})"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  THEME_BASE

print(p_roc)
ggsave(file.path(OUT_FIG, "roc_best_model.png"), p_roc, width = 7, height = 5, dpi = 300)

# ---------------------------
# 10) OPTIONAL: Overlay ROC curves for all models (colored + displayed)
# ---------------------------
roc_all <- bind_rows(
  tibble(model = "GLM",   roc = list(pROC::roc(y_test, p_glm, quiet = TRUE))),
  tibble(model = "Ridge", roc = list(pROC::roc(y_test, p_ridge, quiet = TRUE))),
  tibble(model = "Lasso", roc = list(pROC::roc(y_test, p_lasso, quiet = TRUE))),
  if (!is.null(p_enet)) tibble(model = glue("ElasticNet(alpha={enet_best_alpha})"),
                               roc = list(pROC::roc(y_test, p_enet, quiet = TRUE))) else NULL
) %>%
  mutate(
    fpr = map(roc, ~1 - .x$specificities),
    tpr = map(roc, ~.x$sensitivities),
    auc = map_dbl(roc, ~as.numeric(pROC::auc(.x)))
  ) %>%
  select(model, auc, fpr, tpr) %>%
  unnest(c(fpr, tpr))

model_colors <- c(
  "GLM" = COLORS$glm,
  "Ridge" = COLORS$ridge,
  "Lasso" = COLORS$lasso
)
if (!is.null(p_enet)) {
  model_colors[glue("ElasticNet(alpha={enet_best_alpha})")] <- COLORS$enet
}

p_roc_all <- ggplot(roc_all, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1.0) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = COLORS$ref) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "ROC Curves (All Models)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  THEME_BASE

print(p_roc_all)
ggsave(file.path(OUT_FIG, "roc_all_models.png"), p_roc_all, width = 7.5, height = 5.5, dpi = 300)

# ---------------------------
# 11) Save a clean session summary
# ---------------------------
session_info <- capture.output(sessionInfo())
writeLines(session_info, con = file.path(OUT_RESULTS, "sessionInfo.txt"))

message("DONE. Outputs saved to: ", OUT_RESULTS)
message("Figures saved to: ", OUT_FIG)
message("Tables  saved to: ", OUT_TAB)
