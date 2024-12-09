# Load the libraries in the correct order
library(data.table)      # For efficient data loading
library(tm)              # For text cleaning and preprocessing
library(Matrix)          # Must be loaded before irlba
library(irlba)           # For Singular Value Decomposition
library(dplyr)           # For data manipulation
library(ggplot2)         # For visualizations
library(caret)           # For dummyVars and data splitting
library(randomForest)    # For Random Forest model
library(xgboost)         # For XGBoost model
library(Metrics)         # For RMSE and MAE calculations
library(tidytext)        # For sentiment analysis
library(tidyr)           # For data manipulation
library(SnowballC)       # For stemming
library(lubridate)       # For date manipulation
library(stringr)         # For string operations
library(slam)
library(RSpectra)
rm(list = ls())  # Remove all objects
gc()             # Force garbage collection

# ---------------------
# 2. Data Loading and Initial Filtering
# ---------------------
cat("Loading data...\n")
sampled_data <- fread("/Users/lyndseyhuss/Documents/MGSC_410/MGSC_410/song_lyrics.csv", nrows = 1000000)

# Convert 'views' to numeric and handle NAs
sampled_data[, views := as.numeric(views)]
num_na_views <- sum(is.na(sampled_data$views))
if (num_na_views > 0) {
  cat("Number of rows with NA in 'views':", num_na_views, "\n")
  sampled_data <- sampled_data[!is.na(views)]
}

# Filter data where views >= 10,000
threshold <- 10000
filtered_data <- sampled_data[views >= threshold]
cat("Number of rows after filtering (views >= 10,000):", nrow(filtered_data), "\n")

# Remove the original sampled_data to free up memory
rm(sampled_data)
gc()

# Add log-transformed target variable
filtered_data[, log_views := log1p(views)]

# ---------------------
# 3. Text Preprocessing
# ---------------------
cat("Preprocessing lyrics...\n")
# Remove newline characters and excessive whitespace from lyrics
filtered_data[, lyrics := str_replace_all(lyrics, "\\n", " ")]
filtered_data[, lyrics := str_squish(lyrics)]  # Removes extra spaces

# Remove rows with empty or nearly empty lyrics
empty_lyrics <- which(filtered_data$lyrics == "")
if (length(empty_lyrics) > 0) {
  cat("Number of empty lyrics to remove:", length(empty_lyrics), "\n")
  filtered_data <- filtered_data[-empty_lyrics, ]
}

# Create text corpus
cat("Creating and cleaning text corpus...\n")
corpus <- Corpus(VectorSource(filtered_data$lyrics))

# Define a custom content transformer to handle specific transformations
toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x)) })

# Apply transformations
corpus <- tm_map(corpus, toSpace, "/|@|\\|")              # Replace specific characters with space
corpus <- tm_map(corpus, content_transformer(tolower))   # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)              # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))   # Remove English stopwords
corpus <- tm_map(corpus, stripWhitespace)                # Remove extra whitespace
corpus <- tm_map(corpus, stemDocument)                   # Perform stemming

# ---------------------
# 4. Feature Engineering
# ---------------------
cat("Engineering features...\n")

# a. Parsing 'features' to count collaborators
parse_collaborators <- function(features_str) {
  clean_str <- gsub('[{}"]', '', features_str)
  collaborators <- unlist(strsplit(clean_str, ","))
  collaborators <- str_trim(collaborators)
  collaborators <- collaborators[collaborators != ""]
  return(length(collaborators))
}

filtered_data[, num_collaborators := sapply(features, parse_collaborators)]

# b. Word count from lyrics
filtered_data[, word_count := str_count(lyrics, "\\S+")]

# c. Sentiment Analysis using multiple lexicons
cat("Performing sentiment analysis...\n")
tokenized_lyrics <- filtered_data %>%
  select(id, lyrics) %>%
  unnest_tokens(word, lyrics)

# AFINN Sentiment Scores
afinn <- get_sentiments("afinn")
sentiment_afinn <- tokenized_lyrics %>%
  inner_join(afinn, by = "word") %>%
  group_by(id) %>%
  summarise(sentiment_afinn = sum(value, na.rm = TRUE)) 

# BING Sentiment Counts
bing <- get_sentiments("bing")
sentiment_bing <- tokenized_lyrics %>%
  inner_join(bing, by = "word") %>%
  count(id, sentiment) %>%
  spread(sentiment, n, fill = 0)

# NRC Sentiment Counts
nrc <- get_sentiments("nrc")
sentiment_nrc <- tokenized_lyrics %>%
  inner_join(nrc, by = "word") %>%
  count(id, sentiment) %>%
  spread(sentiment, n, fill = 0)

# Merge all sentiment scores
filtered_data <- filtered_data %>%
  left_join(sentiment_afinn, by = "id") %>%
  left_join(sentiment_bing, by = "id") %>%
  left_join(sentiment_nrc, by = "id")

# Replace NAs with 0
filtered_data[is.na(filtered_data)] <- 0

# Interaction Features
filtered_data <- filtered_data %>%
  mutate(
    word_sentiment_interaction = word_count * sentiment_afinn,
    collab_sentiment_interaction = num_collaborators * sentiment_afinn
  )

# d. TF-IDF Vectorization with Dimensionality Reduction
cat("Creating Document-Term Matrix (DTM) with TF-IDF weighting...\n")
dtm <- DocumentTermMatrix(corpus, control = list(
  weighting = weightTfIdf,
  bounds = list(global = c(50, Inf))  # Keep terms appearing in at least 50 documents
))

# ---------------------
# 5. Handling Empty Documents
# ---------------------
cat("Identifying and removing empty documents...\n")
empty_docs <- which(slam::row_sums(dtm) == 0)  # Use slam::row_sums for efficient computation
if (length(empty_docs) > 0) {
  cat("Number of empty documents to remove:", length(empty_docs), "\n")
  dtm <- dtm[-empty_docs, ]  # Remove empty rows from the DTM
  filtered_data <- filtered_data[-empty_docs, ]  # Remove corresponding rows from the dataset
}

# ---------------------
# 6. Converting DTM to Sparse Matrix
# ---------------------
cat("Converting DTM to sparse matrix format...\n")

# Convert DocumentTermMatrix to simple_triplet_matrix
dtm_triplet <- slam::as.simple_triplet_matrix(dtm)

# Convert simple_triplet_matrix to dgCMatrix (sparse matrix format)
dtm_sparse <- sparseMatrix(
  i = dtm_triplet$i,
  j = dtm_triplet$j,
  x = dtm_triplet$v,
  dims = c(dtm_triplet$nrow, dtm_triplet$ncol)
)


# ---------------------
# 7. Performing Singular Value Decomposition (SVD)
# ---------------------
# Perform SVD using RSpectra
svd_result <- svds(dtm_sparse, k = 100)  # Replace nv with k

# Extract the reduced dimensions
lyrics_features <- as.data.frame(svd_result$u %*% diag(svd_result$d))
colnames(lyrics_features) <- paste0("lyrics_dim_", 1:100)

# Combine lyrics features with the main dataset
filtered_data <- cbind(filtered_data, lyrics_features)

# Release memory
rm(dtm, dtm_triplet, dtm_sparse, svd_result, lyrics_features)
gc()
# ---------------------
# 8. One-hot Encoding for 'tag' (genre)
# ---------------------
cat("Encoding categorical variables...\n")
genre_dummies <- dummyVars(~ tag, data = filtered_data)
genre_encoded <- as.data.frame(predict(genre_dummies, newdata = filtered_data))
filtered_data <- cbind(filtered_data, genre_encoded)

# ---------------------
# 9. Additional Feature Engineering
# ---------------------
# Example: Song Age
filtered_data[, song_age := 2024 - year]

# ---------------------
# 10. Normalize Numeric Features (Optional for Tree-based Models)
# ---------------------
numeric_features <- c("song_age", "num_collaborators", "word_count", "sentiment_afinn", 
                      "word_sentiment_interaction", "collab_sentiment_interaction")
filtered_data[, (numeric_features) := lapply(.SD, scale), .SDcols = numeric_features]

# Visualize the distribution of views
ggplot(filtered_data, aes(x = views)) +
  geom_histogram(bins = 50, fill = "blue", alpha = 0.7) +
  scale_x_log10() +  # Log scale to better visualize skewed data
  theme_minimal() +
  labs(title = "Distribution of Views", x = "Views (log scale)", y = "Frequency")

# Define function to handle outliers
handle_outliers <- function(data, column, lower_quantile = 0.01, upper_quantile = 0.99) {
  lower_bound <- quantile(data[[column]], lower_quantile, na.rm = TRUE)
  upper_bound <- quantile(data[[column]], upper_quantile, na.rm = TRUE)
  data <- data[data[[column]] >= lower_bound & data[[column]] <= upper_bound, ]
  return(data)
}

# Apply outlier handling to 'views'
cat("Handling outliers in 'views'...\n")
filtered_data <- handle_outliers(filtered_data, "views")

# Recalculate log_views after outlier handling
filtered_data[, log_views := log1p(views)]

# Visualize post-processing
ggplot(filtered_data, aes(x = views)) +
  geom_histogram(bins = 50, fill = "green", alpha = 0.7) +
  scale_x_log10() +
  theme_minimal() +
  labs(title = "Distribution of Views (After Outlier Removal)", x = "Views (log scale)", y = "Frequency")

# ---------------------
# 11. Feature Selection for Modeling
# ---------------------
cat("Selecting features for modeling...\n")
model_data <- filtered_data %>%
  select(log_views, song_age, num_collaborators, word_count, sentiment_afinn, 
         word_sentiment_interaction, collab_sentiment_interaction, starts_with("tag_"), starts_with("lyrics_dim_"))

# Handle any remaining missing values (if any)
model_data <- model_data %>%
  mutate(across(everything(), ~ ifelse(is.na(.), 0, .)))

# Release memory
rm(filtered_data)
gc()

# ---------------------
# 12. Train-Test Split
# ---------------------
cat("Splitting data into training and testing sets...\n")
train_indices <- createDataPartition(model_data$log_views, p = 0.7, list = FALSE)
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

# ---------------------
# 13. Modeling
# ---------------------
cat("Building models...\n")

# ----- Baseline Model: Linear Regression -----
cat("Training Linear Regression model...\n")
lm_model <- lm(log_views ~ ., data = train_data)
summary(lm_model)

# Evaluate Linear Regression
cat("Evaluating Linear Regression model...\n")
lm_predictions <- predict(lm_model, test_data)
lm_rmse <- rmse(test_data$log_views, lm_predictions)
lm_mae <- mae(test_data$log_views, lm_predictions)
lm_r2 <- R2(lm_predictions, test_data$log_views)
cat("Linear Regression - RMSE:", lm_rmse, "MAE:", lm_mae, "R-squared:", lm_r2, "\n")

# ----- Advanced Model: Random Forest -----
cat("Training Random Forest model...\n")
rf_model <- randomForest(log_views ~ ., data = train_data, ntree = 100, mtry = 2, importance = TRUE)
rf_predictions <- predict(rf_model, test_data)
rf_rmse <- rmse(test_data$log_views, rf_predictions)
rf_mae <- mae(test_data$log_views, rf_predictions)
rf_r2 <- R2(rf_predictions, test_data$log_views)
cat("Random Forest - RMSE:", rf_rmse, "MAE:", rf_mae, "R-squared:", rf_r2, "\n")

# ----- Advanced Model: XGBoost -----
cat("Training XGBoost model...\n")
train_matrix <- as.matrix(train_data[, -1])  # Exclude target variable
test_matrix <- as.matrix(test_data[, -1])
dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$log_views)
dtest <- xgb.DMatrix(data = test_matrix)

xgb_params <- list(
  objective = "reg:squarederror",
  eta = 0.1,  # Learning rate
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgboost(data = dtrain, params = xgb_params, nrounds = 100, verbose = 0)
xgb_predictions <- predict(xgb_model, dtest)
xgb_rmse <- rmse(test_data$log_views, xgb_predictions)
xgb_mae <- mae(test_data$log_views, xgb_predictions)
xgb_r2 <- R2(xgb_predictions, test_data$log_views)
cat("XGBoost - RMSE:", xgb_rmse, "MAE:", xgb_mae, "R-squared:", xgb_r2, "\n")

# ---------------------
# 14. Original Scale Evaluation
# ---------------------
cat("Evaluating models on the original scale...\n")
rf_predictions_original <- expm1(rf_predictions)
xgb_predictions_original <- expm1(xgb_predictions)
test_views_original <- expm1(test_data$log_views)

rf_rmse_original <- rmse(test_views_original, rf_predictions_original)
xgb_rmse_original <- rmse(test_views_original, xgb_predictions_original)
cat("Random Forest RMSE (Original Scale):", rf_rmse_original, "\n")
cat("XGBoost RMSE (Original Scale):", xgb_rmse_original, "\n")

# ---------------------
# 15. Visualizations
# ---------------------
cat("Generating visualizations...\n")

# a. Actual vs. Predicted for Random Forest
ggplot(data = data.frame(actual = test_views_original, predicted = rf_predictions_original),
       aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  theme_minimal() +
  labs(title = "Random Forest: Actual vs. Predicted Views", x = "Actual Views", y = "Predicted Views")

# b. Actual vs. Predicted for XGBoost
ggplot(data = data.frame(actual = test_views_original, predicted = xgb_predictions_original),
       aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  theme_minimal() +
  labs(title = "XGBoost: Actual vs. Predicted Views", x = "Actual Views", y = "Predicted Views")

# c. Feature Importance for Random Forest
varImpPlot(rf_model, main = "Feature Importance: Random Forest")

# d. Feature Importance for XGBoost
xgb_importance <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)
xgb.plot.importance(xgb_importance, main = "Feature Importance: XGBoost")

