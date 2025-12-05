# --- 1. Instalación y Carga de Librerías ---

# Instalar si es necesario (descomenta si no las tienes)
# install.packages("e1071")
# install.packages("readr")
# install.packages("caret") 
# install.packages("nnet") 

library(e1071)
library(readr)
library(caret)
library(nnet) 


# --- 2. Carga y División de los Datos ---
TRAIN_FILE_PATH <- "C:/Users/Usuario/Desktop/Uni/APRENDIZAJE COMPUTACIONAL/Practica en grupo digit/Dataset_digit/train.csv"

# Cargar el conjunto de datos de entrenamiento (train.csv)
train_data <- read_csv(TRAIN_FILE_PATH)

# Usamos una muestra pequeña (1000 filas)
set.seed(42) 
sample_size <- 10000
full_sample <- train_data[1:sample_size, ]

# División en Conjunto de Entrenamiento (90%) y Prueba/Validación (10%)
train_index <- createDataPartition(full_sample$label, p = 0.9, list = FALSE)
train_set <- full_sample[train_index, ] 
test_set <- full_sample[-train_index, ] 

cat(paste0("Muestras de Entrenamiento para PCA/Tune: ", nrow(train_set), "\n"))
cat(paste0("Muestras de Prueba/Validación: ", nrow(test_set), "\n"))

# Preparación del conjunto de Entrenamiento
X_train <- train_set[, -1] 
# CORRECCIÓN CLAVE: Renombrar etiquetas a C0, C1, etc. para evitar error de 'caret'
y_train <- as.factor(paste0("C", train_set$label))
X_train_norm <- X_train / 255.0

# Preparación del conjunto de Prueba/Validación
X_test <- test_set[, -1]
y_test <- as.factor(paste0("C", test_set$label))
X_test_norm <- X_test / 255.0


# --- 3. Implementación del PCA ---

cat("\nIniciando PCA y determinando el número de componentes (k)...\n")
pca_model <- prcomp(X_train_norm, center = TRUE, scale. = FALSE) 

# Determinar k para el 90% de la varianza explicada
variance_explained <- (pca_model$sdev^2) / sum(pca_model$sdev^2)
cumulative_variance <- cumsum(variance_explained)
k <- which(cumulative_variance >= 0.90)[1] 

cat(paste0("Se usarán ", k, " componentes principales para el entrenamiento.\n"))

# Transformar AMBOS conjuntos
X_train_pca <- as.data.frame(pca_model$x[, 1:k])
X_test_pca_projection <- predict(pca_model, newdata = X_test_norm)
X_test_pca <- as.data.frame(X_test_pca_projection[, 1:k])

# Crear el data frame combinado para 'caret'
data_pca <- data.frame(label = y_train, X_train_pca)


# ----------------------------------------------------------------------
# --- 4. Optimización de Hiperparámetros con 'caret::train' para nnet ---
# ----------------------------------------------------------------------

cat("\nIniciando la optimización de parámetros (caret::train) para Red Neuronal (nnet)...\n")
cat("Usando 5-Fold Cross-Validation.\n")

# 4.1. Definir los parámetros a tunear (size y decay)
tune_grid <- expand.grid(
  .size = c(3, 8, 15), 
  .decay = c(0.001, 0.01) 
)

# 4.2. Definir el control de entrenamiento (Validación Cruzada)
train_control <- trainControl(
  method = "cv", 
  number = 5, # 5-Fold Cross-Validation
  verboseIter = FALSE,
  classProbs = TRUE 
)

# 4.3. Entrenar y tunear la Red Neuronal
nnet_train_results <- train(
  label ~ ., 
  data = data_pca, 
  method = "nnet", 
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "Accuracy", 
  maxit = 200, 
  trace = FALSE 
)

cat("Optimización completada.\n")

# Extraer los parámetros óptimos
best_size <- nnet_train_results$bestTune$size
best_decay <- nnet_train_results$bestTune$decay
best_model_accuracy <- max(nnet_train_results$results$Accuracy)

cat(paste0("\n--- Parámetros Óptimos Encontrados (Red Neuronal) ---\n"))
cat(paste0("Mejor size (Neuronas): ", best_size, "\n"))
cat(paste0("Mejor decay (Reg.): ", best_decay, "\n"))
cat(paste0("Accuracy de Validación Cruzada (CV): ", round(best_model_accuracy * 100, 2), "%\n"))
cat("--------------------------------------\n")


# ----------------------------------------------------------------------
# --- 5. Evaluación del Modelo Final en el Conjunto de PRUEBA/VALIDACIÓN ---
# ----------------------------------------------------------------------

cat("\n--- Evaluación en el Conjunto de PRUEBA (Red Neuronal) ---\n")

# El mejor modelo ya está entrenado en nnet_train_results
nnet_final_model <- nnet_train_results 

# Realizar predicciones
predictions_final_nnet <- predict(nnet_final_model, X_test_pca)

# Calcular la Accuracy REAL
accuracy_final_nnet <- mean(predictions_final_nnet == y_test) * 100

cat(paste0("Accuracy FINAL (Red Neuronal) en PRUEBA (10%): ", round(accuracy_final_nnet, 2), "%\n"))
cat("----------------------------------------------------------------\n")

# Generar la Matriz de Confusión en el conjunto de PRUEBA
confusion_matrix_final_nnet <- confusionMatrix(predictions_final_nnet, y_test)
print(confusion_matrix_final_nnet)