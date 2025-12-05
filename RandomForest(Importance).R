# --- 1. Instalación y Carga de Librerías ---

# install.packages("e1071")
# install.packages("readr")
# install.packages("caret") 
# install.packages("randomForest") 

library(e1071)
library(readr)
library(caret)
library(randomForest)

inicio_global <- Sys.time()


# --- 2. Carga y División de los Datos ---

# Ajusta la ruta a tu archivo
train_data <- read_csv("C:/Users/danie/OneDrive/Escritorio/Danii/Uni/4º Año/Aprendizaje Computacional/Practica (Digit Recognition)/train.csv")

# Usamos una muestra pequeña (1000 filas) para la demostración
set.seed(123) 
sample_size <- 5000
full_sample <- train_data[1:sample_size, ]

# División en Conjunto de Entrenamiento (90%) y Prueba/Validación (10%)
train_index <- createDataPartition(full_sample$label, p = 0.9, list = FALSE)
train_set <- full_sample[train_index, ] 
test_set <- full_sample[-train_index, ] 

cat(paste0("Muestras de Entrenamiento: ", nrow(train_set), "\n"))
cat(paste0("Muestras de Prueba/Validación: ", nrow(test_set), "\n"))

# Preparación de las variables (Normalización)
X_train <- train_set[, -1] 
y_train <- as.factor(train_set$label)
X_train_norm <- X_train / 255.0

X_test <- test_set[, -1]
y_test <- as.factor(test_set$label)
X_test_norm <- X_test / 255.0


# ----------------------------------------------------------------------
# --- 3. Selección de Características por IMPORTANCE (Reemplaza a PCA) ---
# ----------------------------------------------------------------------

cat("\nIniciando selección de variables mediante Importance...\n")
cat("1. Entrenando RF preliminar para calcular importancia de píxeles...\n")

# Entrenamos un RF rápido (menos árboles) solo para medir qué píxeles importan
rf_preliminar <- randomForest(
  x = X_train_norm,
  y = y_train,
  ntree = 100,      # Pocos árboles para que sea rápido
  importance = TRUE # Fundamental para obtener las métricas
)

# Extraer la importancia (MeanDecreaseGini es robusto para clasificación)
importancia <- importance(rf_preliminar)
var_importance_scores <- importancia[, "MeanDecreaseGini"]

# Seleccionar los TOP N píxeles más importantes
# (Por ejemplo, nos quedamos con los 100 mejores para reducir dimensión)
num_vars_to_keep <- 100 
vars_seleccionadas <- names(sort(var_importance_scores, decreasing = TRUE))[1:num_vars_to_keep]

cat(paste0("Se han seleccionado las ", num_vars_to_keep, " variables (píxeles) más importantes.\n"))

# Crear los nuevos conjuntos de datos REDUCIDOS (Solo con las columnas seleccionadas)
X_train_reduced <- X_train_norm[, vars_seleccionadas]
data_reduced <- data.frame(label = y_train, X_train_reduced) # Dataframe para 'tune'

X_test_reduced <- X_test_norm[, vars_seleccionadas] # Aplicar misma selección al test


# ----------------------------------------------------------------------
# --- 4. Optimización de Hiperparámetros con 'tune' para Random Forest ---
# ----------------------------------------------------------------------

cat("\nIniciando la optimización de parámetros (tune) sobre datos reducidos...\n")

# mtry: Número de variables a muestrear en cada división.
# Regla: sqrt(variables). Variables = 100 -> sqrt(100) = 10.
# Probamos valores alrededor de 10.
mtry_values <- c(5, 8, 10, 12, 15) 

tune_rf_results <- tune(
  randomForest, 
  train.x = label ~ ., 
  data = data_reduced, # Usamos el dataset reducido por importance
  ranges = list(
    mtry = mtry_values 
  ),
  ntree = 100 # Reducido para que el tune sea rápido en la prueba
)

cat("Optimización completada.\n")

best_mtry <- tune_rf_results$best.parameters$mtry
best_model_error <- tune_rf_results$best.performance

cat(paste0("\n--- Parámetros Óptimos Encontrados ---\n"))
cat(paste0("Mejor mtry: ", best_mtry, "\n"))
cat(paste0("Error CV: ", round(best_model_error, 4), "\n"))


# -------------------------------------------------------------------
# --- 5. Entrenamiento del Modelo Final con Parámetros Óptimos ---
# -------------------------------------------------------------------

cat("\nEntrenando el modelo Random Forest final con las variables seleccionadas...\n")

rf_final_model <- randomForest(
  label ~ .,
  data = data_reduced,
  mtry = best_mtry, 
  ntree = 500, # Más árboles para el modelo final
  importance = TRUE 
)

cat("Modelo final Random Forest entrenado.\n")


# ----------------------------------------------------------------------
# --- 6. Evaluación del Modelo Final en el Conjunto de PRUEBA ---
# ----------------------------------------------------------------------

cat("\n--- Evaluación en el Conjunto de PRUEBA (Random Forest + Importance) ---\n")

# Predecir usando el set de prueba reducido (solo las 100 columnas importantes)
predictions_final_rf <- predict(rf_final_model, X_test_reduced)

# Calcular Accuracy
accuracy_final_rf <- mean(predictions_final_rf == y_test) * 100

cat(paste0("Accuracy FINAL en PRUEBA: ", round(accuracy_final_rf, 2), "%\n"))
cat("----------------------------------------------------------------\n")

# Matriz de Confusión
confusion_matrix_final_rf <- confusionMatrix(predictions_final_rf, y_test)
print(confusion_matrix_final_rf)


fin_global <- Sys.time()
tiempo_total <- fin_global - inicio_global

cat("\n----------------------------------------------------\n")
cat("Tiempo TOTAL de ejecución: ")
print(tiempo_total)
cat("----------------------------------------------------\n")


save(rf_final_model, file = "model_RandomForest.RData")
load("model_RandomForest.RData")