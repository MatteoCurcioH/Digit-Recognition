library(e1071)        # Para SVM
library(nnet)         # Para MLP
library(randomForest) # Para Random Forest
library(caret)        # Útil para matrices de confusión

inicio_global <- Sys.time()

# --- 2. Carga y División de los Datos ---

# Cargar el conjunto de datos de entrenamiento (train.csv)
train_data <- read_csv("train.csv")

# Usamos una muestra pequeña (1000 filas) para la demostración
set.seed(42) 
sample_size <- 5000
full_sample <- train_data[1:sample_size, ]

# **NUEVO:** División en Conjunto de Entrenamiento (90%) y Prueba/Validación (10%)
# El modelo será evaluado por primera vez en 'test_index'
train_index <- createDataPartition(full_sample$label, p = 0.9, list = FALSE)
train_set <- full_sample[train_index, ] # 800 muestras para entrenar y tunear
test_set <- full_sample[-train_index, ] # 200 muestras para evaluación final

cat(paste0("Muestras de Entrenamiento para PCA/Tune: ", nrow(train_set), "\n"))
cat(paste0("Muestras de Prueba/Validación: ", nrow(test_set), "\n"))

# Preparación del conjunto de Entrenamiento
X_train <- train_set[, -1] 
y_train <- as.factor(train_set$label)
X_train_norm <- X_train / 255.0

# Preparación del conjunto de Prueba/Validación
X_test <- test_set[, -1]
y_test <- as.factor(test_set$label)
X_test_norm <- X_test / 255.0


# --- 3. Implementación del PCA (Ajustado SOLO en el Conjunto de Entrenamiento) ---
cat("\nIniciando PCA y determinando el número de componentes (k)...\n")
# Ajustamos PCA solo al conjunto de entrenamiento
pca_model <- prcomp(X_train_norm, center = TRUE, scale. = FALSE) 

# Determinar k para el 90% de la varianza explicada
variance_explained <- (pca_model$sdev^2) / sum(pca_model$sdev^2)
cumulative_variance <- cumsum(variance_explained)
k <- which(cumulative_variance >= 0.90)[1] 

cat(paste0("Se usarán ", k, " componentes principales para el entrenamiento.\n"))

# Transformar AMBOS conjuntos usando el modelo PCA ajustado
X_train_pca <- as.data.frame(pca_model$x[, 1:k])
data_pca <- data.frame(label = y_train, X_train_pca)

# **NUEVO:** Transformar el conjunto de prueba/validación
X_test_pca_projection <- predict(pca_model, newdata = X_test_norm)
X_test_pca <- as.data.frame(X_test_pca_projection[, 1:k])

# --- 1. Cargar los Modelos ---

# Nota: Si usaste saveRDS, usa readRDS asignando a una variable.
# Si usaste save(), usa load("archivo.RData") y el objeto aparecerá con su nombre original.

cat("Cargando modelos...\n")
load("model_svm.RData") # Ajusta los nombres de archivo
load("modelo_mlp.RData")
load("model_RandomForest.RData")

svm_model <- svm_final_model
mlp_model <- nnet_final_model
rf_model  <- rf_final_model
# --- 2. Generar Predicciones Individuales ---

# IMPORTANTE: Asegúrate de que 'X_test_pca' (o tus datos de entrada)
# tengan exactamente las mismas columnas/PCA que usaste para entrenar.

cat("Generando predicciones individuales...\n")

# Predicción SVM
pred_svm <- predict(svm_model, X_test_pca, type = "class")

# Predicción Random Forest
pred_rf  <- predict(rf_model, X_test_pca, type = "class")

# Predicción MLP (nnet)
# nnet suele devolver probabilidades, así que extraemos la clase con mayor probabilidad
pred_mlp_prob <- predict(mlp_model, X_test_pca, type = "raw")
pred_mlp_idx  <- max.col(pred_mlp_prob) # Índice de la columna más alta
# Convertimos el índice a la etiqueta real (asumiendo que las columnas son las clases)
# Si tu target original era factor, nnet usa levels(y_train) como nombres de columna
labels_mlp <- colnames(pred_mlp_prob) 
pred_mlp   <- factor(labels_mlp[pred_mlp_idx], levels = levels(pred_svm))


# --- 3. Crear el Ensemble (Votación) ---

# Unimos todo en un Data Frame
preds_df <- data.frame(
  SVM = as.character(pred_svm),
  MLP = as.character(pred_mlp),
  RF  = as.character(pred_rf),
  stringsAsFactors = FALSE
)

# Función para calcular la moda (el valor más repetido en una fila)
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Aplicamos la votación fila por fila
cat("Calculando votación mayoritaria...\n")
ensemble_pred <- apply(preds_df, 1, get_mode)

# Convertimos de nuevo a Factor para poder evaluar
ensemble_pred <- factor(ensemble_pred, levels = levels(y_test))


# --- 4. Evaluar el Resultado ---

cat("\n--- Resultados del Ensemble ---\n")
accuracy_ensemble <- mean(ensemble_pred == y_test) * 100
cat(paste0("Accuracy Ensemble: ", round(accuracy_ensemble, 2), "%\n"))

# Ver matriz de confusión
print(confusionMatrix(ensemble_pred, y_test))