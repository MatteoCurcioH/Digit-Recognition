# --- 1. Instalación y Carga de Librerías ---

# Instalar si es necesario (descomenta si no las tienes)
# install.packages("e1071")
# install.packages("readr")
# install.packages("caret") 

library(e1071)
library(readr)
library(caret)

# --- 2. Carga y División de los Datos ---

# Cargar el conjunto de datos de entrenamiento (train.csv)
train_data <- read_csv("C:/Users/Usuario/Desktop/Uni/APRENDIZAJE COMPUTACIONAL/Practica en grupo digit/Dataset_digit/train.csv")

# Usamos una muestra pequeña (1000 filas) para la demostración
set.seed(42) 
sample_size <- 10000
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


# --- 4. Optimización de Hiperparámetros con 'tune.svm' ---

cat("\nIniciando la optimización de parámetros (tune.svm) con 10-fold CV...\n")


# --- 5. Entrenamiento del Modelo Final con Parámetros Óptimos ---

cat("Entrenando el modelo SVM final con los parámetros óptimos...\n")

# Entrenar el modelo final usando TODOS los datos de entrenamiento (80%)
svm_final_model <- svm(
  label ~ .,
  data = data_pca,
  kernel = "radial",
  cost = 1, 
  gamma = 0.01
)

cat("Modelo final entrenado.\n")


# --- 6. Evaluación del Modelo Final en el Conjunto de PRUEBA/VALIDACIÓN ---

cat("\n--- Evaluación en el Conjunto de PRUEBA (Datos NO Vistos) ---\n")

# **NUEVO:** Realizar predicciones sobre el conjunto de prueba (X_test_pca)
predictions_final <- predict(svm_final_model, X_test_pca)

# Calcular la Accuracy REAL
accuracy_final <- mean(predictions_final == y_test) * 100

cat(paste0("Accuracy FINAL en el conjunto de PRUEBA (20%): ", round(accuracy_final, 2), "%\n"))
cat("----------------------------------------------------------------\n")

# Generar la Matriz de Confusión en el conjunto de PRUEBA
confusion_matrix_final <- confusionMatrix(predictions_final, y_test)
print(confusion_matrix_final)