# --- 1. Instalación y Carga de Librerías ---

inicio_global <- Sys.time()

# install.packages("rpart")       # Librería para árboles CART
# install.packages("rpart.plot")  # Librería para visualizar el árbol
# install.packages("e1071")
# install.packages("readr")
# install.packages("caret") 

library(rpart)
library(rpart.plot) # Para dibujar el árbol
library(e1071)
library(readr)
library(caret)


# --- 2. Carga y Preparación de Datos ---

# Ajusta la ruta a tu archivo
train_data <- read_csv("C:/Users/danie/OneDrive/Escritorio/Danii/Uni/4º Año/Aprendizaje Computacional/Practica (Digit Recognition)/train.csv")

# Muestra reducida para demostración (usa todo el dataset para el modelo final)
set.seed(123) 
sample_size <- 5000 # Un poco más grande para que el árbol tenga "carne"
full_sample <- train_data[1:sample_size, ]

# División Train/Test
train_index <- createDataPartition(full_sample$label, p = 0.8, list = FALSE)
train_set <- full_sample[train_index, ] 
test_set <- full_sample[-train_index, ] 

# Separar X e y
# Nota: Para rpart, es importante que la etiqueta sea FACTOR
train_set$label <- as.factor(train_set$label)
test_set$label <- as.factor(test_set$label)


# ----------------------------------------------------------------------
# --- 3. PCA (Opcional, pero recomendado para reducir recursos) ---
# ----------------------------------------------------------------------
# CART puede trabajar con píxeles crudos, pero PCA ayuda a reducir 
# el tamaño del árbol y mejorar la velocidad (cumpliendo el objetivo de recursos).

cat("Aplicando PCA para reducción de dimensionalidad...\n")

# Quitamos la etiqueta para el PCA
X_train <- train_set[, -1] / 255.0
X_test <- test_set[, -1] / 255.0

pca_model <- prcomp(X_train, center = TRUE, scale. = FALSE)

# Nos quedamos con el 90% de varianza
variance_explained <- (pca_model$sdev^2) / sum(pca_model$sdev^2)
k <- which(cumsum(variance_explained) >= 0.90)[1]

cat(paste0("Componentes seleccionados (k): ", k, "\n"))

# Proyectar datos
X_train_pca <- data.frame(label = train_set$label, pca_model$x[, 1:k])
X_test_pca <- data.frame(predict(pca_model, newdata = X_test))[, 1:k]


# ----------------------------------------------------------------------
# --- 4. Optimización (Tune) del Parámetro de Complejidad (cp) ---
# ----------------------------------------------------------------------

cat("\nIniciando 'tune' para encontrar el mejor cp (Complexity Parameter)...\n")

# El parámetro cp controla el tamaño del árbol. 
# Si cp es muy bajo -> Overfitting (árbol gigante).
# Si cp es muy alto -> Underfitting (árbol muy simple).
cp_values <- c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1)

tune_cart <- tune.rpart(
  label ~ ., 
  data = X_train_pca,
  cp = cp_values
)

print(summary(tune_cart))
best_cp <- tune_cart$best.parameters$cp

cat(paste0("\nMejor cp encontrado: ", best_cp, "\n"))


# ----------------------------------------------------------------------
# --- 5. Entrenamiento del Modelo Final (CART) ---
# ----------------------------------------------------------------------

cat("\nEntrenando árbol final con el mejor cp...\n")

cart_model <- rpart(
  label ~ ., 
  data = X_train_pca,
  method = "class",   # Importante: "class" para clasificación
  cp = best_cp        # Usamos el cp optimizado
)

fin_train <- Sys.time()


# ----------------------------------------------------------------------
# --- 6. Evaluación y Visualización ---
# ----------------------------------------------------------------------

cat("\n--- Evaluación en Test ---\n")

# Predicción (type = "class" nos da la etiqueta directamente)
pred_cart <- predict(cart_model, newdata = X_test_pca, type = "class")

# Accuracy
acc <- mean(pred_cart == test_set$label)
cat(paste0("Accuracy CART: ", round(acc * 100, 2), "%\n"))

# Matriz de Confusión
print(confusionMatrix(pred_cart, test_set$label))

# Visualizar el árbol (Guardar como PDF o ver en plot)
# Esto es muy útil para el informe ("modelo interpretable")
rpart.plot(cart_model, extra = 104, box.palette = "GnBu", 
           branch.lty = 3, shadow.col = "gray", nn = TRUE, 
           main = "Árbol de Decisión (CART) sobre Componentes PCA")


fin_global <- Sys.time()
tiempo_total <- fin_global - inicio_global

cat("\n----------------------------------------------------\n")
cat("Tiempo TOTAL de ejecución: ")
print(tiempo_total)
cat("----------------------------------------------------\n")


save(cart_model, file = "model_CART.RData")
