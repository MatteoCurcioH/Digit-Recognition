library(e1071)
library(caret)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(readr)

# --- 0. CARGAR DATOS DESDE LOS CSV ---

cat("Cargando conjuntos de datos desde CSV...\n")

# 1. Cargar datos PCA (Para predecir)
test_pca_df <- read_csv("test_set_PCA.csv", show_col_types = FALSE)

# Separamos etiqueta y variables
# FORZAMOS que label sea factor con niveles 0-9 para evitar errores si faltan números
y_test <- factor(test_pca_df$label, levels = 0:9) 
X_test_pca <- as.data.frame(test_pca_df[, -1]) 

# 2. Cargar datos de PIXELES (Para dibujar)
test_pixels_df <- read_csv("test_set_PIXELS.csv", show_col_types = FALSE)
X_test_norm <- as.data.frame(test_pixels_df[, -1])

cat("Datos cargados. Dimensiones Test: ", nrow(X_test_pca), " filas.\n")


# --- 1. Cargar el Modelo ---

cat("Cargando modelo (model_svm.RData)...\n")
# Asegúrate de que el archivo está en la carpeta de trabajo
if(file.exists("model_svm.RData")) {
  load("model_svm.RData")
} else {
  stop("❌ ERROR: No se encuentra el archivo 'model_svm.RData'.")
}


# --- PARTE A: Análisis de Hiperparámetros (Heatmap) ---

if(exists("tune_results")) {
  cat("\n--- Generando Gráfico de Tuning ---\n")
  perf_data <- tune_results$performances
  
  g1 <- ggplot(perf_data, aes(x = factor(gamma), y = factor(cost), fill = error)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Mapa de Calor: Error de Validación",
         x = "Gamma", y = "Cost", fill = "Error") +
    theme_minimal()
  print(g1)
} else {
  cat("AVISO: 'tune_results' no encontrado. Saltando Parte A.\n")
}


# --- PARTE B: Evaluación y Matriz de Confusión ---

cat("\n--- Evaluando Modelo ---\n")

predicciones <- predict(svm_final_model, X_test_pca)

# CORRECCIÓN DE SEGURIDAD: Asegurar mismos niveles para evitar error 'Ops.factor'
predicciones <- factor(predicciones, levels = levels(y_test))

# 1. Matriz Numérica
cm <- confusionMatrix(predicciones, y_test)
print(cm$overall['Accuracy'])

# 2. Matriz Visual
cm_table <- as.data.frame(cm$table)
colnames(cm_table) <- c("Prediccion", "Realidad", "Frecuencia")

g2 <- ggplot(cm_table, aes(x = Realidad, y = Prediccion, fill = Frecuencia)) +
  geom_tile() +
  geom_text(aes(label = Frecuencia), color = "white", size = 3) +
  scale_fill_gradient(low = "blue", high = "orange") +
  labs(title = "Matriz de Confusión", subtitle = "Diagonal = Aciertos") +
  theme_minimal()
print(g2)


# --- PARTE C: Análisis de Errores (CON GIRO 90° DERECHA) ---

cat("\n--- Visualizando Errores ---\n")
indices_error <- which(predicciones != y_test)

if(length(indices_error) > 0) {
  num_plot <- min(9, length(indices_error))
  sample_errors <- indices_error[1:num_plot]
  
  par(mfrow = c(3, 3), mar = c(1,1,3,1)) # Grid 3x3
  
  for(idx in sample_errors) {
    vec <- as.numeric(X_test_norm[idx, ])
    
    # 1. Matriz base (R llena por columnas, la imagen sale tumbada)
    mat <- matrix(vec, nrow = 28, ncol = 28)
    
    # 2. ROTACIÓN CORRECTIVA (90 grados a la derecha)
    mat <- t(apply(mat, 2, rev))
    
    lbl_real <- as.character(y_test[idx])
    lbl_pred <- as.character(predicciones[idx])
    
    # Usamos gray(0:255 / 255) para blanco y negro estándar
    image(1:28, 1:28, mat, col = gray.colors(255), axes = FALSE, 
          main = paste0("Real: ", lbl_real, "\nPred: ", lbl_pred))
  }
  par(mfrow = c(1, 1)) # Resetear grid
} else {
  cat("¡Sin errores! (O algo falla en la comparación)\n")
}


# --- PARTE C.2: Galería de Aciertos (CON GIRO 90° DERECHA) ---

cat("\n--- Visualizando Aciertos ---\n")
indices_aciertos <- which(predicciones == y_test)

if(length(indices_aciertos) > 0) {
  set.seed(99) 
  sample_correctos <- sample(indices_aciertos, min(9, length(indices_aciertos)))
  
  par(mfrow = c(3, 3), mar = c(1,1,3,1))
  
  for(idx in sample_correctos) {
    vec <- as.numeric(X_test_norm[idx, ])
    mat <- matrix(vec, nrow = 28, ncol = 28)
    
    # ROTACIÓN CORRECTIVA
    mat <- t(apply(mat, 2, rev))
    
    lbl_real <- as.character(y_test[idx])
    image(1:28, 1:28, mat, col = gray.colors(255), axes = FALSE, 
          main = paste0("✅ Correcto: ", lbl_real))
  }
  par(mfrow = c(1, 1)) # Resetear grid
}


# --- PARTE D: Gráfico F1 Score ---

metrics_df <- data.frame(
  Clase = gsub("Class: ", "", rownames(cm$byClass)),
  F1_Score = cm$byClass[, "F1"]
)

# Manejo de NA si alguna clase no tiene predicciones
metrics_df$F1_Score[is.na(metrics_df$F1_Score)] <- 0

g3 <- ggplot(metrics_df, aes(x = Clase, y = F1_Score, fill = F1_Score)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "red", high = "green") +
  coord_cartesian(ylim = c(0.7, 1.0)) + # Zoom ajustado
  labs(title = "Calidad por Dígito (F1 Score)", y = "F1 Score") +
  theme_minimal()

print(g3)
