library(e1071)
library(nnet)
library(randomForest)
library(caret)

# --- Función para predecir con tu Ensemble Guardado ---
predecir_con_ensemble <- function(archivo_rds, nuevos_datos_raw) {
  
  # A. Cargar el "super objeto"
  if (is.character(archivo_rds)) {
    modelo <- readRDS(archivo_rds)
  } else {
    modelo <- archivo_rds # Por si le pasas el objeto directamente
  }
  
  # B. Preparar datos
  # 1. Normalizar (asumiendo pixeles 0-255)
  datos_norm <- nuevos_datos_raw / 255.0
  if(!is.data.frame(datos_norm)) datos_norm <- as.data.frame(datos_norm)
  
  # 2. Aplicar PCA (Para SVM y MLP)
  # Usamos el PCA guardado dentro de la lista
  proyeccion <- predict(modelo$pca_trans, newdata = datos_norm)
  datos_pca  <- as.data.frame(proyeccion[, 1:modelo$k_comps])
  
  
  # C. Predicciones Individuales
  
  # 1. SVM (Usa PCA)
  p_svm <- predict(modelo$svm_brain, datos_pca)
  
  # 2. MLP (Usa PCA + Corrección de niveles)
  p_mlp_prob <- predict(modelo$mlp_brain, datos_pca, type = "raw")
  p_mlp_idx  <- max.col(p_mlp_prob)
  # Usamos los niveles guardados para mapear correctamente
  p_mlp <- factor(modelo$niveles[p_mlp_idx], levels = modelo$niveles)
  
  # 3. Random Forest (Usa Datos RAW Normalizados, SIN PCA)
  p_rf <- predict(modelo$rf_brain, newdata = datos_norm, type = "class")
  
  
  # D. Votación (Hard Voting)
  preds_df <- data.frame(
    SVM = as.character(p_svm),
    MLP = as.character(p_mlp),
    RF  = as.character(p_rf),
    stringsAsFactors = FALSE
  )
  
  get_mode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  voto_final <- apply(preds_df, 1, get_mode)
  return(factor(voto_final, levels = modelo$niveles))
}

data <- read.csv("train.csv")


set.seed(42) 
#sample_size <- 0000
#full_sample <- data[1:sample_size, ]
full_sample  <- data # Esto si queremos probar sobre todo el dataset
full_sample$label <- as.factor(full_sample$label)

indexTrain <- createDataPartition(full_sample$label, p = 0.3, list=FALSE)
trainData  <- full_sample[indexTrain,]
testData   <- full_sample[-indexTrain,]

predicciones_finales <- predecir_con_ensemble("mi_ensemble_votos.rds", testData)

predicciones_finales
testData$label
levels(predicciones_finales)
levels(testData$label)

cm <- confusionMatrix(predicciones_finales, testData$label)
cm
 

