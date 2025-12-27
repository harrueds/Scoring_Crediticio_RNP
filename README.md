# Scoring Crediticio con Redes Neuronales Profundas

## 1. Descripción General

Este proyecto desarrolla un modelo de **scoring crediticio** utilizando **redes neuronales profundas (Deep Neural Networks, DNN)** para estimar la probabilidad de incumplimiento de pago de clientes.  
El trabajo se enmarca en la aplicación de técnicas de *machine learning supervisado* a problemas financieros reales, considerando el tratamiento de datos estructurados y el manejo de desbalance de clases, común en escenarios crediticios.

El notebook `scoring_credito_RNP.ipynb` presenta el proceso completo desde la preparación de los datos hasta la evaluación final del modelo, integrando tanto aspectos técnicos como interpretativos del aprendizaje profundo.

---

## 2. Objetivos del Proyecto

- Implementar un flujo de trabajo reproducible para un problema de clasificación binaria aplicado al riesgo crediticio.  
- Comparar métricas de rendimiento y estabilidad del modelo mediante validación y curvas ROC.  
- Analizar la influencia de las variables predictoras mediante técnicas de **interpretabilidad (SHAP)**.  
- Promover buenas prácticas en el preprocesamiento de datos, balanceo de clases y diseño de redes neuronales densas.  

---

## 3. Tecnologías y Librerías Utilizadas

### **1. Manipulación de Datos**

- **NumPy (`numpy`)**: cálculos vectorizados y manejo eficiente de matrices.  
- **Pandas (`pandas`)**: limpieza, exploración y transformación de datos tabulares.

### **2. Visualización**

- **Matplotlib (`matplotlib.pyplot`)**: gráficos estáticos y curvas de aprendizaje.

### **3. Scikit-learn: Preprocesamiento y Evaluación**

- `train_test_split`: división de datos.  
- `OneHotEncoder`, `StandardScaler`: codificación y normalización de variables.  
- `ColumnTransformer`: combinación de transformaciones heterogéneas.  
- `compute_class_weight`: ajuste de pesos de clase para datos desbalanceados.  
- Métricas: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `RocCurveDisplay`.

### **4. TensorFlow / Keras: Modelado de Redes Neuronales**

- **TensorFlow (`tensorflow`)**: framework principal para deep learning.  
- **Keras (`tensorflow.keras`)**: construcción modular del modelo con:
  - `layers`: definición de arquitectura densa.  
  - `regularizers`: control del sobreajuste (L1, L2).  
  - `callbacks`: estrategias de entrenamiento (`EarlyStopping`, `ReduceLROnPlateau`).  
  - `Model`: creación y compilación del modelo personalizado.

### **5. Interpretabilidad**

- **SHAP (`shap`)**: interpretación de la contribución de cada variable al resultado final del modelo.

### **6. Otros**

- **Warnings**: silenciado controlado para mejorar la legibilidad de los resultados.

---

## 4. Estructura del Notebook

1. **Carga y exploración del dataset**  
   - Revisión de valores nulos, tipos de variables y distribución de clases.  
2. **Preprocesamiento de datos**  
   - Codificación de variables categóricas y estandarización de variables numéricas.  
   - División en conjuntos de entrenamiento y prueba.  
3. **Construcción del modelo de red neuronal**  
   - Arquitectura secuencial densa con regularización y funciones de activación ReLU / Sigmoid.  
   - Optimización con *Adam* y monitoreo mediante *EarlyStopping*.  
4. **Evaluación del modelo**  
   - Cálculo de métricas de desempeño y visualización de la curva ROC.  
   - Generación de matriz de confusión.  
5. **Interpretabilidad**  
   - Análisis de valores SHAP para comprender el aporte de las variables al score.  
6. **Conclusiones**  
   - Discusión sobre desempeño, sesgo, estabilidad y capacidad de generalización del modelo.

---

## 5. Resultados Principales

El modelo de red neuronal alcanzó métricas satisfactorias de **precisión y sensibilidad**, mostrando una adecuada capacidad para discriminar entre clientes con y sin riesgo de impago.  
El uso de **pesos de clase balanceados** permitió mitigar el sesgo hacia la clase mayoritaria, mientras que la regularización L2 y la detención temprana evitaron el sobreajuste.

El análisis con **SHAP** reveló las variables más influyentes en la predicción, reforzando la interpretabilidad del modelo, un aspecto clave en aplicaciones financieras.

---

## 6. Reflexión Personal

Durante el desarrollo de este proyecto aprendí a diseñar un flujo completo de *machine learning* orientado al sector financiero, comprendiendo las particularidades de los problemas de desbalance y la necesidad de transparencia en modelos predictivos.  
El uso de **TensorFlow/Keras** me permitió dominar la creación de arquitecturas densas y la configuración de hiperparámetros, mientras que el empleo de **SHAP** reforzó la importancia de la interpretabilidad en contextos sensibles como el crédito.

Este trabajo consolida la relación entre el análisis estadístico, la ingeniería de datos y el aprendizaje profundo, integrando teoría y práctica de forma aplicada.

---

## 7. Ejecución del Notebook en Entorno Local

### 🔹 Requisitos Previos

- Python **3.10+**
- Jupyter Notebook o JupyterLab
- pip y venv (para entornos virtuales)

### 🔹 Instalación del Entorno

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno (Linux / macOS)
source venv/bin/activate

# Activar entorno (Windows)
venv\Scripts\activate
```

### 🔹Ejecución

- Abrir Jupyter Notebook:

```bash
jupyter notebook
```

- Navegar hasta el archivo scoring_credito_RNP.ipynb y abrirlo.

- Ejecutar las celdas en orden para reproducir el flujo completo del modelo.

---

### 🔹 Autor

- Henzo Alejandro Arrué Muñoz
- Desarrollador con enfoque en Machine Learning y Linux con sólida formación matemática y estadística

### 🔹 Contacto

- Correo: [harrueds@gmail.com](mailto:harrueds@gmail.com)

- GitHub: [harrueds](https://github.com/harrueds)
