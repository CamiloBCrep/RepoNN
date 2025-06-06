# Neural Network from Scratch

Este proyecto implementa una red neuronal artificial (ANN) desde cero, sin usar frameworks de machine learning como TensorFlow o PyTorch. Está estructurado en archivos modulares para facilitar la mantenibilidad y la reutilización del código.

## Estructura del Proyecto
```plaintext
my_neural_network/
├── layers.py # Capas de la red (densa, activación, pooling, etc.)
├── network.py # Clase principal Network (entrenamiento y predicción)
├── activations.py # Funciones de activación y sus derivadas
├── losses.py # Funciones de pérdida
├── utils.py # Funciones auxiliares
├── data.py # Carga y preprocesamiento de datos (MNIST o Titanic)
└── main.py # Script principal para construir, entrenar y evaluar
```

## Características

- Implementación propia de:
  - Capas densas (FCLayer)
  - Capas de activación (sigmoid, relu, tanh)
  - Capa de dropout
  - Capa de pooling (opcional, útil para imágenes)
  - Capa de flatten
  - Capas convolucionales (no funcional)
- Funciones de pérdida:
  - MSE (error cuadrático medio)
  - BCE (binary cross-entropy)
- Entrenamiento por lotes, retropropagación y optimización básica
- Compatible con datasets numéricos (como Titanic) e imágenes (como MNIST)

## Ejecución

Desde la carpeta del proyecto, ejecuta:

```bash
python main.py
```
Dependencia necesarias
```bash
pip install numpy scikit-learn keras
pip install tensorflow
```
## Uso con MNIST
El archivo `data.py` puede cargar y procesar imágenes de dígitos MNIST. El modelo incluirá capas de pooling, flatten, etc.

Ejemplo en `main.py`:
```bash
model.add(PoolingLayer(pool_size=2, stride=2, mode='max'))
model.add(FlattenLayer())
model.add(FCLayer(196, 256))
```
##Uso con Titanic
Para datos tabulares como Titanic, el modelo se reduce a capas densas:
```bash
model.add(FCLayer(X_train.shape[1], 64))
model.add(ActivationLayer(relu, relu_prime))
model.add(FCLayer(64, 1))
model.add(ActivationLayer(sigmoid, sigmoid_prime))
```
Evaluación
El archivo `main.py` incluye métricas como matriz de confusión y exactitud (accuracy) para evaluar el rendimiento del modelo:
```bash
from sklearn.metrics import confusion_matrix, accuracy_score
```


