from network import Network
from layers import FCLayer, ActivationLayer, PoolingLayer, FlattenLayer, DropoutLayer
from activations import relu, relu_prime, sigmoid, sigmoid_prime
from losses import bce, bce_prime
from data import load_data
import numpy as np


# Carga de datos
X_train, X_test, y_train, y_test, y_train_value = load_data()

# Crear el modelo
model = Network()
lambda_l1 = 0.0015

# Añadir capas al modelo
#model.add(PoolingLayer(pool_size=2, stride=2, mode='max'))
#model.add(FlattenLayer())
model.add(FCLayer(196, 256, lambda_l1=lambda_l1))
model.add(ActivationLayer(relu, relu_prime))
model.add(DropoutLayer(0.2))
model.add(FCLayer(256, 256))
model.add(ActivationLayer(sigmoid, sigmoid_prime))
model.add(DropoutLayer(0.2))
model.add(FCLayer(256, 10))
model.add(ActivationLayer(sigmoid, sigmoid_prime))

# Compilar y entrenar el modelo
model.use(bce, bce_prime)
model.fit(X_train, y_train, epochs=5, learning_rate=0.1, batch_size=32)

# Evaluación del modelo
from sklearn.metrics import confusion_matrix, accuracy_score

y_hat = model.predict(X_test)
for i in range(len(y_hat)):
    y_hat[i] = np.argmax(y_hat[i])

matriz_conf = confusion_matrix(y_test, y_hat)
print('\nMATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf)
print('La exactitud de testeo del modelo ANN es: {:.3f}'.format(accuracy_score(y_test, y_hat)))

y_hat_train = model.predict(X_train)
for i in range(len(y_hat_train)):
    y_hat_train[i] = np.argmax(y_hat_train[i])

matriz_conf_train = confusion_matrix(y_train_value, y_hat_train)
print('\nMATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf_train)
print('La exactitud de ENTRENAMIENTO del modelo ANN es: {:.3f}'.format(accuracy_score(y_train_value, y_hat_train)))


