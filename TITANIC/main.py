from network import Network
from layers import FCLayer, ActivationLayer, DropoutLayer
from activations import sigmoid, sigmoid_prime
from losses import bce, bce_prime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Carga de datos
data = pd.read_csv('Titanic.csv')

# Eliminar columnas innecesarias
data = data.drop(columns=['Name', 'Ticket', 'Cabin'], errors='ignore')

# Convertir la columna 'Sex' en variables numéricas (0 y 1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Llenar valores nulos (por ejemplo, con la media de la edad)
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Eliminar filas con valores nulos en columnas clave
data.dropna(subset=['Embarked'], inplace=True)

# Convertir la columna 'Embarked' en variables numéricas
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Seleccionar características (features) y etiquetas (labels)
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = data['Survived'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear la red neuronal
lambda_l1 = 0.0001

model = Network()

model.add(FCLayer(input_size=X_train.shape[1], output_size=100, lambda_l1=lambda_l1))  # Capa oculta
model.add(ActivationLayer(sigmoid, sigmoid_prime))
model.add(DropoutLayer(0.2))
model.add(FCLayer(input_size=100, output_size=1))  # Capa de salida
model.add(ActivationLayer(sigmoid, sigmoid_prime))  # Usando sigmoid para salida binaria

# Usar Binary Cross-Entropy como función de pérdida
model.use(bce, bce_prime)

# Entrenar la red
model.fit(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)

# Hacer predicciones para el conjunto de test
y_hat = model.predict(X_test)
# Convertir las predicciones a clases
y_hat = np.array([1 if output > 0.5 else 0 for output in y_hat]).flatten()

# Calcular la matriz de confusión y la exactitud
matriz_conf = confusion_matrix(y_test, y_hat)
print('\nMATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf)
print('La exactitud de testeo del modelo ANN es: {:.3f}'.format(accuracy_score(y_test, y_hat)))

# Hacer predicciones para el conjunto de entrenamiento
y_hat_train = model.predict(X_train)
# Convertir las predicciones a clases
y_hat_train = np.array([1 if output > 0.5 else 0 for output in y_hat_train]).flatten()

# Asegúrate de que y_train_value sea un arreglo plano
y_train_value = y_train.flatten() if y_train.ndim > 1 else y_train

# Calcular la matriz de confusión y la exactitud
matriz_conf_train = confusion_matrix(y_train_value, y_hat_train)
print('\nMATRIZ DE CONFUSIÓN para modelo ANN')
print(matriz_conf_train)
print('La exactitud de ENTRENAMIENTO del modelo ANN es: {:.3f}'.format(accuracy_score(y_train_value, y_hat_train)))