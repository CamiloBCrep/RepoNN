import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

# Derivada aproximada de la función de activación ReLU
def relu_prime(x):
    # Usamos una aproximación para manejar el punto no derivable
    return np.where(x > 0, 1, 0)  # 1 para x > 0, 0 para x <= 0