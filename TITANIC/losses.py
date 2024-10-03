import numpy as np

def mse(y_real, y_hat):
    return (y_real - y_hat) ** 2

def mse_prime(y_real, y_hat):
    return 2 * (y_hat - y_real)
def bce(y_real, y_hat):
    # Evitamos log(0) añadiendo un pequeño valor epsilon
    epsilon = 1e-8
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # Limita valores a [epsilon, 1 - epsilon]

    # Calculamos la pérdida usando BCE
    bce_loss = -np.mean(y_real * np.log(y_hat) + (1 - y_real) * np.log(1 - y_hat))
    return bce_loss

# Derivada de la Binary Cross-Entropy (BCE)
def bce_prime(y_real, y_hat):
    # Evitamos log(0) añadiendo un pequeño valor epsilon
    epsilon = 1e-8
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    # Derivada de la pérdida BCE
    bce_prime = (y_hat - y_real) / (y_hat * (1 - y_hat))
    return bce_prime