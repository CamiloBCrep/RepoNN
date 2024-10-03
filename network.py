import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        input_data = np.array([[x] for x in input_data])
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size):
        x_train = np.array([[x] for x in x_train])
        #y_train = np.array(y_train)
        samples = len(x_train)

        for i in range(epochs):
            total_error = 0

            # Shuffling data
            indices = np.arange(samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for start in range(0, samples, batch_size):
                end = min(start + batch_size, samples)
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]

                cumulative_error = 0

                for j in range(len(batch_x)):
                    output = batch_x[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # Sumar el error de la predicción
                    cumulative_error += self.loss(batch_y[j], output)

                    # Calcular el error para la retropropagación
                    error = self.loss_prime(batch_y[j], output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(error, learning_rate)

                # Calcular el error promedio del mini lote
                cumulative_error /= len(batch_x)
                total_error += cumulative_error

            # Calcular el error promedio total para la época
            avg_error = total_error / (samples // batch_size)  # Promedio de errores por mini lote
            avg_error = np.mean(avg_error)# convierte a escalar si es un arreglo

            print('epoch %d/%d   error=%f' % (i + 1, epochs, avg_error))
