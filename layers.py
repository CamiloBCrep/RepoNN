import numpy as np

# Clase base para Capa
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

# La capa de convolución hereda de la clase base, madre de todas las capas
class ConvLayer(Layer):
    def __init__(self, input_shape, kernel_shape, num_kernels, lambda_l1=0):
        self.lambda_l1 = lambda_l1
        self.input_shape = input_shape  # Tamaño de la entrada (alto, ancho, profundidad)
        self.input_depth = input_shape[2]  # Profundidad de la entrada (número de canales)
        self.kernel_shape = kernel_shape  # Tamaño del kernel (alto, ancho)
        self.num_kernels = num_kernels  # Número de kernels (cada uno genera una salida)
        # La salida tendrá el mismo tamaño de entrada
        # esto no tendría por qué ser así, pero nos ahorra problemas para manejar tamaños entre capas
        self.output_shape = (input_shape[0], input_shape[1], num_kernels) 
        # Inicializamos los pesos de forma aleatoria con rango entre -0.5 y 0.5
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, num_kernels) - 0.5
        self.bias = np.random.rand(num_kernels) - 0.5


    # Función para agregar padding a la entrada
    # es necesario para mantener el tamaño de la matriz
    def pad_input(self, input_matrix, pad):
        # Agregamos padding de ceros alrededor de la matriz de entrada
        return np.pad(input_matrix, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Correlación 2D
    # la parte previa a la convolución, que además sirve de convolución en fordward prop
    def correlate(self, input_matrix, kernel):
        # Obtenemos las dimensiones de entrada y del kernel
        input_height, input_width = input_matrix.shape
        kernel_height, kernel_width = kernel.shape
        # Calculamos las dimensiones de la salida tras la correlación
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        # Inicializamos la salida como una matriz de ceros
        output = np.zeros((output_height, output_width))

        # Recorremos la entrada y aplicamos la correlación
        for i in range(output_height):
            for j in range(output_width):
                # Las dos iteraciones controlan el movimiento de una ventana que recorre
                # a lo ancho y a lo alto la matriz original, extayendo submatrices
                sub_matrix = input_matrix[i:i+kernel_height, j:j+kernel_width]  # Seleccionamos la submatriz de la entrada
                output[i, j] = np.sum(sub_matrix * kernel)  # Producto punto entre la submatriz y el kernel

        return output

    # Convolución 2D (correlación con kernel volteado)
    # usaremos la convolución en la etapa de backward prop
    def convolve(self, input_matrix, kernel):
        # Volteamos el kernel en ambas direcciones (filas y columnas)
        flipped_kernel = np.flipud(np.fliplr(kernel))
        # Aplicamos la correlación con el kernel volteado
        return self.correlate(input_matrix, flipped_kernel)

    def forward_propagation(self, input_data):
        # Guardamos la entrada para usarla en backward_propagation
        self.input = input_data
        # Inicializamos la salida con ceros
        self.output = np.zeros(self.output_shape)

        # Calculamos el padding en función del tamaño del kernel
        pad_h = self.kernel_shape[0] // 2
        pad_w = self.kernel_shape[1] // 2

        # Recorremos cada filtro en la profundidad de la capa
        for k in range(self.num_kernels):
            for d in range(self.input_depth):
                # Agregamos padding a la entrada actual (canal d)
                padded_input = self.pad_input(self.input[:,:,d], pad_h)
                # Realizamos la correlación y obtenemos la salida parcial
                correlated_output = self.correlate(padded_input, self.weights[:,:,d,k])
                # Sumamos el resultado de la correlación al acumulador de la salida
                self.output[:,:,k] += correlated_output[:self.output_shape[0], :self.output_shape[1]]  
            # Añadimos el bias al filtro actual
            self.output[:,:,k] += self.bias[k]

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        # Inicializamos los errores de entrada y de pesos como matrices de ceros
        in_error = np.zeros(self.input_shape)
        weights_error = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.num_kernels))
        dBias = np.zeros(self.num_kernels)

        # Calculamos el padding basado en el tamaño del kernel
        pad_h = self.kernel_shape[0] // 2
        pad_w = self.kernel_shape[1] // 2

        # Con cada kernel recorreremos cada capa de la entrada. 
        for k in range(self.num_kernels):
            for d in range(self.input_depth):
                # Aplicamos padding al error de salida antes de convolucionarlo con los pesos
                padded_output_error = self.pad_input(output_error[:,:,k], pad_h)
                # Realizamos la convolución inversa y la sumamos al error de entrada
                convolved_error = self.convolve(padded_output_error, self.weights[:,:,d,k])
                in_error[:,:,d] += convolved_error[:self.input_shape[0], :self.input_shape[1]]  
                # Calculamos el error de los pesos usando correlación entre entrada y error
                weights_error[:,:,d,k] = self.correlate(self.input[:,:,d], output_error[:,:,k])
            # Sumamos el error del bias en la salida actual
            dBias[k] = np.sum(output_error[:,:,k])  

        # Aplicamos la regularización L1 sobre los errores de los pesos
        weights_error += self.lambda_l1 * np.sign(self.weights)

        # Actualizamos los pesos y los sesgos usando el gradiente descendente
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * dBias

        # Retornamos el error de la entrada para la siguiente capa
        return in_error

# Clase para capas densas (fully connected)
class FCLayer(Layer):
    def __init__(self, input_size, output_size, lambda_reg=0, lambda_l1=0):
        # Inicialización de He mejor para ReLu
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))  # Puedes inicializar los sesgos en cero
        #self.weights = np.random.rand(input_size, output_size) - 0.5
        #self.bias = np.random.rand(1, output_size) - 0.5
        self.lambda_reg = lambda_reg
        self.lambda_l1 = lambda_l1

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        weights_error += self.lambda_reg * self.weights + self.lambda_l1 * np.sign(self.weights)

        # Actualizar los parámetros
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)  # Sumar sobre el batch
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class PoolingLayer(Layer):
    def __init__(self, pool_size=2, stride=2, mode='max'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode

    def forward_propagation(self, input_data):
        self.input = input_data
        (batch_size, height, width, channels) = input_data.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, out_height, out_width, channels))

        for b in range(batch_size):
            for h in range(0, height - self.pool_size + 1, self.stride):
                for w in range(0, width - self.pool_size + 1, self.stride):
                    for c in range(channels):
                        h_end = h + self.pool_size
                        w_end = w + self.pool_size
                        window = input_data[b, h:h_end, w:w_end, c]

                        if self.mode == 'max':
                            output[b, h // self.stride, w // self.stride, c] = np.max(window)
                        elif self.mode == 'avg':
                            output[b, h // self.stride, w // self.stride, c] = np.mean(window)

        return output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape)
        (batch_size, height, width, channels) = self.input.shape

        for b in range(batch_size):
            for h in range(0, height - self.pool_size + 1, self.stride):
                for w in range(0, width - self.pool_size + 1, self.stride):
                    for c in range(channels):
                        h_end = h + self.pool_size
                        w_end = w + self.pool_size
                        window = self.input[b, h:h_end, w:w_end, c]

                        if self.mode == 'max':
                            max_val = np.max(window)
                            input_error[b, h:h_end, w:w_end, c] += (window == max_val) * output_error[b, h // self.stride, w // self.stride, c]
                        elif self.mode == 'avg':
                            input_error[b, h:h_end, w:w_end, c] += output_error[b, h // self.stride, w // self.stride, c] / (self.pool_size * self.pool_size)

        return input_error

class FlattenLayer(Layer):
    def forward_propagation(self, input_data):
        # Guardamos la forma original para usarla en backward
        self.input_shape = input_data.shape
        # Aplanamos los datos
        return input_data.reshape((input_data.shape[0], -1))  # Aplanar a 2D (batch_size, features)

    def backward_propagation(self, output_error, learning_rate):
        # Volver a la forma original en la retropropagación
        return output_error.reshape(self.input_shape)

# Clase DropoutLayer
class DropoutLayer(Layer):
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward_propagation(self, input_data):
        self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape)
        return input_data * self.mask

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.mask
