import numpy as np
import os
import tensorflow as tf

class mlp:
    def __init__(self, hidden_units, minibatch_size, regularization_rate, learning_rate):
        self.hidden_units = hidden_units
        self.minibatch_size = minibatch_size
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate

    # 렐루함수
    def relu_function(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                ret_vector[i, j] = max(0, matrix_content[i, j])

        return ret_vector

    # the gradient of ReLu
    def grad_relu(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                if matrix_content[i, j] > 0:
                    ret_vector[i, j] = 1
                else:
                    ret_vector[i, j] = 0

        return ret_vector

        # 미니배치
        def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
            assert inputs.shape[0] == targets.shape[0]  # 만약 input / output shape 체크

            if shuffle:
                indices = np.arange(inputs.shape[0])
                np.random.shuffle(indices)

            for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)

                yield inputs[excerpt], targets[excerpt]

    # 소프트맥스 함수
    def softmax_function(self, vector_content):
        return np.exp(vector_content - np.max(vector_content)) / np.sum(np.exp(vector_content - np.max(vector_content)),
                                                                        axis=0)


    #
    # 설계
    #
    def train(self, trainX, trainY, epochs):

        w1_mat = np.random.randn(self.hidden_units, 28 * 28) * np.sqrt(2. / (self.hidden_units + 28 * 28))
        w2_mat = np.random.randn(10, self.hidden_units) * np.sqrt(2. / (10 + self.hidden_units))
        b1_vec = np.zeros((self.hidden_units, 1))
        b2_vec = np.zeros((10, 1))


        trainX = np.reshape(trainX, (trainX.shape[0], 28 * 28))
        trainY = np.reshape(trainY, (trainY.shape[0], 1))

        for num_epochs in range(epochs):
            print("epoch : ", num_epochs)

            for batch in self.iterate_minibatches(trainX, trainY, self.minibatch_size, shuffle=True):
                x_batch, y_batch = batch
                x_batch = x_batch.T
                y_batch = y_batch.T

                # Logit/ReLu
                z1 = np.dot(w1_mat, x_batch) + b1_vec
                a1 = self.relu_function(z1, self.hidden_units, self.minibatch_size)

                # Logit/softmax
                z2 = np.dot(w2_mat, a1) + b2_vec
                a2_softmax = self.softmax_function(z2)

                # loss function
                gt_vector = np.zeros((10, self.minibatch_size))
                for example_num in range(self.minibatch_size):
                    gt_vector[y_batch[0, example_num], example_num] = 1

                # nomalization
                d_w2_mat = self.regularization_rate * w2_mat
                d_w1_mat = self.regularization_rate * w1_mat

                # Back-Propagation
                delta_2 = np.array(a2_softmax - gt_vector)
                d_w2_mat = d_w2_mat + np.dot(delta_2, (np.matrix(a1)).T)
                d_b2_vec = np.sum(delta_2, axis=1, keepdims=True)

                delta_1 = np.array(np.multiply((np.dot(w2_mat.T, delta_2)),
                                               self.grad_relu(z1, self.hidden_units, self.minibatch_size)))
                d_w1_mat = d_w1_mat + np.dot(delta_1, np.matrix(x_batch).T)
                d_b1_vec = np.sum(delta_1, axis=1, keepdims=True)

                d_w2_mat = d_w2_mat / self.minibatch_size
                d_w1_mat = d_w1_mat / self.minibatch_size
                d_b2_vec = d_b2_vec / self.minibatch_size
                d_b1_vec = d_b1_vec / self.minibatch_size

                # weight update
                w2_mat = w2_mat - self.learning_rate * d_w2_mat
                b2_vec = b2_vec - self.learning_rate * d_b2_vec

                w1_mat = w1_mat - self.learning_rate * d_w1_mat
                b1_vec = b1_vec - self.learning_rate * d_b1_vec

        self.w1_mat, self.b1_vec, self.w2_mat, self.b2_vec = w1_mat, b1_vec, w2_mat, b2_vec

    # test()
    def test(self, testX):
        output_labels = np.zeros(testX.shape[0])

        num_examples = testX.shape[0]

        testX = np.reshape(testX, (num_examples, 28 * 28))
        testX = testX.T

        # test model
        z1 = np.dot(self.w1_mat, testX) + self.b1_vec
        a1 = self.relu_function(z1, self.hidden_units, num_examples)

        z2 = np.dot(self.w2_mat, a1) + self.b2_vec
        a2_softmax = self.softmax_function(z2)

        for i in range(num_examples):
            pred_col = a2_softmax[:, [i]]
            output_labels[i] = np.argmax(pred_col)

        return output_labels

# mnist loading
def load_mnist():
    dataset = './'

    fd = open(os.path.join(dataset, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(dataset, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(dataset, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(dataset, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY


def main():
    trainX, trainY, testX, testY = load_mnist()
    print("Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape)

    epochs = 50
    num_hidden_units = 100
    minibatch_size = 100
    regularization_rate = 0.01
    learning_rate = 0.001

    models = mlp(num_hidden_units, minibatch_size, regularization_rate, learning_rate)

    print("Learning Start")
    models.train(trainX, trainY, epochs)
    print("Learning End")

    print("Test Starting")
    labels = models.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print("accuracy: %lf%%" % accuracy)


if __name__ == '__main__':
    main()