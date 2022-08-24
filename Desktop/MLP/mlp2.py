import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def one_hot(labels, targets=10):
    samples = len(labels)
    out = np.zeros((samples, targets))
    out[range(samples), labels] = 1
    return out


mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
train = mnist[0]
test = mnist[1]


x_train = train[0].astype(np.float32) / 255
y_train = train[1]
x_test = test[0].astype(np.float32) / 255
y_test = test[1]

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# 하이퍼 파라미터
learning_rate = 0.05
NODES = 100


# 설계
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='target')

w1 = tf.Variable(tf.truncated_normal(shape=[784, NODES], stddev=0.1), name='weight')
b1 = tf.Variable(tf.zeros(shape=[NODES], dtype=tf.float32), name='bias')
z1 = tf.matmul(x, w1) + b1
a1 = tf.sigmoid(z1)

w2 = tf.Variable(tf.truncated_normal(shape=[NODES, 10], stddev=0.1), name='W2')
b2 = tf.Variable(tf.zeros(shape=[10]), name='b2')
z2 = tf.matmul(a1, w2) + b2

yhat = tf.nn.softmax(z2)


# 손실함수, 최적화방법 결정
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z2)
loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)


# 인식성능계산
correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 세션 생성
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 에포크/스텝
EPOCH = 50
BATCH = 100
train_samples = x_train.shape[0]
steps = train_samples

x_train_batch = x_train.reshape(-1, BATCH, 28 * 28)
y_train_batch = y_train.reshape(-1, BATCH, 10)


x_test = x_test.reshape(-1, 28 * 28)
y_test = y_test.reshape(-1, 10)

loss_epoch = []
accuracy_epoch = []

loss_test = []
accuracy_test = []

for epoch in range(EPOCH):
    # 학습시작
    loss_batch = []
    accuracy_batch = []

    for step in range(steps):
        x_batch = x_train_batch[step]
        y_batch = y_train_batch[step]
        _, loss_, acc_ = sess.run([train, loss, accuracy], feed_dict={x: x_batch, y: y_batch})
        print('epoch={:4d} step={:4d} loss={:12.8f} accuracy={:6.5f}'.format(epoch, step, loss_, acc_))

        loss_batch.append(loss_)
        accuracy_batch.append(acc_)

    mean_loss = np.mean(loss_batch)
    mean_accuracy = np.mean(accuracy_batch)
    loss_epoch.append(mean_loss)
    accuracy_epoch.append(mean_accuracy)
    print('train: loss = {:.8f} accuracy = {:.5f}'.format(mean_loss, mean_accuracy))


    # 테스트
    loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: x_test, y: y_test})
    loss_test.append(loss_)
    accuracy_test.append(acc_)
    print('test: loss = {:.8f} accuracy ={:.5f}'.format(loss_, acc_))

sess.close()

plt.plot(loss_epoch, 'r')
plt.plot(loss_test, 'b')
plt.show()

plt.plot(accuracy_epoch, 'r')
plt.plot(accuracy_test, 'b')

plt.show()
