import numpy as np
import tensorflow as tf

data = np.array([
    # X1,  X2,  X3,  Y
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.]
], dtype=np.float32)

# slice data
X = data[:, :-1]  # 모든 행을 포함하고, 0열부터 -1열 이전까지의 열을 포함
Y = data[:, [-1]]  # 모든 행을 포함하고, -1열만 포함

W = tf.Variable(tf.random.normal([3, 1]))  # X가 3개이므로, 3행 1열의 형상을 띄도록
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001


def predict(X):
    return tf.matmul(X, W) + b


n_epochs = 2000
for i in range(n_epochs + 1):
    # tf.GradientTape() 를 이용해 cost function의 gradient를 기록
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(X) - Y))

    # cost의 gradient를 계산
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # update W, b
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print('{:5} | {:10.4f}'.format(i, cost.numpy()))
