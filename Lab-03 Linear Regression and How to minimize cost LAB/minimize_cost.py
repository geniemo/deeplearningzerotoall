import tensorflow as tf

tf.random.set_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable([5.0])  # -100에서 100 사이 값, [1] 형태로 W 정의

for step in range(301):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01  # learning rate
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
