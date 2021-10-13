import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])  # x가 2개인 임의의 개수의 데이터를 보관하는 형상
Y = tf.placeholder(tf.float32, shape=[None, 1])  # y가 1개인 임의의 개수의 데이터를 보관하는 형상

W = tf.Variable(tf.random.normal([2, 1]), name='weight')  # [들어오는 값의 수, 나가는 값의 수]
b = tf.Variable(tf.random.normal([1]), name='bias')  # [나가는 값의 수]

# 시그모이드 함수를 사용해 hypothesis를 구한다.
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# tensorflow를 사용하기 때문에 직접 미분할 필요가 없고, cost를 전달해서 minimize 한다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 정확도 계산
# hypothesis가 0.5보다 크면 True
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 세션을 만든다.
with tf.Session() as sess:
    # tensorflow var들을 초기화
    sess.run(tf.global_variables_initializer())

    for step in range(10000 + 1):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # 정확도 계산
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print('Hypothesis: {}\nCorrect: {}\nAccuracy: {}\n'.format(h, c, a))
