import tensorflow as tf

# tf.enable_eager_execution()  # 즉시 실행하게 한다.

# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W, b initialize, 아무 값으로 일단 초기화
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# learning rate initialize, 기울기의 변화를 얼마나 반영할지를 말하는 변수
learning_rate = 0.01

for i in range(100 + 1):  # W, b update
    # Gradient descent
    # 변수들(W, b)에 대한 정보를 tape에 저장하고, 이후 tape의 gradient 함수를 이용해
    # 경사도 값을 구한다
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W, b])  # 각각의 gradient를 저장

    # -= 와 같은 연산이다. cost가 줄어드는 방향으로 W와 b를 조정해준다.
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print('{:5}|{:10.4f}|{:10.4}|{:10.6}'.format(i, W.numpy(), b.numpy(), cost))

# 새로운 데이터에 대해서 입력과 출력이 잘 나오는지 확인해보면 잘 나온다.
print(W * 5 + b)
print(W * 2.5 + b)
