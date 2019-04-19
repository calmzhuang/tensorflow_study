import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def data_transform(data):
    if 'Survived' in data.columns.values:
        data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].copy()
    else:
        data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].copy()
    data['Age'] = data['Age'].fillna(data['Age'].mean())

    data['Cabin'] = pd.factorize(data.Cabin)[0]

    data.fillna(0, inplace=True)

    data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]

    data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
    data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
    data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)

    del data['Pclass']

    data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
    data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
    data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)

    del data['Embarked']

    return data


def run():
    data = pd.read_csv('./data/train.csv')
    data = data_transform(data)
    data_train_source = data[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'p1', 'p2', 'p3', 'e1', 'e2', 'e3']]
    data_train_target = data['Survived'].values.reshape(len(data), 1)

    x = tf.placeholder("float", shape=[None, 12])
    y = tf.placeholder("float", shape=[None, 1])

    weight = tf.Variable(tf.random_normal([12, 1]))
    bias = tf.Variable(tf.random_normal([1]))
    output = tf.matmul(x, weight) + bias
    pred = tf.cast(tf.sigmoid(output) > 0.5, tf.float32)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

    train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

    data_test = pd.read_csv('./data/test.csv')

    data_test = data_transform(data_test)

    test_lable = pd.read_csv('./data/gender.csv')

    test_lable = np.reshape(test_lable.Survived.values.astype(np.float32), (418, 1))

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    loss_train = []
    train_acc = []
    test_acc = []

    data_train_source = data_train_source.values
    # data_train_target = data_train_target.values

    for i in range(25000):
        index = np.random.permutation(len(data_train_target))
        data_train_source = data_train_source[index]
        data_train_target = data_train_target[index]
        for n in range(len(data_train_target)//100 + 1):
            batch_xs = data_train_source[n*100: n*100 + 100]
            batch_ys = data_train_target[n*100: n*100 + 100]
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if i % 1000 == 0:
            loss_temp = sess.run(loss, feed_dict={x: data_train_source, y: data_train_target})
            loss_train.append(loss_temp)
            train_acc_temp = sess.run(accuracy, feed_dict={x: data_train_source, y: data_train_target})
            train_acc.append(train_acc_temp)
            test_acc_temp = sess.run(accuracy, feed_dict={x: data_test, y: test_lable})
            test_acc.append(test_acc_temp)
            print(loss_temp, train_acc_temp, test_acc_temp)

    plt.plot(loss_train, 'k-')
    plt.title('train loss')
    plt.show()

    plt.plot(train_acc, 'b-', label='train_acc')
    plt.plot(test_acc, 'r--', label='test_acc')
    plt.title('train and test accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
