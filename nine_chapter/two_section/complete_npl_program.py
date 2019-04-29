import numpy as np
import tensorflow as tf

train_data = 'ptb.train'
eval_data = 'ptb.valid'
test_data = 'ptb.test'

hidden_size = 300
num_layers = 2
vocab_size = 10000
train_batch_size = 20
train_num_step = 35

eval_batch_size = 1
eval_num_step = 1
num_epoch = 5
lstsm_keep_prob = 0.9
embedding_keep_prob = 0.9
max_grad_norm = 5
share_emb_and_softmax = True


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        dropout_keep_prob = lstsm_keep_prob if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(hidden_size),
                output_keep_prob=dropout_keep_prob
            )
            for _ in range(num_layers)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [vocab_size, hidden_size])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, embedding_keep_prob)

        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

        if share_emb_and_softmax:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [hidden_size, vocab_size])
        bias = tf.get_variable("bias", [vocab_size])
        logits = tf.matmul(output, weight) + bias

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables),
            max_grad_norm
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


def run_epoch(session, model, batches, train_op, output_log, step):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for x, y in batches:
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y, model.initial_state: state}
        )
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After {} steps, perplexity is {}".format(step, np.exp(total_costs / iters)))
        step += 1

    return step, np.exp(total_costs / iters)


def read_data(file_path):
    with open(file_path, "r") as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_step):
    num_batchs = (len(id_list) - 1) // (batch_size * num_step)

    data = np.array(id_list[: num_batchs * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batchs * num_step])

    data_batches = np.split(data, num_batchs, axis=1)

    label = np.array(id_list[1: num_batchs * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batchs * num_step])
    label_batches = np.split(label, num_batchs, axis=1)
    return list(zip(data_batches, label_batches))


def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, train_batch_size, train_num_step)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, eval_batch_size, eval_num_step)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(train_data), train_batch_size, train_num_step)
        eval_batches = make_batches(read_data(eval_data), eval_batch_size, eval_num_step)
        test_batches = make_batches(read_data(test_data), eval_batch_size, eval_num_step)

        step = 0
        for i in range(num_epoch):
            print("In iteration: {}".format(i + 1))
            step, train_pplx = run_epoch(
                session, train_model, train_batches, train_model.train_op, True, step
            )
            print("Epoch: {} Train Perplexity: {}".format(i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
            print("Epoch: {} Eval Perplexity: {}".format(i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(), False, 0)
        print("Test Perplexity: {}".format(test_pplx))


if __name__ == '__main__':
    main()
