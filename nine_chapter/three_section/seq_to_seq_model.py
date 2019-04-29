import tensorflow as tf


src_train_data = "./ptb.train"
trg_train_data = "./ptb.target"
checkpoint_path = "./seq2seq_checkpoint/seq2seq_ckpt"
hidden_size = 1024
num_layers = 2
src_vocab_size = 10000
trg_vocab_size = 4000
batch_size = 100
num_epoch = 5
keep_prob = 0.8
max_grad_norm = 5
share_emd_and_softmax = True

max_len = 50
sos_id = 1


class NmtModel(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)]
        )
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(num_layers)]
        )

        self.src_embedding = tf.get_variable(
            "src_emb", [src_vocab_size, hidden_size]
        )
        self.trg_embedding = tf.get_variable(
            "trg_emb", [trg_vocab_size, hidden_size]
        )

        if share_emd_and_softmax:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "weight", [hidden_size, trg_vocab_size]
            )
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [trg_vocab_size]
        )

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        src_emb = tf.nn.dropout(src_emb, keep_prob)
        trg_emb = tf.nn.dropout(trg_emb, keep_prob)

        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype=tf.float32)

        with tf.variable_scope("decoder"):
            dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state=enc_state)

        output = tf.reshape(dec_outputs, [-1, hidden_size])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32
        )
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After {} steps, per token cost is {}".format(step, cost))
            if step % 200 == 0:
                saver.save(session, checkpoint_path, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def make_dataset(filepath):
    dataset = tf.data.TextLineDataset(filepath)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def make_src_trg_dataset(src_path, trg_path, batch_size):
    src_data = make_dataset(src_path)
    trg_data = make_dataset(trg_path)

    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def filter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, max_len))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, max_len))
        return tf.logical_and(src_len_ok, trg_len_ok)
    dataset = dataset.filter(filter_length)

    def make_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[sos_id], trg_label[: -1]], axis=0)
        return (src_input, src_len), (trg_input, trg_label, trg_len)
    dataset = dataset.map(make_trg_input)

    dataset = dataset.shuffle(10000)

    padding_shapes = (
        (
            tf.TensorShape([None]),
            tf.TensorShape([])
        ),
        (
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([])
        )
    )

    batched_dataset = dataset.padded_batch(batch_size, padding_shapes)
    return batched_dataset


def main():
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        train_model = NmtModel()

    data = make_src_trg_dataset(src_train_data, trg_train_data, batch_size)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(num_epoch):
            print("In iteration: {}".format(i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()