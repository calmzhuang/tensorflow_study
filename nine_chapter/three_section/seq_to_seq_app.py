import tensorflow as tf

checkpoint_path = "./seq2seq_checkpoint/seq2seq_ckpt-3800"

hidden_size = 1024
num_layers = 2
src_vocab_size = 10000
trg_vocab_size = 4000
share_emd_and_softmax = True

sos_id = 1
eos_id = 2


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

    def inference(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32
            )

        max_dec_len = 100

        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            init_array = init_array.write(0, sos_id)
            init_loop_var = (enc_state, init_array, 0)

            def continue_loop_conditon(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step), eos_id), tf.less(step, max_dec_len - 1)))

            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)
                output = tf.reshape(dec_outputs, [-1, hidden_size])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            state, trg_ids, step = tf.while_loop(continue_loop_conditon, loop_body, init_loop_var)
            return trg_ids.stack()


def main():
    with tf.variable_scope("nmt_model", reuse=None):
        model = NmtModel()
    test_sentence = [90, 13, 9, 689, 4, 2]

    output_op = model.inference(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    output = sess.run(output_op)
    print(output)
    sess.close()


if __name__ == '__main__':
    main()