
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()
filename = [".\Train1_.csv", ".\Eval1_.csv"]
graphs = os.path.abspath(os.path.join('.', 'graphs'))

lr = 0.001
n_epochs = 500
batch_size_num = 140
num_features = 1
time_steps = 3
hidden_sizes = [250, 500, 250]
dense_hidden = [125, 62, 1]
gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
batch_size = tf.placeholder_with_default(tf.cast(batch_size_num, dtype=tf.int64), shape=[])


def windows_data(data, time_steps, labels=False):
    lista_ = []
    for i in range(data.shape[0] - time_steps):
        if labels:
            lista_.append(data[i + time_steps])
        else:
            data_ = data[i: i + time_steps]
            lista_.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(lista_)


with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, time_steps, num_features])
    y = tf.placeholder(tf.float32, [None, 1, num_features])
    tr_data = pd.read_csv(filename[0], header=0)
    val_data = pd.read_csv(filename[1], header=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(tr_data)
    tr_data = scaler.transform(tr_data.values)
    val_data = scaler.transform(val_data.values)

    tr_x = windows_data(tr_data, time_steps).reshape(-1, time_steps, num_features)
    tr_y = windows_data(tr_data, time_steps, True).reshape(-1, 1, num_features)
    val_x = windows_data(val_data, time_steps).reshape(-1, time_steps, num_features)
    val_y = windows_data(val_data, time_steps, True).reshape(-1, 1, num_features)
    n_val = val_x.shape[0]

    tr_data = tf.data.Dataset.from_tensor_slices((X, y)).apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    val_data = tf.data.Dataset.from_tensor_slices((X, y)).apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                               tr_data.output_shapes)
    input, label = iterator.get_next()

    train_init = iterator.make_initializer(tr_data)
    val_init = iterator.make_initializer(val_data)

initializer = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)

with tf.name_scope('rnn'):
    layers = [tf.nn.rnn_cell.GRUCell(size, kernel_initializer=initializer)
              for size in hidden_sizes]
    cells = tf.nn.rnn_cell.MultiRNNCell(layers)
    # init_state = cells.zero_state(tf.cast(batch_size, tf.int32), dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(
        cells, input, dtype=tf.float32)

with tf.name_scope('Dense'):
    Dense_layer_1 = tf.layers.dense(
        output[:, -1], dense_hidden[0], activation=tf.nn.relu, kernel_initializer=initializer, name='Dense_layer_1')
    Dense_layer_2 = tf.layers.dense(
        Dense_layer_1, dense_hidden[1], activation=tf.nn.relu, kernel_initializer=initializer, name='Dense_layer_2')
    final_layer = tf.layers.dense(Dense_layer_2, 1, activation=tf.nn.relu,
                                  kernel_initializer=initializer, name='Final_layer')
    final_layer = tf.reshape(final_layer, [-1, 1, num_features])

loss = tf.losses.mean_squared_error(labels=label, predictions=final_layer)
opt = tf.contrib.layers.optimize_loss(loss, gstep, lr, 'Adam', summaries=["gradients"])
grad_summ = tf.summary.merge_all()

with tf.name_scope('summaries'):
    cost = tf.summary.scalar('Train loss', loss)
    histo_cost = tf.summary.histogram('Train histogram loss', loss)
    train_summary_op = tf.summary.merge([cost, histo_cost, grad_summ], name='Total train summary')
    val_cost = tf.summary.scalar('Val loss', loss)
    val_histo_cost = tf.summary.histogram('Val histogram loss', loss)
    val_summary_op = tf.summary.merge([val_cost, val_histo_cost], name='Total val summary')

writer = tf.summary.FileWriter(graphs, tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        # Training
        sess.run(train_init, feed_dict={X: tr_x, y: tr_y, batch_size: batch_size_num})
        try:
            while True:
                _, l, summaries = sess.run([opt, loss, train_summary_op])
                writer.add_summary(summaries, global_step=gstep.eval())
                n_batches += 1
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass
        print('Tr loss at epoch {}: {}'.format(epoch, total_loss/n_batches))

        # Check val errors
        sess.run(val_init, feed_dict={X: val_x, y: val_y, batch_size: n_val})
        val_loss, preds, summaries = sess.run([loss, final_layer, val_summary_op])
        writer.add_summary(summaries, global_step=gstep.eval())
        print('val_loss at epoch {} is : {}'.format(epoch, val_loss))

print("\ttensorboard --logdir=%s -- port 6006" % (graphs))
print("Browser to: http://localhost:6006/")
