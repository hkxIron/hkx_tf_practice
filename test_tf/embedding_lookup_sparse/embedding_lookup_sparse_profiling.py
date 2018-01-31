import tensorflow as tf
import numpy as np
import random
import argparse

# flags = tf.app.flags
# FLAGS = flags.FLAGS

def gen_sparse_indices(batch, nzdim):
    batch_ids = []
    for i in range(0, batch):
        batch_ids.extend([[i, j] for j in range(0, nzdim)])
    return batch_ids


# generate sparse tensor ids and values.
def gen_sparse_inputs(batch, dim, nzdim):
    batch_ids = []
    batch_values = []
    k = 0
    for i in range(0, batch):
        batch_ids.extend([(k + j) % dim for j in range(0, nzdim)])
        batch_values.extend(np.ones(nzdim, dtype=np.float32))
        k = (k + 1) % dim
    return {"ids": batch_ids,
            "values": batch_values}


def sparse_transform(ids, values, weight_shape):
    assert (len(weight_shape) == 2)
    with tf.device('/cpu:0'):
        weights = []
        # change the number of shards of weight.
        num_shards = 1
        assert (weight_shape[0] % num_shards == 0)

        for i in range(0, num_shards):
            weight_i = tf.get_variable("weight_%02d" % i,
                                       [weight_shape[0] / num_shards] + weight_shape[1:],
                                       trainable=True,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
            weights.append(weight_i)

    ids, _ = tf.sparse_fill_empty_rows(ids, 0)
    values, _ = tf.sparse_fill_empty_rows(values, 0.0)
    return tf.nn.embedding_lookup_sparse(weights, ids, values, partition_strategy='div', combiner='sum')


def test_embedding_lookup_sparse():
    batch = 256 #FLAGS.batch
    nzdim = 100 #FLAGS.nonzero_dim
    layers = "1000000,30"
    weight_shape = [int(d) for d in layers.split(",")]
    inputs = gen_sparse_inputs(batch, weight_shape[0], nzdim)

    batch_ids = gen_sparse_indices(batch, nzdim)

    embedding_op = sparse_transform(tf.SparseTensor(indices=batch_ids, values=inputs["ids"],
                                                    dense_shape=[batch, nzdim]),
                                    tf.SparseTensor(indices=batch_ids, values=inputs["values"],
                                                    dense_shape=[batch, nzdim]),
                                    weight_shape)

    init_op = tf.global_variables_initializer()

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=False)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=True,
                                 graph_options=graph_options)
    sess = tf.Session(config=sess_config)

    sess.run(init_op)

    step = 0
    max_steps = 1000
    while step < max_steps: #FLAGS.max_steps:
        sess.run([embedding_op],
                 options=run_options,
                 run_metadata=run_metadata)
        step += 1

    ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
    opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
                                ).with_node_names(show_name_regexes=['.*train.py.*']).build()

    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='code',
        options=opts)

    # Print to stdout an analysis of the memory usage and the timing information
    # broken down by operation types.
    tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='op',
        options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=run_metadata,
        tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)


def main(_):
    with tf.device('/cpu:0'):
    #with tf.device('/gpu:0'):
        test_embedding_lookup_sparse()


if __name__ == '__main__':
    print("heloow")
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--layers', type=str, default="10000000,100",
    #                     help='Network layers')
    # parser.add_argument('--batch', type=int, default=256,
    #                     help='Batch to train')
    # parser.add_argument('--nonzero_dim', type=int, default=1000,
    #                     help='non-zero dims of sparse tensor')
    # parser.add_argument('--max_steps', type=int, default=1000,
    #                     help='Max number of steps to run')
    #
    # FLAGS, unparsed = parser.parse_known_args()
    #main()
    tf.app.run(main=main)


