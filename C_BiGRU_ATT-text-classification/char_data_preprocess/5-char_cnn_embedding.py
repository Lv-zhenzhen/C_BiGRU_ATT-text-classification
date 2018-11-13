#-*- coding:utf-8 -*-
import os
import time
import json
import warnings
from utils import data_utils
import numpy as np
from numpy import *
import tensorflow as tf
from model.text_cnn import TextCNN
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# Parameters
# ==================================================

# Data loading parameters
tf.flags.DEFINE_string('x_data_file', '../data/char_data/pad_data_to_id/doc_char_to_id_20_30.txt', "Data source for the text data")
tf.flags.DEFINE_string('y_data_file', '../data/doc_cat_id.txt', "Data source for the text data")
tf.flags.DEFINE_float('test_size', 0.99, "Percentage of data to use for validation and test (default: 0.05)")
tf.flags.DEFINE_integer('vocab_size', 5000, "Select words to build vocabulary, according to term frequency (default: 9000)")
tf.flags.DEFINE_integer('sequence_length', 600, "Padding sentences to same length, cut off when necessary (default: 500)")
tf.flags.DEFINE_float('num_classes', 11, "Number of classes")
# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 256, "Dimension of word embedding (default: 128)")
tf.flags.DEFINE_string('filter_sizes', '3,4,5', "Filter sizes to use in convolution layer, comma-separated (default: '3,4,5')")
tf.flags.DEFINE_integer('num_filters', 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('learning_rate', 0.001, "Learning rate for model training (default: 0.001)")
tf.flags.DEFINE_float('grad_clip', 3.0, "Gradients clipping threshold (default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, "Batch size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer('evaluate_every', 50, "Evaluate model on val set after this many steps (default: 50)")
tf.flags.DEFINE_string('init_embedding_path', None, "Using pre-trained word embedding, npy file format")
tf.flags.DEFINE_string('init_model_path', None, "Continue training from saved model at this path")
# Tensorflow parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, "Allow device soft device placement (default: True)")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log placement of ops on devices (default: False)")
tf.flags.DEFINE_boolean('gpu_allow_growth', True, "GPU memory allocation mode (default: True)")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
print("\nParameters:")
for param, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(param.upper(), value))
print("")


def train_cnn():
    # Data Preparation
    # ==================================================
    if FLAGS.init_embedding_path is not None:
        embedding = np.load(FLAGS.init_embedding_path)
        print("Using pre-trained word embedding which shape is {}\n".format(embedding.shape))
        FLAGS.vocab_size = embedding.shape[0]
        FLAGS.embedding_size = embedding.shape[1]
    if FLAGS.init_model_path is not None:
        assert os.path.isdir(FLAGS.init_model_path), "init_model_path must be a directory\n"
        ckpt = tf.train.get_checkpoint_state(FLAGS.init_model_path)
        assert ckpt, "No checkpoint found in {}\n".format(FLAGS.init_model_path)
        assert ckpt.model_checkpoint_path, "No model_checkpoint_path found in checkpoint\n"

    # Create root directory
    timestamp = str(int(time.time()))
    root_dir = os.path.join(os.path.curdir, 'runs', 'textcnn', 'trained_result_' + timestamp)
    os.makedirs(root_dir)

    # Load data
    print("Loading data...\n")
    x_data = np.loadtxt(FLAGS.x_data_file)
    x_data = x_data.reshape(20480, 20, 30)
    x_data = x_data.reshape(20480, 600)
    y_data = np.loadtxt(FLAGS.y_data_file)
    print("data load finished")

    # Split dataset
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=FLAGS.test_size, stratify=y_data, random_state=0)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

    # Training
    # ==================================================
    with tf.Graph().as_default():
        tf_config = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        tf_config.gpu_options.allow_growth = FLAGS.gpu_allow_growth

        with tf.Session(config=tf_config).as_default() as sess:
            cnn = TextCNN(
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_size,
                sequence_length=FLAGS.sequence_length,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_classes=FLAGS.num_classes,
                learning_rate=FLAGS.learning_rate,
                grad_clip=FLAGS.grad_clip,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Output directory for models and summaries
            out_dir = os.path.abspath(root_dir)
            print("Writing to {}...\n".format(out_dir))

            # Summaries for loss and accuracy
            tf.summary.scalar("loss", cnn.loss)
            tf.summary.scalar("accuracy", cnn.accuracy)
            merged_summary = tf.summary.merge_all()

            # Summaries dictionary
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            val_summary_dir = os.path.join(out_dir, 'summaries', 'val')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory, will not create itself
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model.ckpt')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Using pre-trained word embedding
            # if FLAGS.init_embedding_path is not None:
            #     sess.run(cnn.embedding.assign(embedding))
            #     del embedding

            # Continue training from saved model
            if FLAGS.init_model_path is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Training start
            print("Start training...\n")
            best_at_step = 0
            best_val_accuracy = 0

            #****************************************
            # Generate train batches
            train_batches = data_utils.batch_iter(list(zip(x_data, y_data)), FLAGS.batch_size)
            start = time.time()

            cnn_feature_temp = []
            for batch in train_batches:
                # Training model on x_batch and y_batch
                x_batch, y_batch = zip(*batch)
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.keep_prob: FLAGS.dropout_keep_prob,cnn.is_training: True}
                pooled_concat_flat, _, global_step, train_summaries, train_loss, train_accuracy = sess.run(
                    [cnn.pooled_concat_flat, cnn.train_op, cnn.global_step,
                        merged_summary, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
                cnn_feature_temp.append(pooled_concat_flat.tolist())

            np.savetxt("../data/char_data/char_dim/char_cnn_embeddings_20_30_dim256.txt", np.array(cnn_feature_temp).reshape(20480,192))
            # cnn_feature.append(cnn_feature_temp)
            # with open('./embeddings.txt','w', encoding='utf-8')as f:
            #     for line in cnn_feature_temp:
            #         for content in line :
            #                 f.write(str(content).lstrip('[').rstrip(']') + '\n')

            print('finished training')

if __name__ == '__main__':
    train_cnn()
