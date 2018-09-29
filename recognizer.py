import os
import numpy as np
import tensorflow as tf
import pprint

tf.app.flags.DEFINE_integer("epoch", 25, "Epoch to train")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
tf.app.flags.DEFINE_float("momentum", 0.5, "Momentum term of adam")
tf.app.flags.DEFINE_float("train_size", np.inf, "The size of train images")
tf.app.flags.DEFINE_integer("batch_size", 64, "The size of batch images")
tf.app.flags.DEFINE_integer(
    "input_height", 108, "The size of image to use (will be center cropped).")
tf.app.flags.DEFINE_integer(
    "input_width", 108, "The size of image to use (will be center cropped).")
tf.app.flags.DEFINE_string(
    "dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
tf.app.flags.DEFINE_string("input_fname_pattern",
                           "*.jpg", "Glob pattern of filename of input images")
tf.app.flags.DEFINE_string(
    "checkpoint_dir", "checkpoint", "Directory name to save the checkpoints")
tf.app.flags.DEFINE_string("data_dir", "./data", "Root directory of dataset")
tf.app.flags.DEFINE_string("sample_dir", "samples",
                           "Directory name to save the image samples")
tf.app.flags.DEFINE_boolean(
    "train", False, "True for training, False for testing")
tf.app.flags.DEFINE_boolean(
    "crop", False, "True for training, False for testing")
tf.app.flags.DEFINE_boolean(
    "visualize", False, "True for visualizing, False for nothing")
FLAGS = tf.app.flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                z_dim=FLAGS.generate_test_images,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir)
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                z_dim=FLAGS.generate_test_images,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir)

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        OPTION = 1
        visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
