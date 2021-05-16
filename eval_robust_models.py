import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from nets import inception_v3, inception_resnet_v2


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = tf.image.resize_images(imread(f, mode='RGB'), [299, 299])
            image = image.eval().astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def evaluate_with_robust_models(input_dir):    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    slim = tf.contrib.slim

    checkpoint_path = './checkpoints'
    model_checkpoint_map = {
        'adv_inception_v3': os.path.join(checkpoint_path, 'adv_inception_v3_rename.ckpt'),
        'ens3_adv_inception_v3': os.path.join(checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
        'ens4_adv_inception_v3': os.path.join(checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
        'ens_adv_inception_resnet_v2': os.path.join(checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    }

    f2l = np.load("imagenet_class_to_idx.npy", allow_pickle=True)[()]

    # batch_shape = [50, 299, 299, 3]
    batch_shape = [100, 299, 299, 3]
    num_classes = 1001

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ens_adv_res_v2, end_points_ens_adv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

        pred_adv_v3 = tf.argmax(end_points_adv_v3['Predictions'], 1)
        pred_ens3_adv_v3 = tf.argmax(end_points_ens3_adv_v3['Predictions'], 1)
        pred_ens4_adv_v3 = tf.argmax(end_points_ens4_adv_v3['Predictions'], 1)
        pred_ens_adv_res_v2 = tf.argmax(end_points_ens_adv_res_v2['Predictions'], 1)

        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s2.restore(sess, model_checkpoint_map['adv_inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])

            model_name = ['ens3_adv_inception_v3', 'ens4_adv_inception_v3',
                          'ens_adv_inception_resnet_v2', 'adv_inception_v3']
            correct_count = np.zeros(len(model_name))

            idx = 0
            for filenames, images in load_images(input_dir, batch_shape):
                idx += 1
                print("start the i={} eval".format(idx))

                adv_v3, ens3_adv_v3, ens4_adv_v3, ens_adv_res_v2 = sess.run(
                    (
                        pred_adv_v3, 
                        pred_ens3_adv_v3, 
                        pred_ens4_adv_v3,
                        pred_ens_adv_res_v2
                    ), feed_dict={x_input: images})

                for filename, l1, l2, l3, l4 in zip(filenames, adv_v3, ens3_adv_v3, ens4_adv_v3, ens_adv_res_v2):
                    label = int(f2l[filename.split("_")[0]]) + 1
                    
                    l = [l2, l3, l4, l1]
                    for i in range(len(model_name)):
                        if l[i] == label:
                            correct_count[i] += 1

        return correct_count, model_name



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--total-num", type=int, default=1000)
    args = parser.parse_args()

    correct_cnt, model_name = evaluate_with_robust_models(args.input_dir)
    acc_list = []
    for i in range(len(model_name)):
        acc = correct_cnt[i] * 100.0 / args.total_num
        acc_list.append(acc)
        print(f"Transfer to {model_name[i]} accuracy: {acc:.2f}%")
    
    print("\t".join([str(round(v, 2)) for v in acc_list]))
