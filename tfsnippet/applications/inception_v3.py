import os

import numpy as np
import re
import six
import tensorflow as tf

from tfsnippet.mathops import inception_score, npyops
from tfsnippet.utils import CacheDir, get_default_session_or_error

__all__ = ['InceptionV3']


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, model_dir):
        label_lookup_path = os.path.join(
            model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        uid_lookup_path = os.path.join(
            model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """
        Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            assert(val in uid_to_human)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        return self.node_lookup.get(node_id, '')


class InceptionV3(object):
    """
    Inception V3 model from TensorFlow
    `tutorial <https://www.tensorflow.org/tutorials/image_recognition>`_.

    This class directly loads the persisted TensorFlow graph, instead of
    assembling the model using TensorFlow or Keras APIs.  If you do need
    a freshly assembled model, you should turn to Keras instead.

    The major purpose of this class is to compute the same Inception score
    as "Improved techniques for training gans", Salimans, T. et al. 2016.
    """

    N_CLASSES = 1008

    def __init__(self, cache_dir=None):
        """
        Construct a new :class:`InceptionV3` instance.

        Args:
            cache_dir (CacheDir): The cache directory instance.
                If not specified, use a default one.
        """
        if cache_dir is None:
            cache_dir = CacheDir('inception-2015-12-05')
        self._cache_dir = cache_dir
        self._inited = False
        self._graph = None
        self._sess = None
        self._jpeg_input = None  # the jpeg data input
        self._array_input = None  # the numpy array input
        self._softmax_output = None  # the softmax proba output
        self._log_softmax_output = None  # the softmax log-proba output
        self._logits_output = None  # the softmax logits output
        self._node_lookup = None  # the mapping from class id to class label

    def _initialize_model(self):
        if not self._inited:
            # download the pretrained model
            model_dir = self._cache_dir.download_and_extract(
                'http://download.tensorflow.org/models/image/imagenet/'
                'inception-2015-12-05.tgz'
            )

            # load the computation graph
            with tf.gfile.FastGFile(os.path.join(
                    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                with tf.Graph().as_default() as g, \
                        tf.Session().as_default() as sess:
                    _ = tf.import_graph_def(graph_def, name='')

                    # easy to obtain inputs and outputs
                    jpeg_input = g.get_tensor_by_name('DecodeJpeg/contents:0')
                    array_input = g.get_tensor_by_name('ExpandDims:0')
                    softmax_output = g.get_tensor_by_name('softmax:0')

                    # re-assemble the logits output
                    # code derived from:
                    #   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
                    pool3 = g.get_tensor_by_name('pool_3:0')
                    w = g.get_tensor_by_name('softmax/weights:0')
                    b = g.get_tensor_by_name('softmax/biases:0')
                    logits_output = (
                        tf.matmul(tf.squeeze(pool3, axis=(1, 2)), w) + b)
                    log_softmax_output = tf.nn.log_softmax(logits_output)

                    # load the node lookup
                    node_lookup = NodeLookup(model_dir)

            # validate the shapes of tensors
            assert(jpeg_input.get_shape().as_list() == [])
            assert(array_input.get_shape().as_list() == [1, None, None, 3])
            assert(softmax_output.get_shape().as_list() == [1, self.N_CLASSES])
            assert(logits_output.get_shape().as_list() == [1, self.N_CLASSES])

            # memorize all the objects after everything is successfully loaded
            self._graph = g
            self._sess = sess
            self._jpeg_input = jpeg_input
            self._array_input = array_input
            self._softmax_output = softmax_output
            self._log_softmax_output = log_softmax_output
            self._logits_output = logits_output
            self._node_lookup = node_lookup
            self._inited = True

    def get_labels(self, classes):
        """
        Get the labels for specified `classes`.

        Args:
            classes (Iterable[int]): The IDs of the classes.

        Returns:
            list[str]: The class labels.
        """
        return [self._node_lookup.id_to_string(c) for c in classes]

    def _run(self, images, output):
        sess = get_default_session_or_error()
        err_msg = ('`images` must be a list of bytes, or a numpy array of '
                   'shape (?, ?, ?, 3).')
        if isinstance(images, list):
            for im in images:
                if not isinstance(im, six.binary_type):
                    raise TypeError(err_msg)
            input_tensor = self._jpeg_input
            get_image = lambda i: images[i]
        elif isinstance(images, np.ndarray):
            if len(images.shape) != 4 or images.shape[3] != 3:
                raise TypeError(err_msg)
            input_tensor = self._array_input
            get_image = lambda i: images[i: i+1].astype(np.float32)
        else:
            raise TypeError(err_msg)

        ret = []
        for i in range(len(images)):
            ret.append(sess.run(output, {input_tensor: get_image(i)}))
        return np.concatenate(ret, axis=0)

    def predict_proba(self, images):
        """
        Predict the class probabilities for `images`.

        Args:
            images (list[bytes] or np.ndarray): List of JPEG image data
                (each image as bytes), or numpy array of shape ``(?, ?, ?, 3)``,
                the pixels of images.  Note the pixels should be 256-colors.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        self._initialize_model()
        with self._graph.as_default(), self._sess.as_default():
            return self._run(images, self._softmax_output)

    def predict_log_proba(self, images):
        """
        Predict the class log-probabilities for `images`.

        Args:
            images (list[bytes] or np.ndarray): List of JPEG image data
                (each image as bytes), or numpy array of shape ``(?, ?, ?, 3)``,
                the pixels of images.  Note the pixels should be 256-colors.

        Returns:
            np.ndarray: The predicted class log-probabilities.
        """
        self._initialize_model()
        with self._graph.as_default(), self._sess.as_default():
            return self._run(images, self._log_softmax_output)

    def predict_logits(self, images):
        """
        Predict the softmax logits (un-normalized log-proba) for `images`.

        Args:
            images (list[bytes] or np.ndarray): List of JPEG image data
                (each image as bytes), or numpy array of shape ``(?, ?, ?, 3)``,
                the pixels of images.  Note the pixels should be 256-colors.

        Returns:
            np.ndarray: The predicted softmax logits probabilities.
        """
        self._initialize_model()
        with self._graph.as_default(), self._sess.as_default():
            return self._run(images, self._logits_output)

    def predict(self, images):
        """
        Predict the class classes for `images`.

        Args:
            images (list[bytes] or np.ndarray): List of JPEG image data
                (each image as bytes), or numpy array of shape ``(?, ?, ?, 3)``,
                the pixels of images.  Note the pixels should be 256-colors.

        Returns:
            np.ndarray: The predicted classes.
        """
        return np.argmax(self.predict_logits(images), axis=-1)

    def inception_score(self, images):
        """
        Compute the Inception score ("Improved techniques for training gans",
        Salimans, T. et al. 2016.) for specified images, using InceptionV3.

        Args:
            images (list[bytes] or np.ndarray): List of JPEG image data
                (each image as bytes), or numpy array of shape ``(?, ?, ?, 3)``,
                the pixels of images.  Note the pixels should be 256-colors.

        Returns:
            float: The Inception score for `images`.
        """
        self._initialize_model()
        with self._graph.as_default(), self._sess.as_default():
            logits = self._run(images, self._logits_output)
            return inception_score(npyops, logits=logits)
