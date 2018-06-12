import os
import unittest

import imageio
import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.applications import InceptionV3
from tfsnippet.nn import softmax, npyops


class InceptionV3TestCase(tf.test.TestCase):

    def test_predict(self):
        model = InceptionV3()
        cache_dir = model._cache_dir

        # read the image data
        image_path = os.path.join(
            cache_dir.path, 'inception-2015-12-05/cropped_panda.jpg')
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # predict proba with jpeg
        proba = model.predict_proba([image_data]).astype(np.float32)
        self.assertEquals((1, 1008), proba.shape)
        self.assertEquals(169, np.argmax(np.squeeze(proba)))

        # predict logits with binary data
        image_array = imageio.imread(image_path)
        image_array = np.expand_dims(image_array, 0)
        logits = model.predict_logits(image_array).astype(np.float32)
        self.assertEquals((1, 1008), logits.shape)
        self.assertEquals(169, np.argmax(np.squeeze(logits)))
        np.testing.assert_allclose(
            proba[0, 169], softmax(npyops, logits)[0, 169], rtol=1e-2)

        # check get labels
        self.assertListEqual(
            ['giant panda, panda, panda bear, coon bear, Ailuropoda '
             'melanoleuca', 'kit fox, Vulpes macrotis', ''],
            model.get_labels([169, 1, -1])
        )

        # test errors
        with pytest.raises(TypeError, match='images` must be a list of bytes, '
                                            'or a numpy array'):
            _ = model.predict_logits(np.linspace(0, 1, 10000).
                                     reshape((10, 10, 10, 10)))
        with pytest.raises(TypeError, match='images` must be a list of bytes, '
                                            'or a numpy array'):
            _ = model.predict_logits([1, 2, 3])
        with pytest.raises(TypeError, match='images` must be a list of bytes, '
                                            'or a numpy array'):
            _ = model.predict_logits(object())


if __name__ == '__main__':
    unittest.main()
