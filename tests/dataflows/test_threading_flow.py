import unittest

import numpy as np
import pytest
import time

from tfsnippet.dataflows import DataFlow
from tfsnippet.dataflows .threading_flow import ThreadingFlow


class _MyError(Exception):
    pass


class ThreadingFlowTestCase(unittest.TestCase):

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`prefetch_num` must be at least 1'):
            _ = ThreadingFlow(DataFlow.arrays([np.arange(10)], batch_size=2),
                              prefetch=0)

    def test_threaded(self):
        flow = DataFlow.arrays([np.arange(10)], batch_size=2). \
            threaded(prefetch=3)
        self.assertIsInstance(flow, ThreadingFlow)
        self.assertEqual(3, flow.prefetch_num)

    def test_iterator(self):
        epoch_counter = [0]
        external_counter = [1]

        seq_flow = DataFlow.seq(0, 10, batch_size=2)
        map_flow = seq_flow.map(
            lambda x: (x + epoch_counter[0] * 10 + external_counter[0] * 100,))

        def make_iterator():
            epoch_counter[0] += 1
            return map_flow

        it_flow = DataFlow.iterator_factory(make_iterator)
        with it_flow.threaded(prefetch=2) as flow:
            # the first epoch, expect 0 .. 10
            np.testing.assert_array_equal(
                [[110, 111], [112, 113], [114, 115], [116, 117], [118, 119]],
                [a[0] for a in flow]
            )
            time.sleep(.1)
            external_counter[0] += 1

            # the second epoch, the epoch counter should affect more than
            # the external counter
            np.testing.assert_array_equal(
                # having `prefetch = 2` should affect 3 items, because
                # while the queue size is 2, there are 1 additional prefetched
                # item waiting to be enqueued
                [[120, 121], [122, 123], [124, 125], [226, 227], [228, 229]],
                [a[0] for a in flow]
            )
            time.sleep(.1)
            external_counter[0] += 1

            # the third epoch, we shall carry out an incomplete epoch by break
            for a in flow:
                np.testing.assert_array_equal([230, 231], a[0])
                break
            time.sleep(.1)
            external_counter[0] += 1

            # verify that the epoch counter increases after break
            for i, (a,) in enumerate(flow):
                # because the interruption is not well-predictable under
                # multi-threading context, we shall have a weaker verification
                # than the above
                self.assertTrue((340 + i * 2 == a[0]) or (440 + i * 2 == a[0]))
                self.assertTrue((341 + i * 2 == a[1]) or (441 + i * 2 == a[1]))
            time.sleep(.1)
            external_counter[0] += 1

            # carry out the fourth, incomplete epoch by error
            try:
                for a in flow:
                    np.testing.assert_array_equal([450, 451], a[0])
                    raise _MyError()
            except _MyError:
                pass
            time.sleep(.1)
            external_counter[0] += 1

            # verify that the epoch counter increases after error
            for i, (a,) in enumerate(flow):
                self.assertTrue((560 + i * 2 == a[0]) or (660 + i * 2 == a[0]))
                self.assertTrue((561 + i * 2 == a[1]) or (661 + i * 2 == a[1]))

    def test_auto_init(self):
        epoch_counter = [0]

        seq_flow = DataFlow.seq(0, 10, batch_size=2)
        map_flow = seq_flow.map(
            lambda x: (x + epoch_counter[0] * 10,))

        def make_iterator():
            epoch_counter[0] += 1
            return map_flow

        it_flow = DataFlow.iterator_factory(make_iterator)
        flow = it_flow.threaded(3)

        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], batches)

        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]], batches)

        flow.close()
        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[40, 41], [42, 43], [44, 45], [46, 47], [48, 49]], batches)

        flow.close()
