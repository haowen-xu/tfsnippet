import functools

import numpy as np
import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflows import DataFlow
from tfsnippet.scaffold import TrainLoop, AnnealingVariable, EventKeys
from tfsnippet.trainer import *
from tfsnippet.utils import EventSource


class BaseTrainerTestCase(tf.test.TestCase):

    def test_props(self):
        loop = Mock(valid_metric_name='valid_loss')
        t = BaseTrainer(loop)
        self.assertIs(loop, t.loop)
        self.assertIsInstance(t.events, EventSource)

    def test_add_and_remove_hooks(self):
        loop = Mock(
            valid_metric_name='valid_loss',
            print_logs=Mock(return_value=None, __repr__=lambda o: 'print_logs')
        )
        df = Mock()
        eval1 = Evaluator(loop, 1., [], df)
        eval2 = Evaluator(loop, 2., [], df)
        anneal1 = AnnealingVariable('anneal1', 1., .5)
        anneal2 = AnnealingVariable('anneal2', 2., .5)

        # test add
        t = BaseTrainer(loop)
        t.log_after_steps(3)
        t.log_after_epochs(4)
        t.evaluate_after_steps(
            Mock(return_value=None, __repr__=lambda o: 'eval'), 5)
        t.evaluate_after_epochs(
            Mock(return_value=None, __repr__=lambda o: 'eval'), 6)
        t.anneal_after_steps(
            Mock(return_value=None, __repr__=lambda o: 'anneal'), 7)
        t.anneal_after_epochs(
            Mock(return_value=None, __repr__=lambda o: 'anneal'), 8)
        t.evaluate_after_steps(eval1, 9)
        t.evaluate_after_epochs(eval2, 10)
        t.anneal_after_steps(anneal1, 11)
        t.anneal_after_epochs(anneal2, 12)
        t.log_after(steps=13)
        t.log_after(epochs=14)
        t.evaluate_after(
            Mock(return_value=None, __repr__=lambda o: 'eval2'),
            steps=15
        )
        t.evaluate_after(
            Mock(return_value=None, __repr__=lambda o: 'eval2'),
            epochs=16
        )
        t.anneal_after(
            Mock(return_value=None, __repr__=lambda o: 'anneal2'),
            steps=17
        )
        t.anneal_after(
            Mock(return_value=None, __repr__=lambda o: 'anneal2'),
            epochs=18
        )

        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.STEP_EVALUATION]),
            '[eval:step:5, {!r}:step:9, eval2:step:15]'.format(eval1.run)
        )
        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.STEP_ANNEALING]),
            '[anneal:step:7, {!r}:step:11, anneal2:step:17]'.
            format(anneal1.anneal)
        )
        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.STEP_LOGGING]),
            '[print_logs:step:3, print_logs:step:13]'
        )

        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.EPOCH_EVALUATION]),
            '[eval:epoch:6, {!r}:epoch:10, eval2:epoch:16]'.format(eval2.run)
        )
        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.EPOCH_ANNEALING]),
            '[anneal:epoch:8, {!r}:epoch:12, anneal2:epoch:18]'.
            format(anneal2.anneal)
        )
        self.assertEqual(
            repr(t.events._event_handlers_map[EventKeys.EPOCH_LOGGING]),
            '[print_logs:epoch:4, print_logs:epoch:14]'
        )

        # test remove
        t.remove_log_hooks()
        self.assertNotIn(
            EventKeys.STEP_LOGGING, t.events._event_handlers_map)
        self.assertNotIn(
            EventKeys.EPOCH_LOGGING, t.events._event_handlers_map)

        t.remove_validation_hooks()
        self.assertNotIn(
            EventKeys.STEP_EVALUATION, t.events._event_handlers_map)
        self.assertNotIn(
            EventKeys.EPOCH_EVALUATION, t.events._event_handlers_map)

        t.remove_annealing_hooks()
        self.assertNotIn(
            EventKeys.STEP_ANNEALING, t.events._event_handlers_map)
        self.assertNotIn(
            EventKeys.EPOCH_ANNEALING, t.events._event_handlers_map)

        # test error add
        func_list = [
            t.log_after,
            functools.partial(t.evaluate_after, Mock()),
            functools.partial(t.anneal_after, Mock()),
        ]
        kwargs_list = [
            {'steps': None, 'epochs': None},
            {'steps': 1, 'epochs': 1}
        ]
        for func in func_list:
            for kwargs in kwargs_list:
                with pytest.raises(
                        ValueError, match='One and only one of `epochs` and '
                                          '`steps` should be specified'):
                    func(**kwargs)

    def test_hook_freq(self):
        loop = Mock(
            valid_metric_name='valid_loss',
            print_logs=Mock(return_value=None, __repr__=lambda o: 'print_logs')
        )
        t = BaseTrainer(loop)
        f = Mock()
        t.evaluate_after(f, steps=5)

        for i in range(1, 6):
            t.loop.step = i
            t.events.fire(EventKeys.STEP_EVALUATION, t)
        t.loop.step = 7
        t.events.fire(EventKeys.STEP_EVALUATION, t)
        t.loop.step = 10
        t.events.fire(EventKeys.STEP_EVALUATION, t)

        self.assertEqual(f.call_count, 2)

    def test_run(self):
        with self.test_session() as session:
            df = DataFlow.arrays([np.arange(6, dtype=np.float32)], batch_size=4)

            def log_event(m, trainer):
                logged_events.append((m, trainer))
            logged_events = []

            # test default loss weight and merged feed dict
            with TrainLoop([], max_epoch=2) as loop:
                t = BaseTrainer(loop)
                t._run_step = Mock(return_value=None)
                t._iter_steps = Mock(wraps=lambda: loop.iter_steps(df))
                for key in [EventKeys.BEFORE_EPOCH,
                            EventKeys.BEFORE_STEP,
                            EventKeys.STEP_ANNEALING,
                            EventKeys.STEP_EVALUATION,
                            EventKeys.STEP_LOGGING,
                            EventKeys.AFTER_STEP,
                            EventKeys.EPOCH_ANNEALING,
                            EventKeys.EPOCH_EVALUATION,
                            EventKeys.EPOCH_LOGGING,
                            EventKeys.AFTER_EPOCH]:
                    t.events.on(key, functools.partial(log_event, key))

                t.run()
                self.assertEqual(4, len(t._run_step.call_args_list))
                for i, call_args in enumerate(t._run_step.call_args_list[:-2]):
                    call_session, call_payload = call_args[0]
                    self.assertIs(session, call_session)
                    self.assertEqual(i + 1, call_payload[0])
                    self.assertIsInstance(call_payload[1], tuple)
                    self.assertEqual(1, len(call_payload[1]))
                    np.testing.assert_equal(
                        np.arange(6, dtype=np.float32)[i * 4: (i + 1) * 4],
                        call_payload[1][0]
                    )

                expected_logged_events = sum(
                    [
                        [
                            (EventKeys.BEFORE_EPOCH, t),
                        ] + sum([
                            [
                                (EventKeys.BEFORE_STEP, t),
                                (EventKeys.STEP_EVALUATION, t),
                                (EventKeys.STEP_ANNEALING, t),
                                (EventKeys.STEP_LOGGING, t),
                                (EventKeys.AFTER_STEP, t),
                            ]
                            for step in [0, 1]
                        ], []) + [
                            (EventKeys.EPOCH_EVALUATION, t),
                            (EventKeys.EPOCH_ANNEALING, t),
                            (EventKeys.EPOCH_LOGGING, t),
                            (EventKeys.AFTER_EPOCH, t)
                        ]
                        for epoch in [0, 1]
                    ],
                    []
                )
                self.assertListEqual(logged_events, expected_logged_events)

            # test re-entrant error
            with TrainLoop([], max_epoch=1) as loop:
                t = BaseTrainer(loop)
                t._run_step = Mock(return_value=None)
                t._iter_steps = Mock(wraps=lambda: loop.iter_steps(df))

                def reentrant_error(trainer):
                    self.assertIs(trainer, t)
                    with pytest.raises(
                            RuntimeError, match=r'`run\(\)` is not re-entrant'):
                        t.run()
                reentrant_error = Mock(wraps=reentrant_error)
                t.events.on(EventKeys.AFTER_STEP, reentrant_error)
                t.run()
                self.assertTrue(reentrant_error.called)
