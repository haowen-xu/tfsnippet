import unittest

from tfsnippet.utils import humanize_duration, camel_to_underscore


class HumanizeDurationTestCase(unittest.TestCase):
    cases = [
        (0.0, '0 sec'),
        (1e-8, '1e-08 sec'),
        (0.1, '0.1 sec'),
        (1.0, '1 sec'),
        (1, '1 sec'),
        (1.1, '1.1 secs'),
        (59, '59 secs'),
        (59.9, '59.9 secs'),
        (60, '1 min'),
        (61, '1 min 1 sec'),
        (62, '1 min 2 secs'),
        (119, '1 min 59 secs'),
        (120, '2 mins'),
        (121, '2 mins 1 sec'),
        (122, '2 mins 2 secs'),
        (3599, '59 mins 59 secs'),
        (3600, '1 hr'),
        (3601, '1 hr 1 sec'),
        (3661, '1 hr 1 min 1 sec'),
        (86399, '23 hrs 59 mins 59 secs'),
        (86400, '1 day'),
        (86401, '1 day 1 sec'),
        (172799, '1 day 23 hrs 59 mins 59 secs'),
        (259199, '2 days 23 hrs 59 mins 59 secs'),
    ]

    def test_positive(self):
        for seconds, answer in self.cases:
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )

    def test_negative(self):
        for seconds, answer in self.cases[1:]:
            seconds = -seconds
            answer = answer + ' ago'
            result = humanize_duration(seconds)
            self.assertEqual(
                result, answer,
                msg='humanize_duraion({!r}) is expected to be {!r}, '
                    'but got {!r}.'.format(seconds, answer, result)
            )


class CamelToUnderscoreTestCase(unittest.TestCase):
    def assert_convert(self, camel, underscore):
        self.assertEqual(
            camel_to_underscore(camel),
            underscore,
            msg='%r should be converted to %r.' % (camel, underscore)
        )

    def test_camel_to_underscore(self):
        examples = [
            ('simpleTest', 'simple_test'),
            ('easy', 'easy'),
            ('HTML', 'html'),
            ('simpleXML', 'simple_xml'),
            ('PDFLoad', 'pdf_load'),
            ('startMIDDLELast', 'start_middle_last'),
            ('AString', 'a_string'),
            ('Some4Numbers234', 'some4_numbers234'),
            ('TEST123String', 'test123_string'),
        ]
        for camel, underscore in examples:
            self.assert_convert(camel, underscore)
            self.assert_convert(underscore, underscore)
            self.assert_convert('_%s_' % camel, '_%s_' % underscore)
            self.assert_convert('_%s_' % underscore, '_%s_' % underscore)
            self.assert_convert('__%s__' % camel, '__%s__' % underscore)
            self.assert_convert('__%s__' % underscore, '__%s__' % underscore)
            self.assert_convert(
                '_'.join([s.capitalize() for s in underscore.split('_')]),
                underscore
            )
            self.assert_convert(
                '_'.join([s.upper() for s in underscore.split('_')]),
                underscore
            )
