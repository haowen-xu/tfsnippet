import re
import unittest
from argparse import ArgumentParser

import pytest
import six

from tfsnippet.utils import *


class ConfigValidatorTestCase(unittest.TestCase):

    def test_int_validator(self):
        v = IntConfigValidator()

        self.assertEqual(v.validate(123), 123)
        self.assertEqual(v.validate(123.), 123)
        self.assertEqual(v.validate('123'), 123)

        with pytest.raises(TypeError, match='casting a float number into '
                                            'integer is not allowed'):
            _ = v.validate(123.5)

        with pytest.raises(ValueError, match='invalid literal for int'):
            _ = v.validate('xxx')

        with pytest.raises(TypeError, match='123.0? is not an integer'):
            _ = v.validate(123., strict=True)

    def test_float_validator(self):
        v = FloatConfigValidator()

        self.assertEqual(v.validate(123), 123.)
        self.assertEqual(v.validate(123.), 123.)
        self.assertEqual(v.validate(123.5), 123.5)
        self.assertEqual(v.validate('123.5'), 123.5)

        with pytest.raises(ValueError,
                           match='could not convert string to float'):
            _ = v.validate('xxx')

        with pytest.raises(TypeError, match='123 is not a float number'):
            _ = v.validate(123, strict=True)

    def test_bool_validator(self):
        v = BoolConfigValidator()

        self.assertEqual(v.validate(True), True)
        self.assertEqual(v.validate('TRUE'), True)
        self.assertEqual(v.validate('On'), True)
        self.assertEqual(v.validate('yes'), True)
        self.assertEqual(v.validate(1), True)

        self.assertEqual(v.validate(False), False)
        self.assertEqual(v.validate('false'), False)
        self.assertEqual(v.validate('OFF'), False)
        self.assertEqual(v.validate('No'), False)
        self.assertEqual(v.validate(0), False)

        with pytest.raises(TypeError,
                           match='\'xxx\' cannot be casted into boolean'):
            _ = v.validate('xxx')

        with pytest.raises(TypeError, match='1 is not a boolean'):
            _ = v.validate(1, strict=True)

    def test_str_validator(self):
        v = StrConfigValidator()

        self.assertEqual(v.validate(''), '')
        self.assertEqual(v.validate('text'), 'text')
        self.assertEqual(v.validate(123), '123')
        self.assertEqual(v.validate(True), 'True')
        self.assertEqual(v.validate(None), 'None')

        with pytest.raises(TypeError, match='1 is not a string'):
            _ = v.validate(1, strict=True)

    def test_get_config_validator(self):
        class _MyType(object): pass
        class _MySubType(_MyType): pass
        class _MyValidator(ConfigValidator): pass

        self.assertIsInstance(get_config_validator(int), IntConfigValidator)
        self.assertIsInstance(get_config_validator(float), FloatConfigValidator)
        self.assertIsInstance(get_config_validator(bool), BoolConfigValidator)
        self.assertIsInstance(get_config_validator(str), StrConfigValidator)
        self.assertIsInstance(get_config_validator(six.binary_type),
                              StrConfigValidator)
        self.assertIsInstance(get_config_validator(six.text_type),
                              StrConfigValidator)

        with pytest.raises(TypeError, match='No validator has been registered '
                                            'for `type`'):
            _ = get_config_validator(_MyType)

        register_config_validator(_MyType, _MyValidator)
        self.assertIsInstance(get_config_validator(_MyType), _MyValidator)
        self.assertIsInstance(get_config_validator(_MySubType), _MyValidator)


class ConfigFieldTestCase(unittest.TestCase):

    def test_config_field(self):
        with pytest.raises(ValueError,
                           match='`nullable` is False, but `default` value is '
                                 'not specified'):
            _ = ConfigField(int)

        with pytest.raises(ValueError, match=r'Invalid value for `default`: '
                                             r'0 is not one of \[1, 2, 3\]'):
            _ = ConfigField(int, default=0, choices=(1, 2, 3))

        # test int with default value
        field = ConfigField(int, default=123)
        self.assertEqual(field.type, int)
        self.assertEqual(field.default_value, 123)
        self.assertEqual(field.validate(456), 456)
        self.assertIsNone(field.description)
        self.assertFalse(field.nullable)
        self.assertIsNone(field.choices)

        with pytest.raises(ValueError, match='null value is not allowed'):
            _ = field.validate(None)

        with pytest.raises(ValueError, match='invalid literal for int'):
            _ = field.validate('xxx')

        # test bool nullable
        field = ConfigField(bool, nullable=True, description='desc')
        self.assertEqual(field.type, bool)
        self.assertIsNone(field.default_value)
        self.assertEqual(field.validate('yes'), True)
        self.assertEqual(field.description, 'desc')
        self.assertTrue(field.nullable)
        self.assertIsNone(field.validate(None, strict=True))

        # test str, choices
        field = ConfigField(str, default='123', choices=['123', '456', '789'])
        self.assertEqual(field.type, str)
        self.assertEqual(field.default_value, '123')
        self.assertEqual(field.validate('456'), '456')
        self.assertFalse(field.nullable)
        self.assertEqual(field.choices, ('123', '456', '789'))

        with pytest.raises(ValueError, match=r"\'xxx\' is not one of \['123', "
                                             r"'456', '789'\]"):
            _ = field.validate('xxx')

        with pytest.raises(ValueError, match='null value is not allowed'):
            _ = field.validate(None)

        # test str, choices, nullable
        field = ConfigField(str, choices=['123', '456', '789'], nullable=True)
        self.assertEqual(field.type, str)
        self.assertIsNone(field.default_value)
        self.assertTrue(field.nullable)
        self.assertIsNone(field.validate(None))


class ConfigTestCase(unittest.TestCase):

    def test_config(self):
        class ParentConfig(Config):
            x = 123

            @property
            def x_property(self):
                return self.x

        class MyConfig(ParentConfig):
            y = ConfigField(str, nullable=True, choices=['abc', 'def'])
            z = ConfigField(int, default=1000, description='the z field')
            w = ['w']

            def get_x(self):
                return self.x

        # test get defaults from class
        defaults = {'x': 123, 'y': None, 'z': 1000, 'w': ['w']}
        self.assertDictEqual(get_config_defaults(MyConfig), defaults)

        # test populate the default values
        config = MyConfig()
        self.assertEqual(config.x, 123)
        self.assertIsNone(config.y)
        self.assertEqual(config.z, 1000)
        self.assertEqual(config.x_property, 123)
        self.assertEqual(config.get_x(), 123)

        # test cannot set a ConfigField to attribute
        with pytest.raises(TypeError, match='`value` must not be a '
                                            'ConfigField object'):
            config.x = ConfigField(int, default=0)
        with pytest.raises(TypeError, match='`value` must not be a '
                                            'ConfigField object'):
            config['ww'] = ConfigField(int, default=0)

        # test scan through the default values
        keys = ['w', 'x', 'y', 'z']
        self.assertListEqual(sorted(config), keys)
        for k in keys:
            self.assertIn(k, config)
            self.assertEqual(config[k], getattr(config, k))
        self.assertNotIn('xyz', config)

        # test default values to_dict
        self.assertDictEqual(get_config_defaults(config), defaults)
        self.assertDictEqual(config.to_dict(), defaults)

        # test set attr on recognized simple value
        config.x = 456
        self.assertEqual(config.x, 456)
        self.assertEqual(config['x'], 456)
        config['x'] = 789
        self.assertEqual(config.x, 789)
        self.assertEqual(config['x'], 789)

        with pytest.raises(ValueError, match='invalid literal for int'):
            config.x = 'xxx'
        with pytest.raises(ValueError, match='invalid literal for int'):
            config['x'] = 'xxx'
        with pytest.raises(ValueError, match='null value is not allowed'):
            config.x = None
        with pytest.raises(ValueError, match='null value is not allowed'):
            config['x'] = None

        # test set attr on nullable config field
        config.y = 'abc'
        self.assertEqual(config.y, 'abc')
        self.assertEqual(config['y'], 'abc')
        config.y = None
        self.assertIsNone(config.y)
        self.assertIsNone(config['y'])
        config['y'] = None
        self.assertIsNone(config.y)
        self.assertIsNone(config['y'])
        config['y'] = 'def'
        self.assertEqual(config.y, 'def')
        self.assertEqual(config['y'], 'def')

        with pytest.raises(ValueError, match=r"not one of \['abc', 'def'\]"):
            config.y = 'xxx'
        with pytest.raises(ValueError, match=r"not one of \['abc', 'def'\]"):
            config['y'] = 'xxx'

        # test set attr on non-nullable config field
        config.z = 2000
        self.assertEqual(config.z, 2000)
        self.assertEqual(config['z'], 2000)
        config['z'] = 3000
        self.assertEqual(config.z, 3000)
        self.assertEqual(config['z'], 3000)

        with pytest.raises(ValueError, match='invalid literal for int'):
            config.z = 'xxx'
        with pytest.raises(ValueError, match='invalid literal for int'):
            config['z'] = 'xxx'
        with pytest.raises(ValueError, match='null value is not allowed'):
            config.z = None
        with pytest.raises(ValueError, match='null value is not allowed'):
            config['z'] = None

        # test set attr on unrecognized simple value
        config.w = {'w': 'ww'}
        self.assertDictEqual(config.w, {'w': 'ww'})
        self.assertDictEqual(config['w'], {'w': 'ww'})
        config['w'] = {'ww': 'www'}
        self.assertDictEqual(config.w, {'ww': 'www'})
        self.assertDictEqual(config['w'], {'ww': 'www'})

        # test set attr on in-exist attr
        config.t = 'tt'
        self.assertEqual(config.t, 'tt')
        self.assertEqual(config['t'], 'tt')
        config['t'] = 'ttt'
        self.assertEqual(config.t, 'ttt')
        self.assertEqual(config['t'], 'ttt')

        # test get in-exist attr
        with pytest.raises(AttributeError, match="has no attribute 'xyz'"):
            _ = config.xyz
        with pytest.raises(KeyError, match='`xyz` is not a config key'):
            _ = config['xyz']

        # test set reserved key
        with pytest.raises(KeyError, match='`_xyz` is reserved'):
            config['_xyz'] = 1234

        with pytest.raises(KeyError, match='`to_dict` is reserved'):
            config['to_dict'] = 1234

        config._xyz = 5678  # set attr can work, but will not be a config key
        self.assertEqual(config._xyz, 5678)

        # test scan through the default values, and to dict
        keys = ['t', 'w', 'x', 'y', 'z']
        self.assertListEqual(sorted(config), keys)
        for k in keys:
            self.assertIn(k, config)
            self.assertEqual(config[k], getattr(config, k))
        self.assertNotIn('xyz', config)
        self.assertNotIn('_xyz', config)
        self.assertDictEqual(
            config.to_dict(),
            {'x': 789, 'y': 'def', 'z': 3000, 'w': {'ww': 'www'}, 't': 'ttt'}
        )
        self.assertDictEqual(get_config_defaults(config), defaults)

        # test update
        config = MyConfig()
        config.update({'x': 456})
        config.update([('y', 'def'), ('z', 2000)])
        self.assertDictEqual(
            config.to_dict(),
            {'x': 456, 'y': 'def', 'z': 2000, 'w': ['w']}
        )

        # test config_defaults on non-Config object
        class NotAConfig(object):
            pass

        with pytest.raises(TypeError,
                           match='`config` must be an instance of `Config`, '
                                 'or a subclass of `Config`'):
            _ = get_config_defaults(NotAConfig)

        with pytest.raises(TypeError,
                           match='`config` must be an instance of `Config`, '
                                 'or a subclass of `Config`'):
            _ = get_config_defaults(NotAConfig())

    def test_register_config_arguments(self):
        class ModelConfig(Config):
            activation = ConfigField(str, nullable=True,
                                     choices=['relu', 'leaky_relu'])
            normalizer = None

        class TrainConfig(Config):
            max_step = ConfigField(int, default=1000,
                                   description='Maximum step to run.')
            max_epoch = 1

        # test errors
        with pytest.raises(TypeError, match='`config` is not an instance of '
                                            '`Config`'):
            register_config_arguments(ModelConfig, ArgumentParser())
        with pytest.raises(TypeError, match='`config` is not an instance of '
                                            '`Config`'):
            register_config_arguments(object(), ArgumentParser())
        with pytest.raises(ValueError, match='`title` is required when '
                                             '`description` is specified'):
            register_config_arguments(
                ModelConfig(), ArgumentParser(), description='abc')

        # test no title
        parser = ArgumentParser()
        model_config = ModelConfig()
        train_config = TrainConfig()

        register_config_arguments(model_config, parser, prefix='model')
        register_config_arguments(train_config, parser, sort_keys=True)

        help = parser.format_help()
        self.assertTrue(
            re.match(
                r".*"
                r"--model\.activation\s+MODEL\.ACTIVATION\s+"
                r"\(default None; choices \['leaky_relu', 'relu'\]\)\s+"
                r"--model\.normalizer\s+MODEL\.NORMALIZER\s+"
                r"\(default None\)\s+"
                r"--max_epoch\s+MAX_EPOCH\s+"
                r"\(default 1\)\s+"
                r"--max_step\s+MAX_STEP\s+"
                r"Maximum step to run. \(default 1000\)"
                r".*",
                help,
                re.S
            )
        )
        args = parser.parse_args([
            '--max_epoch=123',
            '--max_step=456',
            '--model.activation=relu'
        ])
        self.assertEqual(model_config.activation, 'relu')
        self.assertIsNone(model_config.normalizer)
        self.assertEqual(train_config.max_epoch, 123)
        self.assertEqual(train_config.max_step, 456)
        self.assertEqual(getattr(args, 'model.activation'), 'relu')
        self.assertIsNone(getattr(args, 'model.normalizer'))
        self.assertEqual(args.max_epoch, 123)
        self.assertEqual(args.max_step, 456)

        # test has title
        parser = ArgumentParser()
        model_config = ModelConfig()
        train_config = TrainConfig()

        register_config_arguments(model_config, parser, title='Model options')
        register_config_arguments(train_config, parser, prefix='train',
                                  sort_keys=True, title='Train options')

        help = parser.format_help()
        self.assertTrue(
            re.match(
                r".*"
                r"Model options:\s+"
                r"--activation\s+ACTIVATION\s+"
                r"\(default None; choices \['leaky_relu', 'relu'\]\)\s+"
                r"--normalizer\s+NORMALIZER\s+"
                r"\(default None\)\s+"
                r"Train options:\s+"
                r"--train\.max_epoch\s+TRAIN\.MAX_EPOCH\s+"
                r"\(default 1\)\s+"
                r"--train\.max_step\s+TRAIN\.MAX_STEP\s+"
                r"Maximum step to run. \(default 1000\)"
                r".*",
                help,
                re.S
            )
        )
        args = parser.parse_args([
            '--train.max_epoch=123',
            '--normalizer=789'
        ])
        self.assertIsNone(model_config.activation)
        # note the value is parsed as yaml, thus should be 789 rather than '789'
        self.assertEqual(model_config.normalizer, 789)
        self.assertEqual(train_config.max_epoch, 123)
        self.assertEqual(train_config.max_step, 1000)
        self.assertIsNone(args.activation)
        self.assertEqual(args.normalizer, 789)
        self.assertEqual(getattr(args, 'train.max_epoch'), 123)
        self.assertEqual(getattr(args, 'train.max_step'), 1000)

        # test parse error
        parser = ArgumentParser()
        model_config = ModelConfig()
        register_config_arguments(model_config, parser, prefix='model')

        with pytest.raises(Exception, match=r"Invalid value for argument "
                                            r"`--model.activation`; 'sigmoid' "
                                            r"is not one of \['relu', "
                                            r"'leaky_relu'\]\."):
            _ = parser.parse_args(['--model.activation=sigmoid'])

    def test_scoped_set_config(self):
        class MyConfig(Config):
            a = 123
            b = None

        config = MyConfig()
        self.assertDictEqual(config.to_dict(), {'a': 123, 'b': None})
        with scoped_set_config(config, a=456, c='hello'):
            self.assertDictEqual(config.to_dict(),
                                 {'a': 456, 'b': None, 'c': 'hello'})

            with scoped_set_config(config, b=789.):
                self.assertDictEqual(config.to_dict(),
                                     {'a': 456, 'b': 789., 'c': 'hello'})

            self.assertDictEqual(config.to_dict(),
                                 {'a': 456, 'b': None, 'c': 'hello'})
        self.assertDictEqual(config.to_dict(), {'a': 123, 'b': None})
