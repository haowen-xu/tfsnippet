import codecs
import json

import os
import six

__all__ = ['write_result']


def write_result(result_dict):
    """
    Print the `result_dict` to screen, and write to an external JSON file
    if ``env["EXPERIMENT_RESULT_FILE"]`` presents.

    The ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]`` would be set if the program
    is run via `mlrun` from MLToolkit. See
    `MLToolkit <https://github.com/haowen-xu/mltoolkit>`_ for details.

    Args:
        result_dict (dict[str, any]): JSON serializable result dict.

    """
    print('')
    print('Final Result')
    print('------------')
    for k, v in six.iteritems(result_dict):
        print('{}: {}'.format(k, v))

    result_json_file = os.environ.get('MLTOOLKIT_EXPERIMENT_RESULT_FILE')
    if result_json_file:
        json_result = json.dumps(result_dict, sort_keys=True)
        with codecs.open(result_json_file, 'wb', 'utf-8') as f:
            f.write(json_result)
