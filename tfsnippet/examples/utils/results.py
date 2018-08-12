import codecs
import json
import os
import sys

import six

from tfsnippet.utils import makedirs
from .jsonutils import JsonEncoder

__all__ = ['Results']


class Results(object):
    """
    Class to help save the results of an experiment.

    If ``env["MLSTORAGE_EXPERIMENT_ID"]`` does not present, the results
    will be saved to ``os.getcwd() + "/results/" + name``.  Otherwise the
    results will be saved directly in the current working directory.
    """

    def __init__(self, name=None):
        """
        Construct a :class:`ExpResult` object.

        Args:
            name (str): Name of the experiment.  If not specified, will use
                the name of the main script.
        """
        if not name:
            main_script = sys.modules['__main__'].__file__
            if not main_script:
                raise ValueError('Failed to infer the name automatically.')
            name = os.path.splitext(os.path.split(main_script)[1])[0]

        if os.environ.get('MLSTORAGE_EXPERIMENT_ID'):
            result_dir = os.path.abspath(os.getcwd())
        else:
            result_dir = os.path.abspath(os.path.join('./results', name))

        self._name = name
        self._result_dir = result_dir
        self._result_json_file = os.path.join(result_dir, 'result.json')
        self._result_dict = {}

    @property
    def name(self):
        """Get the name of this experiment."""
        return self._name

    @property
    def result_dir(self):
        """Get the path of the result directory."""
        return self._result_dir

    @property
    def result_json_file(self):
        """Get the path of the result JSON file."""
        return self._result_json_file

    @property
    def result_dict(self):
        """Get the latest result dict."""
        return self._result_dict

    def commit(self, result_dict):
        """
        Update the results with `result_dict`, and save the merged results
        to "result.json".

        Args:
            result_dict (dict):  JSON serializable result dict.
                It will be merged with ``self.result_dict``.
        """
        self.result_dict.update(result_dict)
        parent_dir = os.path.split(self.result_json_file)[0]
        makedirs(parent_dir, exist_ok=True)
        json_result = json.dumps(self.result_dict, sort_keys=True,
                                 cls=JsonEncoder)
        with codecs.open(self.result_json_file, 'wb', 'utf-8') as f:
            f.write(json_result)

    def commit_and_print(self, result_dict):
        """
        Update the results with `result_dict`, save the merged results
        to "result.json", and print the result to screen.

        Args:
            result_dict (dict):  JSON serializable result dict.
                It will be merged with ``self.result_dict``.
        """
        self.commit(result_dict)
        print('=======')
        print('Results')
        print('-------')
        for k in sorted(six.iterkeys(self.result_dict)):
            print('{}: {}'.format(k, self.result_dict[k]))

    def resolve_path(self, sub_path):
        """
        Resolve the full path of `sub_path` under the result directory.

        Args:
            sub_path (str): The sub path.

        Returns:
            str: The resolved full path.
        """
        return os.path.join(self.result_dir, sub_path)

    def make_dir(self, sub_path):
        """
        Ensure the `sub_path` directory exists.

        Args:
            sub_path (str): The sub path.

        Returns:
            str: The full path of the directory.
        """
        path = self.resolve_path(sub_path)
        makedirs(path, exist_ok=True)
        return path

    def prepare_parent(self, sub_path):
        """
        Ensure the parent directory of `sub_path` exists.

        Args:
            sub_path (str): The sub path.

        Returns:
            str: The full path of `sub_path`.
        """
        path = self.resolve_path(sub_path)
        parent_dir = os.path.split(path)[0]
        makedirs(parent_dir, exist_ok=True)
        return path
