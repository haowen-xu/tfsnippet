import ast
import codecs
import os


def parse_all_list(python_file):
    with codecs.open(python_file, 'rb', 'utf-8') as f:
        tree = ast.parse(f.read(), python_file)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                node.targets[0].id == '__all__':
            return list(ast.literal_eval(node.value))


def process_dir(module_dir):
    module_all_list = []
    module_import_list = []

    for name in os.listdir(module_dir):
        path = os.path.join(module_dir, name)
        if name.endswith('.py') and name != '__init__.py':
            all_list = parse_all_list(path)
            if all_list is None:
                print('Warning: cannot parse __all__ list of: {}'.format(path))
            else:
                module_all_list.extend(all_list)
                module_import_list.append(name[:-3])

        elif name not in ('__pycache__',) and os.path.isdir(path):
            all_list = process_dir(path)
            if all_list:
                module_all_list.extend(all_list)
                module_import_list.append(name)

    module_import_list.sort()
    module_all_list.sort()

    module_all_list_lines = ['    ']
    for n in module_all_list:
        new_s = module_all_list_lines[-1] + repr(n) + ', '
        if len(new_s) >= 81:
            module_all_list_lines.append('    {!r}, '.format(n))
        else:
            module_all_list_lines[-1] = new_s
    module_all_list_lines = [s.rstrip() for s in module_all_list_lines
                             if s.strip()]

    init_content = '\n'.join(
        ['from .{} import *'.format(n) for n in module_import_list] +
        [''] +
        ['__all__ = ['] +
        module_all_list_lines +
        [']'] +
        ['']
    )

    module_init_file = os.path.join(module_dir, '__init__.py')
    with codecs.open(module_init_file, 'wb', 'utf-8') as f:
        f.write(init_content)
    print(module_init_file)

    return module_all_list


if __name__ == '__main__':
    tfsnippet_root = os.path.join(
        os.path.split(os.path.abspath(__file__))[0], '../tfsnippet')
    for name in os.listdir(tfsnippet_root):
        path = os.path.join(tfsnippet_root, name)
        if name not in ('examples', '__pycache__') and os.path.isdir(path):
            process_dir(path)
