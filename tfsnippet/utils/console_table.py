from .config_utils import Config

__all__ = ['ConsoleTable', 'print_as_table']


class ConsoleTable(object):
    """A class to help format a console table."""

    class _Row(list):
        pass

    class _EmptyRow(object):
        pass

    class _HR(object):
        def __init__(self, c):
            self.c = c

    class _TitleRow(object):
        def __init__(self, title, space=1):
            if isinstance(title, tuple):
                assert(len(title) == 2)
                self.title = tuple(str(s) for s in title)
            else:
                self.title = str(title)
            self.space = space

        def format(self, row_width=0):
            if isinstance(self.title, tuple):
                assert(len(self.title) == 2)
                pieces_len = sum(map(len, self.title))
                row_width = max(
                    row_width, pieces_len + self.space * (len(self.title) - 1))
                return (self.title[0] + (' ' * (row_width - pieces_len)) +
                        self.title[1])
            else:
                return self.title

    def __init__(self, col_count, col_space=3, col_align=None, expand_col=0):
        """
        Construct a new :class:`ConsoleTable`.

        Args:
            col_count (int): The number of columns.
            col_space (int): The number of spaces between each column.
            col_align (list[str]): The alignment of each column.
                '<' or 'l' indicates to align left;
                '>' or 'r' indicates to align right;
                '^' or 'c' indicates to align center.
            expand_col (int): The index of column to expand if title width is
                larger than the sum of column width.
        """
        col_count = int(col_count)
        if col_count < 1:
            raise ValueError('`col_count` must be at least 1: got {!r}'.
                             format(col_count))
        col_space = int(col_space)
        if col_space < 1:
            raise ValueError('`col_space` must be at least 1: got {!r}'.
                             format(col_space))
        if col_align is None:
            col_align = ['<'] * col_count
        else:
            col_align = list(col_align)
            for i, a in enumerate(col_align):
                if a not in ('<', '^', '>', 'l', 'c', 'r'):
                    raise ValueError('Invalid alignment: {}'.format(a))
                col_align[i] = {'l': '<', 'c': '^', 'r': '>'}.get(a, a)
            if len(col_align) != col_count:
                raise ValueError('The length of `col_align` must equal to '
                                 '`col_count`: col_align {}, col_count {}'.
                                 format(col_align, col_count))
        expand_col = int(expand_col)

        self._col_count = col_count
        self._col_space = col_space
        self._col_align = col_align
        self._expand_col = expand_col
        self._rows = []

    def add_row(self, columns):
        """
        Add a row with full columns.

        Args:
            columns (Iterable[str]): The column contents.
        """
        row = ConsoleTable._Row(map(str, columns))
        if len(row) != self._col_count:
            raise ValueError('Expect exactly {} columns, but got: {!r}'.
                             format(self._col_count, row))
        self._rows.append(row)

    def add_title(self, title, top_right=None):
        """
        Add row of title.

        Args:
            title (str): The title content.
            top_right (str): Optional top-right content.
        """
        title = str(title)
        if top_right is not None:
            top_right = str(top_right)
            self._rows.append(ConsoleTable._TitleRow((title, top_right)))
        else:
            self._rows.append(ConsoleTable._TitleRow(title))

    def add_hr(self, c='-'):
        """
        Add a horizon line.

        Args:
            c (str): The character of the horizon line.
        """
        c = str(c)
        if len(c) != 1:
            raise ValueError('`c` must be exactly one character: got {!r}'.
                             format(c))
        self._rows.append(ConsoleTable._HR(c))

    def add_skip(self):
        """Add an empty row."""
        self._rows.append(ConsoleTable._EmptyRow())

    def add_key_values(self, key_values, sort_keys=False):
        """
        Add a key-value sequence to the table.

        Args:
            key_values: Dict, or key-value sequence.
            sort_keys (bool): Whether or not to sort the keys?
        """
        if self._col_count != 2:
            raise TypeError('`col_count` of this table is not 2, cannot add '
                            'a key-value sequence.')

        if hasattr(key_values, 'items'):
            key_values = list(key_values.items())
        else:
            key_values = list(key_values)

        if sort_keys:
            key_values.sort(key=lambda x: x[0])

        self.add_skip()
        for key, value in key_values:
            self.add_row([key, value])

    add_dict = add_key_values

    def add_config(self, config, sort_keys=False):
        """
        Add a config to the table.

        Args:
            config (Config): A config object.
            sort_keys (bool): Whether or not to sort the keys?
        """
        self.add_key_values(
            ((key, config[key]) for key in config),
            sort_keys=sort_keys
        )

    def format(self):
        """
        Format the table to string.

        Returns:
            str: The formatted table.
        """
        # first, calculate the width of each column
        widths = [0] * self._col_count
        rol_width = 0
        for row in self._rows:
            if isinstance(row, ConsoleTable._Row):
                for i, col in enumerate(row):
                    widths[i] = max(widths[i], len(col))
            elif isinstance(row, ConsoleTable._TitleRow):
                rol_width = max(rol_width, len(row.format()))
        col_width_sum = sum(widths) + self._col_space * (self._col_count - 1)
        if rol_width > col_width_sum:
            widths[self._expand_col] += rol_width - col_width_sum
        else:
            rol_width = col_width_sum

        # next, format each rows
        ret = []
        col_span = ' ' * self._col_space

        for i, row in enumerate(self._rows):
            if isinstance(row, ConsoleTable._Row):
                assert(isinstance(row, ConsoleTable._Row))
                row_text = []
                for j, col in enumerate(row):
                    s = '{text:{align}{length}}'.format(
                        text=col,
                        align=self._col_align[j],
                        length=widths[j]
                    )
                    if j == len(row) - 1:
                        s = s.rstrip()
                    row_text.append(s)
                ret.append(col_span.join(row_text))

            elif isinstance(row, ConsoleTable._TitleRow):
                ret.append(row.format(rol_width))

            elif isinstance(row, ConsoleTable._HR):
                ret.append(row.c * rol_width)

            else:
                assert(isinstance(row, ConsoleTable._EmptyRow))
                if i > 0 and \
                        not isinstance(
                                self._rows[i - 1],
                                (ConsoleTable._EmptyRow, ConsoleTable._HR)
                            ):
                        ret.append('')

        return '\n'.join(ret)

    def __str__(self):
        return self.format()


def print_as_table(title, key_values, hr='='):
    """
    Print a key-value sequence as a table.

    Args:
        title: Title of the table.
        key_values: Dict, or key-value sequence.
        hr: Character for the horizon line.
    """
    table = ConsoleTable(2)
    table.add_title(title)
    table.add_hr(hr)
    table.add_key_values(key_values)
    print(table.format())
