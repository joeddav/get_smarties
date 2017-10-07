import numpy as np
import pandas as pd
from pandas import compat
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from pandas.core.indexing import is_list_like
from pandas.core.categorical import _factorize_from_iterable

class Smarties:
    def __init__(self, main_lookup=None):
        self.main_lookup=main_lookup
        return None

    def transform(self, df):
        result_lookup = self.main_lookup

        try:
            df = df[result_lookup['normal']]
        except:
            list1 = result_lookup['normal']
            list2 = df.columns

            for i in list1:
                if i in list2:
                    print('ok',i)
                else:
                    print('missing!',i)

            raise Exception('You are missing a column key, should be:' + str(result_lookup['normal']))

        encoding_lookup = result_lookup['encoding']
        row = df.index
        with_dummies = [df.drop(encoding_lookup.keys(), axis=1)]   #drop columns to encode

        for key in encoding_lookup:
            value_name = df[key].values[0]
            #print value_name
            #Check to see if encoding took place
            number_of_cols = len(encoding_lookup[key])
            dummy_mat = np.zeros((1, number_of_cols), dtype=np.uint8)

            indices = [i for i, s in enumerate(encoding_lookup[key]) if key + '_' + str(value_name) == s]
            if len(indices) > 0:
                dummy_mat[0][indices[0]] = 1

            with_dummies.append(DataFrame(dummy_mat, index=row, columns=encoding_lookup[key]))
        return pd.concat(with_dummies, axis=1)


    def fit_transform(self, data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False):
        """
        Convert categorical variable into dummy/indicator variables
        """
        #from pandas.core.reshape.concat import concat
        from itertools import cycle

        if 'DataFrame' not in str(type(data)):  #convert series to dataframe
            data = data.to_frame()

        main_lookup={}
        main_lookup['normal'] = data.columns

        if isinstance(data, DataFrame):
            # determine columns being encoded

            if columns is None:
                columns_to_encode = data.select_dtypes(
                    include=['object', 'category']).columns
            else:
                columns_to_encode = columns

            # validate prefixes and separator to avoid silently dropping cols
            def check_len(item, name):
                length_msg = ("Length of '{0}' ({1}) did not match the length of "
                              "the columns being encoded ({2}).")

                if is_list_like(item):
                    if not len(item) == len(columns_to_encode):
                        raise ValueError(length_msg.format(name, len(item),
                                                           len(columns_to_encode)))

            check_len(prefix, 'prefix')
            check_len(prefix_sep, 'prefix_sep')
            if isinstance(prefix, compat.string_types):
                prefix = cycle([prefix])
            if isinstance(prefix, dict):
                prefix = [prefix[col] for col in columns_to_encode]

            if prefix is None:
                prefix = columns_to_encode

            # validate separators
            if isinstance(prefix_sep, compat.string_types):
                prefix_sep = cycle([prefix_sep])
            elif isinstance(prefix_sep, dict):
                prefix_sep = [prefix_sep[col] for col in columns_to_encode]

            if set(columns_to_encode) == set(data.columns):
                with_dummies = []
            else:
                with_dummies = [data.drop(columns_to_encode, axis=1)]

            #print(with_dummies)
            encoding_lookup={}
            for (col, pre, sep) in zip(columns_to_encode, prefix, prefix_sep):

                dummy = self._get_dummies_1d(data[col], prefix=pre, prefix_sep=sep,
                                        dummy_na=dummy_na, sparse=sparse,
                                        drop_first=drop_first)

                encoding_lookup[col]=dummy.columns
                with_dummies.append(dummy)

            main_lookup['encoding'] = encoding_lookup
            result = pd.concat(with_dummies, axis=1)
        else:
            result = self._get_dummies_1d(data, prefix, prefix_sep, dummy_na,
                                     sparse=sparse, drop_first=drop_first)
        self.main_lookup = main_lookup  #save class variables
        return result #, dummy, columns_to_encode, main_lookup

    def _get_dummies_1d(self, data, prefix, prefix_sep='_', dummy_na=False,
                        sparse=False, drop_first=False):
        # Series avoids inconsistent NaN handling
        codes, levels = _factorize_from_iterable(Series(data))

        def get_empty_Frame(data, sparse):
            if isinstance(data, Series):
                index = data.index
            else:
                index = np.arange(len(data))
            if not sparse:
                return DataFrame(index=index)
            else:
                return SparseDataFrame(index=index, default_fill_value=0)

        # if all NaN
        if not dummy_na and len(levels) == 0:
            return get_empty_Frame(data, sparse)

        codes = codes.copy()
        if dummy_na:
            codes[codes == -1] = len(levels)
            levels = np.append(levels, np.nan)

        # if dummy_na, we just fake a nan level. drop_first will drop it again
        if drop_first and len(levels) == 1:
            return get_empty_Frame(data, sparse)

        number_of_cols = len(levels)

        if prefix is not None:
            dummy_cols = ['%s%s%s' % (prefix, prefix_sep, v) for v in levels]
        else:
            dummy_cols = levels

        if isinstance(data, Series):
            index = data.index
        else:
            index = None

        if sparse:
            sparse_series = {}
            N = len(data)
            sp_indices = [[] for _ in range(len(dummy_cols))]
            for ndx, code in enumerate(codes):
                if code == -1:
                    # Blank entries if not dummy_na and code == -1, #GH4446
                    continue
                sp_indices[code].append(ndx)

            if drop_first:
                # remove first categorical level to avoid perfect collinearity
                # GH12042
                sp_indices = sp_indices[1:]
                dummy_cols = dummy_cols[1:]
            for col, ixs in zip(dummy_cols, sp_indices):
                sarr = SparseArray(np.ones(len(ixs), dtype=np.uint8),
                                   sparse_index=IntIndex(N, ixs), fill_value=0,
                                   dtype=np.uint8)
                sparse_series[col] = SparseSeries(data=sarr, index=index)

            out = SparseDataFrame(sparse_series, index=index, columns=dummy_cols,
                                  default_fill_value=0,
                                  dtype=np.uint8)
            return out

        else:
            dummy_mat = np.eye(number_of_cols, dtype=np.uint8).take(codes, axis=0)

            if not dummy_na:
                # reset NaN GH4446
                dummy_mat[codes == -1] = 0

            if drop_first:
                # remove first GH12042
                dummy_mat = dummy_mat[:, 1:]
                dummy_cols = dummy_cols[1:]
            return DataFrame(dummy_mat, index=index, columns=dummy_cols)
