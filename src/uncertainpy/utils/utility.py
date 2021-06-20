from __future__ import absolute_import, division, print_function, unicode_literals

import six
import warnings
import numpy as np


def set_nan(values, index):
    """
    Set the index of a arbitrarly nested list to nan

    Parameters
    ----------
    values : array_like, list, number
        Values where to set  index to ``numpy.nan``. Can be irregular and have
        any number of nested elements.
    index : array_like, list, number
        Index where to set `values` to ``numpy.nan``.
    """
    if hasattr(index, "__iter__"):
        if(len(index) == 1):
            values[index[0]] = np.nan
        else:
            set_nan(values[index[0]], index[1:])
    else:
        values[index] = np.nan



def none_to_nan(values):
    """
    Converts ``None`` values in `values` to ``np.nan``.

    Parameters
    ----------
    values : array_like, list, number
        Values where to convert occurrences of ``None`` converted to ``np.nan``.
        Can be irregular and have any number of nested elements.

    Returns
    -------
    values : array_like, list, number
        `values` with all occurrences of ``None`` converted to ``np.nan``.
    """
    if values is None:
        values = np.nan
    elif isinstance(values, six.string_types):
        pass

    elif isinstance(values, np.ndarray):
        if values.dtype == "object":
            try:
                return values.astype(float)
            except ValueError:
                for i, value in enumerate(values):
                    values[i] = none_to_nan(value)
        else:
            return values

    elif hasattr(values, "__iter__"):
        try:
            values_array = np.array(values, dtype=float)
            indices = np.argwhere(np.isnan(values_array))

            for idx in indices:
                set_nan(values, idx)

        except ValueError:
            for i, value in enumerate(values):
                values[i] = none_to_nan(value)

    return values


def contains_nan(values):
    """
    Checks if ``None`` or ``numpy.nan`` exists in `values`. Returns ``True`` if
    any there are at least one occurrence of ``None`` or ``numpy.nan``.

    Parameters
    ----------
    values : array_like, list, number
        `values` where to check for occurrences of ``None`` or ``np.nan``.
        Can be irregular and have any number of nested elements.

    Returns
    -------
    bool
        ``True`` if `values` has at least one occurrence of ``None`` or
        ``numpy.nan``.
    """
    # To speed up we first try the fast option np.any(np.isnan(values))
    try:
        return np.any(np.isnan(values))
    except (ValueError, TypeError):
        if values is None or values is np.nan:
            return True
        # To solve the problem of float/int as well as numpy int/flaot
        elif np.isscalar(values) and np.isnan(values):
            return True
        elif hasattr(values, "__iter__"):
            for value in values:
                if contains_nan(value):
                    return True

            return False
        else:
            return False



# Not working, but currently not needed
# def only_none_or_nan(values):
#     """
#     Checks if `values` only contains``None`` and/or ``numpy.nan``. Returns
#     ``True`` if `values` only contains``None`` and/or ``numpy.nan``.

#     Parameters
#     ----------
#     values : array_like, list, number
#         `values` where to check for occurrences of ``None`` or ``np.nan``.
#         Can be irregular and have any number of nested elements.

#     Returns
#     -------
#     bool
#         ``True`` if `values`  only contains ``None`` and/or ``numpy.nan``.
#     """
#     # To speed up we first try the fast option np.all(np.isnan(values))
#     try:
#         return np.all(np.isnan(values))
#     except (ValueError, TypeError):
#         print "valies", values
#         if hasattr(values, "__iter__"):
#             for value in values:
#                 if not only_none_or_nan(value):
#                     return True

#             return False
#         # To solve the problem of numpy float and int
#         elif np.isscalar(values) and not np.isnan(values):
#             return False
#         elif values is not None and values is not np.nan:
#             return False

#         else:
#             return True




def lengths(values):
    """
    Get the lengths of a list and all its sublists.

    Parameters
    ----------
    values : list
        List where we want to find the lengths of the list and all sublists.

    Returns
    -------
    list
        A list with the lengths of the list and all sublists.
    """
    lengths = []

    def recursive_len(values, lengths):
        if hasattr(values, "__iter__"):
            lengths.append(len(values))

            for value in values:
                recursive_len(value, lengths)

    recursive_len(values, lengths)

    return lengths




def is_regular(values):
    """
    Test if `values` is regular or not, meaning it has a varying length of
    nested elements.

    Parameters
    ----------
    values : array_like, list, number
        `values` to check if it is regular or not, meaning it has a varying
        length of nested elements.

    Returns
    -------
    bool
        True if the feature is regular or False if the feature is irregular.

    Notes
    -----
    Does not ignore ``numpy.nan``, so ``[numpy.nan, [1, 2]]`` returns False.
    """

    try:
        np.array(values, dtype=float)
    except ValueError:
        return False

    return True




###################
# Not used anymore
###################

# def none_to_nan_regularize(values):
#     """
#     Converts None values in `values` to a arrays of numpy.nan.

#     If `values` is a 2 dimensional or above array, each instance of None is converted to an
#     array of numpy.nan of the correct shape, which makes the array regular.


#     Parameters
#     ----------
#     values : array_like
#         Result from model or features. Can be of any dimensions.

#     Returns
#     -------
#     array
#         Array with all None converted to arrays of NaN of the correct shape.


#     Examples
#     --------
#     >>> from uncertainpy import Parallel
#     >>> parallel = Parallel()
#     >>> U_irregular = np.array([None, np.array([None, np.array([1, 2, 3]), None, np.array([1, 2, 3])])])
#     >>> result = parallel.none_to_nan(U_irregular)
#         array([[[ nan,  nan,  nan],
#                 [ nan,  nan,  nan],
#                 [ nan,  nan,  nan],
#                 [ nan,  nan,  nan]],
#                 [[ nan,  nan,  nan],
#                 [  1.,   2.,   3.],
#                 [ nan,  nan,  nan],
#                 [  1.,   2.,   3.]]])
#     """
#     warnings.warn(
#         "regularize_nan_results is no longer used as nan results no longer are required to be regular.",
#         DeprecationWarning
#     )

#     is_array = False
#     if isinstance(values, np.ndarray):
#         is_array = True
#         values = values.tolist()

#     if values is None:
#         values = np.nan
#     # elif hasattr(values, "__iter__") and len(values) == 0:
#     #     values_list = np.nan
#     else:
#         # To handle the special case of 0d arrays,
#         # which have an __iter__, but cannot be iterated over
#         try:
#             for i, value in enumerate(values):
#                 if hasattr(value, "__iter__"):
#                     values[i] = none_to_nan_regularize(value)

#             fill = np.nan
#             for i, value in enumerate(values):
#                 if value is not None:
#                     fill = np.full(np.shape(values[i]), np.nan, dtype=float).tolist()
#                     break

#             for i, value in enumerate(values):
#                 if value is None:
#                     values[i] = fill

#         except TypeError:
#             if is_array:
#                 value = np.array(values)

#             return values

#     if is_array:
#         value = np.array(values)

#     return values


def create_model_parameters(nodes, uncertain_parameters):
        """
        Combine nodes (values) with the uncertain parameter names to create a
        list of dictionaries corresponding to the model values for each
        model evaluation.
        Parameters
        ----------
        nodes : array
            A series of different set of parameters. The model and each feature is
            evaluated for each set of parameters in the series.
        uncertain_parameters : list
            A list of names of the uncertain parameters.
        Returns
        -------
        model_parameters : list
             A list where each element is a dictionary with the model parameters
             for a single evaluation.
        """
        model_parameters = []
        for node in nodes.T:
            if node.ndim == 0:
                node = [node]
            # New set parameters
            parameters = {}
            for j, parameter in enumerate(uncertain_parameters):
                parameters[parameter] = node[j]
            model_parameters.append(parameters)
        return model_parameters
