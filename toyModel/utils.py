def ternary_search_max(function, low, high, epsilon):
    """
    Perform ternary search to find the maximum of a unimodal function.

    Parameters
    ----------
    function : callable
        The function to maximize.
    low : float
        The lower bound of the search.
    high : float
        The upper bound of the search.
    epsilon : float
        The precision of the search.

    Returns
    -------
    tuple
        The maximum value of the function and the corresponding argument.
    """
    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        # Evaluate function at mid1 and mid2
        f_mid1 = function(mid1)
        f_mid2 = function(mid2)

        # Narrow the search range based on the evaluated values
        if f_mid1 > f_mid2:
            high = mid2
        else:
            low = mid1

    mid = (low + high) / 2
    return function(mid), mid


def ternary_search_min(function, low, high, epsilon):
    """
    Perform ternary search to find the minimum of a unimodal function.

    Parameters
    ----------
    function : callable
        The function to minimize.
    low : float
        The lower bound of the search.
    high : float
        The upper bound of the search.
    epsilon : float
        The precision of the search.

    Returns
    -------
    tuple
        The minimum value of the function and the corresponding argument.
    """
    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        # Evaluate function at mid1 and mid2
        f_mid1 = function(mid1)
        f_mid2 = function(mid2)

        # Narrow the search range based on the evaluated values
        if f_mid1 < f_mid2:
            high = mid2
        else:
            low = mid1

    mid = (low + high) / 2
    return function(mid), mid