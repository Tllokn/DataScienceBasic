from __future__ import annotations

import numpy as np


def compute_return(portfolio_values: np.ndarray) -> float:
    """Computes total return given portfolio values over time.

    Parameters
    ----------
    portfolio_values : np.ndarray[float]
        Time series of portfolio values.

    Returns
    -------
    return_percentage : float
        Total return, in percentage points.

    Example
    -------
    >>> values = np.array([50, 48, 51, 55, 57, 60, 58, 54, 56, 57])
    14.0
    """
    return 100 * (portfolio_values[-1] - portfolio_values[0]) / (1.0 * portfolio_values[0])


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Computes maximum drawdown given portfolio values over time.

    Parameters
    ----------
    portfolio_values : np.ndarray[float]
        Time series of portfolio values.

    Returns
    -------
    max_drawdown : float
        Maximum drawdown, in percentage points.

    Example
    -------
    >>> values = np.array([50, 48, 51, 55, 57, 60, 58, 54, 56, 57])
    10.0
    """
    return 100 * ((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(
        portfolio_values)).max()


def compute_ror(multi_values: np.ndarray):
    '''

    Parameters
    ----------
    multi_values: assets values we used to compute mean value and the return_over_risk

    Returns
    -------
    return_over_risk on the mean of multi_values

    '''
    mean_value = np.array(multi_values).mean(axis=0)
    gain = compute_return(mean_value)
    loss = compute_max_drawdown(mean_value)
    return gain / loss


def optimize_portfolio(asset_values: np.ndarray) -> tuple[list[int], float]:
    """Uses local search to choose a subset of the assets which maximizes return/risk.

    Parameters
    ----------
    asset_values : np.ndarray[float]
        2D array where row i contains a time series of values for the ith asset.

    Returns
    -------
    portfolio_assets : list[int]
        Sorted list of indices corresponding to the assets included in the portfolio.
    return_over_risk : float
        Return divided by maximum drawdown (objective function value) for the chosen assets.

    Example
    -------
    >>> values = [[100, 99, 98, 101, 102], [100, 95, 88, 96, 103], [100, 103, 107, 106, 104]]
    >>> optimize_portfolio(np.array(values))
    ([0, 2], 6.21)
    """
    curr_ror = 0

    # greedy algorithm to find start:
    for index,values in enumerate(asset_values):
        temp_ror=compute_ror([values])
        if temp_ror>curr_ror:
            curr_ror=temp_ror
            curr_index=[index]
            curr_values=[values]

    start=curr_index[0]


    for index, asset in enumerate(asset_values):
        '''
        temp_ror used to store return_over_risk of the situation that we take current asset as part of 
        our choice.
        If should have no influence on the curr_ror.
        
        When we compute temp_ror out and make sure it larger than curr_ror, we update, 
        otherwise, we give up this choice.
        '''
        if index!=start:
            temp_values = curr_values.copy()
            temp_values.append(asset)
            temp_ror = compute_ror(temp_values)

            if temp_ror > curr_ror:
                curr_ror = temp_ror
                curr_index.append(index)
                curr_values.append(asset)

    return (sorted(curr_index), curr_ror)


def optimize_portfolio_with_restarts(
        asset_values: np.ndarray, num_restarts: int = 10
) -> tuple[list[int], float]:
    """Uses local search with restarts to choose a subset of the assets which maximizes return/risk.

    Parameters
    ----------
    asset_values : np.ndarray[float]
        2D array where row i contains a time series of values for the ith asset.
    num_restarts : int
        Number of times to restart the local search.

    Returns
    -------
    portfolio_assets : list[int]
        Sorted list of indices corresponding to the assets included in the portfolio.
    return_over_risk : float
        Return divided by maximum drawdown (objective function value) for the chosen assets.
    """
    high = asset_values.shape[0]
    start_points = np.random.randint(low=0, high=high, size=num_restarts)

    print(start_points)

    best_ror=0
    best_index=[]

    for sp in start_points:
        curr_values = [asset_values[sp]]
        curr_ror = compute_ror(curr_values)
        curr_index = [sp]

        for index, asset in enumerate(asset_values):
            '''
            temp_ror used to store return_over_risk of the situation that we take current asset as part of 
            our choice.
            If should have no influence on the curr_ror.

            When we compute temp_ror out and make sure it larger than curr_ror, we update, 
            otherwise, we give up this choice.
            '''
            if index != sp:
                temp_values = curr_values.copy()
                temp_values.append(asset)
                temp_ror = compute_ror(temp_values)

                if temp_ror > curr_ror:
                    curr_ror = temp_ror
                    curr_index.append(index)
                    curr_values.append(asset)

        if curr_ror>best_ror:
            best_ror=curr_ror
            best_index=sorted(curr_index)

    return (best_index, best_ror)


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    # values = [[100, 99, 98, 101, 102], [100, 95, 88, 96, 103], [100, 103, 107, 106, 104]]
    values = [[100, 65, 70, 50, 110], [100, 100, 60, 60, 110]]
    print(optimize_portfolio(np.array(values)))
    # print(optimize_portfolio_with_restarts(np.array(values)))

    # print(compute_ror([values[2]]))

