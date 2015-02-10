"""
Functions for optimization and fitting
"""

def err_func(params, x, y, func, w=None, func_list=None):
    """
    Error function for fitting a function
    
    Parameters
    ----------
    params : tuple
        A tuple with the parameters of `func` according to their order of
        input

    x : float array 
        An independent variable. 
        
    y : float array
        The dependent variable.

    func : function
        A function with inputs: `(x, *params)`

    w : ndarray
        A weighting function. Allows emphasizing certain parts of the
        original function. Should have the same length as x/y.

    func_list : dict
        dict of callables, each with it's own set of indices into the params

    Returns
        -------
        The marginals of the fit to x/y given the params.
	
    """
    err = y - func(x, *params)
    if w is not None:
        err = err * w
    if func_list is not None:
        err2 = 0
        for f in func_list: 
            this_err = (y - f[0](x, *[params[ii] for ii in f[1]]))
            if f[2] is not None:
                this_err = this_err * f[2]
            err2 = err2 + this_err 
        err = err2 + err

    return err
