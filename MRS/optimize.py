"""
Functions for optimization and fitting
"""

def err_func(params, x, y, func, w=None):
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
	
        Returns
        -------
        The marginals of the fit to x/y given the params.
	
        """

	err = y - func(x, *params)
	
	if w is not None:
	    err = err * w

	return err
