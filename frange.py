#! /usr/bin/python
# a range function with float increments (vs normal range which increments by 1)
# http://code.activestate.com/recipes/66472-frange-a-range-function-with-float-increments/

import math
def frange(limit1, limit2 = None, increment = 1.):
  """
  Range function that accepts floats (and integers).

  Usage:
  frange(-2, 2, 0.1)
  frange(10)
  frange(10, increment = 0.5)

  The returned value is an iterator.  Use list(frange) for a list.
  """

  if limit2 is None:
    limit2, limit1 = limit1, 0.
  else:
    limit1 = float(limit1)

  count = int((limit2 - limit1)/increment)
  return (limit1 + n*increment for n in range(count))
