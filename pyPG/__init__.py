import _pyPG as __pyPG
import numpy as _N
import numbers as _nums

## Draw PG(h, z)
##------------------------------------------------------------------------------
#  rpg.gamma <- function(num=1, h=1, z=0.0, trunc=200)

## Draw PG(n, z) where n is a natural number.
##------------------------------------------------------------------------------
#  rpg.devroye <- function(num=1, n=1, z=0.0)

## Draw PG(h, z) where h is >= 1.
##------------------------------------------------------------------------------
#  rpg.alt <- function(num=1, h=1, z=0.0)

## Draw PG(h, z) using SP approx where h is >= 1.
##------------------------------------------------------------------------------
#  rpg.sp <- function(num=1, h=1, z=0.0, track.iter=FALSE)

#  either n, z are both arrays of equal size
#  n, z are both scalaras, and size given by num

def rpg_devroye(n, z, num=1, out=None):
    if (out is not None):
        if (num > 1) and (num != len(out)):
            raise Exception("parameter num and length of output incompatible.")
        out = out.astype(_N.float64, copy=False)
        num = len(out)
    else:
        out = _N.empty(num, dtype=_N.float64)

    if isinstance(n, _N.ndarray):
        if (n.dtype != _N.int32):
            n = n.astype(_N.int32, copy=False)
    else:
        n = _N.ones(num, dtype=_N.int32) * int(n)
    if isinstance(z, _N.ndarray):
        if (z.dtype != _N.float64):
            z = z.astype(_N.float64, copy=False)
    else:
        z = _N.ones(num, dtype=_N.float64) * z

    nA, zA, outA = _N.broadcast_arrays(n, z, out)


    __pyPG.rpg_devroye(outA, nA, zA, num)

    if out.shape[0] > 1:
       return out
    return out[0]

def rpg_gamma(h, z, num=1, out=None, trunc=200):
    if isinstance(h, _nums.Number) and isinstance(z, _nums.Number):
        h = _N.ones(num, dtype=_N.int32) * h
        z = _N.ones(num) * z
    elif isinstance(h, _N.ndarray) and isinstance(z, _N.ndarray) and (len(h) == len(z)):
        num = len(h)
    else:
        raise Exception("Problems with input.  Either both arrays of same length, or both scalaras")
    if out == None:
        out = _N.empty(num)
    __lw.par_rpg_gamma(out, h, z, num, trunc)
    if num == 1:
        return out[0]
    return out

"""
def rpg_alt(h, z, num=1, trunc=200):
    arr_h = (type(h) == _N.ndarray) or (type(h) == list) or (type(h) == tuple)
    arr_z = (type(z) == _N.ndarray) or (type(z) == list) or (type(z) == tuple)

    if (num > 1) or arr_h or arr_z:
        retArr = _N.empty(num, dtype=_N.float)
        if not arr_h:
            int_h = (type(h) == _N.int16) or (type(h) == _N.int32) or (type(h) == _N.int64) or (type(h) == int)
            if int_h:
                h = _N.ones(num, dtype=_N.int32) * h
            else:
                #  throw an exception
                pass
        if not arr_z:
            z = _N.ones(num, dtype=_N.float) * z
        __lw.rpg_alt(retArr, h, z, num)
        return retArr
    else:  #  both n, z not arrays
        int_h = (type(h) == _N.int16) or (type(h) == _N.int32) or (type(h) == _N.int64) or (type(h) == int)
        if not int_h:
            # throw an exception
            pass
        elif h < 0:
            # throw an exception
            pass
        if z < 0:
            # throw an exception
            pass
        z = 1.0 * z  #  force to float
        x = __lw.rpg_alt(h, z)
        return x
"""
