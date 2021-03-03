import numpy as np

from numba.typed import List

from numba import njit

##     ## ######## ##       ########  ######## ########   ######
##     ## ##       ##       ##     ## ##       ##     ## ##    ##
##     ## ##       ##       ##     ## ##       ##     ## ##
######### ######   ##       ########  ######   ########   ######
##     ## ##       ##       ##        ##       ##   ##         ##
##     ## ##       ##       ##        ##       ##    ##  ##    ##
##     ## ######## ######## ##        ######## ##     ##  ######



@njit
def set_numba_random_seed(seed) :
    np.random.seed(seed)



@njit
def single_random_choice(x) :
    return np.random.choice(x, size=1)[0]


@njit
def set_to_array(input_set) :
    out = List()
    for s in input_set :
        out.append(s)
    return np.asarray(out)


@njit
def nb_random_choice(arr, prob, size=1, replace=False, verbose=False) :
    """
    :param arr : A 1D numpy array of values to sample from.
    :param prob : A 1D numpy array of probabilities for the given samples.
    :param size : Integer describing the size of the output.
    :return : A random sample from the given array with a given probability.
    """

    assert len(arr) == len(prob)
    assert size < len(arr)

    prob = prob / np.sum(prob)
    if replace :
        ra = np.random.random(size=size)
        idx = np.searchsorted(np.cumsum(prob), ra, side="right")
        return arr[idx]
    else :
        if size / len(arr) > 0.5 and verbose :
            print("Warning : choosing more than 50% of the input array with replacement, can be slow.")

        out = set()
        while len(out) < size :
            ra = np.random.random()
            idx = np.searchsorted(np.cumsum(prob), ra, side="right")
            x = arr[idx]
            if not x in out :
                out.add(x)
        return set_to_array(out)


@njit
def exp_func(x, a, b, c) :
    return a * np.exp(b * x) + c