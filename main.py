from numba import jit
import random
import time

@jit(nopython=True)
def monte_carlo_pi_fast(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

def monte_carlo_pi_slow(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return (4.0 * acc / nsamples)

def measure_ms(f, n, params = []):
    """ Measure the average time of completion for function f across n runs """
    time_start = time.time_ns()
    for i in range(n):
        f(*params)
        
    return float((time.time_ns() - time_start) / n) / 1000000.0

# measure
print(measure_ms(monte_carlo_pi_fast, 10000000, [5]))
print(measure_ms(monte_carlo_pi_slow, 10000000, [5]))
