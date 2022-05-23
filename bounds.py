import numpy as np
import matplotlib.pyplot as plt


def constant(const):
    def func(_):
        return const

    return func


def linear(intercept, slope):
    def func(x):
        return slope * x + intercept
    return func


def exponential(intercept, limit, lam):
    def increasing(x):
        return (limit - intercept) * (1 - np.exp(-lam * x)) + intercept

    def decreasing(x):
        return (intercept - limit) * np.exp(-lam * x) + limit

    if intercept > limit:
        # print(f'decreasing: {intercept, limit}')
        return decreasing
    elif intercept < limit:
        return increasing
    else:
        return constant(intercept)


def linear_exponential(lower, higher, length):
    def func(x):
        return np.exp((higher - lower) * x / length + lower)
    return func


def reciprocal_exponential(intercept, limit, a, n):
    '''
    r(x)=r * exp(+\- a / (x+b)^n)
    where b adjusts the initial value
    '''
    def increasing(x):
        b = np.power(a / (np.log(limit) - np.log(intercept)), 1 / n)
        return limit * np.exp(-a / np.power(x + b, n))

    def decreasing(x):
        b = np.power(a / (np.log(intercept) - np.log(limit)), 1 / n)
        return limit * np.exp(a / np.power(x + b, n))

    if intercept > limit:
        return decreasing
    elif intercept < limit:
        return increasing
    else:
        return constant(intercept)


def function_tuple(funcs):
    def func(x):
        return tuple(fun(x) for fun in funcs)
    return func


funcs = {constant, linear, exponential,
         linear_exponential, reciprocal_exponential}
func_name = {func.__name__: func for func in funcs}


class Bounds:
    '''
    Handles the varying bounds of the game environment. Part of the rewards structure.

    The minimum and maximum bounds are functions of the number of game steps and number of 
    epochs. During each epoch, the bounds are varied based on a selected function.
    At the end of each epoch, the bounds are reset. 
    The parameters of the functions are constant within each epoch, and during a reset, 
    the parameters are altered based on how many resets have happened.
    '''

    def __init__(self, rmin_func, rmin_strategy, rmax_func, rmax_strategy) -> None:
        '''
        Initialize the bounds

        Parameters:
            rmin_func: the function to determine the lower bound in the radius
            rmin_strategy: configurations of each minimum radius function parameter during a reset
                example: for rmin_func='exponential',
                    rmin_strategy = [
                        {
                            'name': 'constant',
                            'parameters': { 'const': 0.1 }
                        },
                        {
                            'name': 'constant',
                            'parameters': {'const': 1 }
                        },
                        {
                            'name': 'constant',
                            'parameters': { 'const': np.exp(-4) }
                        }
                    ]
            rmax_func: the function to determine the upper bound in the radius
            rmax_strategy: configurations of each maximum radius function parameter during a reset
        '''
        self.rmin_func = self._get_func(rmin_func)
        self.rmin_strategy = self._unpack_strategy(rmin_strategy)
        self.rmax_func = self._get_func(rmax_func)
        self.rmax_strategy = self._unpack_strategy(rmax_strategy)

        self.resets = 0
        self.reset()

    def _get_func(self, name):
        return func_name[name]

    def _unpack_strategy(self, strategy):
        return function_tuple(tuple(self._get_func(st['name'])(**st['parameters']) for st in strategy))

    def reset(self):
        self.curr_min_func = self.rmin_func(*(self.rmin_strategy(self.resets)))
        self.curr_max_func = self.rmax_func(*(self.rmax_strategy(self.resets)))

        self.resets += 1

    def get_bounds(self, i):
        return self.curr_min_func(i), self.curr_max_func(i)


DEFAULT_BOUNDS = {
    'rmin_func': 'constant',
    'rmin_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 0.1}
        }
    ],
    'rmax_func': 'constant',
    'rmax_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 10}
        }
    ]
}


if __name__ == '__main__':
    config = {
        'rmin_func': 'exponential',
        'rmin_strategy': [
            {
                'name': 'constant',
                'parameters': {'const': 0.1}
            },
            {
                'name': 'constant',
                'parameters': {'const': 1}
            },
            {
                'name': 'constant',
                'parameters': {'const': np.exp(-4)}
            }
        ],
        'rmax_func': 'exponential',
        'rmax_strategy': [
            {
                'name': 'constant',
                'parameters': {'const': 10}
            },
            {
                'name': 'constant',
                'parameters': {'const': 1}
            },
            {
                'name': 'constant',
                'parameters': {'const': np.exp(-4)}
            }
        ]
    }
    bounds = Bounds(**config)
    t = np.arange(500)
    rmin = list()
    rmax = list()
    # for i in t:
    #     _rmin, _rmax = bounds.get_bounds(i)
    #     rmin.append(_rmin)
    #     rmax.append(_rmax)

    # plt.plot(t, rmin)
    # plt.plot(t, rmax)
    # plt.grid(True)
    # plt.show()

    bounds.reset()
    bounds.reset()
    bounds.reset()
    bounds.reset()
