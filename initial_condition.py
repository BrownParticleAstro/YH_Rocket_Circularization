import numpy as np


def constant(value):
    def func(_):
        return value
    return func


def rotated_state(st, random, theta=0):
    def func(_):
        nonlocal st, theta
        
        if random:
            theta = np.random.uniform(0, 2 * np.pi)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        if isinstance(st, list):
            st = np.array(st)

        return [*(rot_mat @ st[:2]), *(rot_mat @ st[2:])]
    return func


funcs = {constant, rotated_state}
func_name = {func.__name__: func for func in funcs}

DEFAULT_INITIAL_CONDITION = {
    'function': 'constant',
    'parameters': {'value': [1, 0, 0, 1.1]}
}


class InitialCondition:
    '''
    Handles varying initial conditions in each episode
    '''

    def __init__(self, function, parameters) -> None:
        self.func = self._get_func(function)(**parameters)
        self.resets = 0

    def _get_func(self, name):
        return func_name[name]

    def get_initial_condition(self):
        return np.array(self.func(self.resets))

    def reset(self):
        self.resets += 1


if __name__ == '__main__':
    config = {
        'function': 'rotated_state',
        'parameters': {'st': [1, 0, 0, 1.1], 'random': True}
    }
    # config = {
    #     'function': 'constant',
    #     'parameters': { 'value': [1, 0, 0, 1.1] }
    # }
    ic = InitialCondition(**config)
    print(ic.get_initial_condition())
    ic.reset()
    print(ic.get_initial_condition())
    ic.reset()
    print(ic.get_initial_condition())
    ic.reset()
    print(ic.get_initial_condition())
