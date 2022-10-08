import numba as nb
#from tensorboardX import SummaryWriter

from ppo import Agent


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new

# if __name__ == '__main__':
ag = Agent()
ag.run()