"""
Training script for reach-avoid RL on 6D and 13D docking dynamics.

Uses DDQNSingle with DRABE (Discounted Reach-Avoid Bellman Equation)
from the copied safety_rl codebase.

Examples:
    6D:  python train_docking.py --dynamics 6d -wi 10000 -w -g 0.9999 -n 6d -sf
    13D: python train_docking.py --dynamics 13d -wi 10000 -w -g 0.9999 -n 13d -sf
    quick test: python train_docking.py --dynamics 6d -mu 2000 -mc 1000 -cp 500 -n smoke
"""

import os
import argparse
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingle import DDQNSingle
from RARL.config import dqnConfig
from RARL.utils import save_obj

matplotlib.use('Agg')
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser(
    description='Train DDQN reach-avoid agent on docking dynamics')

# dynamics selection
parser.add_argument(
    '--dynamics', choices=['6d', '13d'], default='6d',
    help='Which docking dynamics to use')

# environment
parser.add_argument(
    '-dt', '--doneType', default='TF', type=str,
    help='When to raise done flag: TF, fail, toEnd')
parser.add_argument(
    '-rnd', '--randomSeed', default=0, type=int, help='Random seed')
parser.add_argument(
    '--timestep', default=0.1, type=float, help='Integration dt (seconds)')

# training scheme
parser.add_argument(
    '-w', '--warmup', action='store_true', help='Warmup Q-network')
parser.add_argument(
    '-wi', '--warmupIter', default=5000, type=int, help='Warmup iterations')
parser.add_argument(
    '-mu', '--maxUpdates', default=400000, type=int,
    help='Max gradient updates')
parser.add_argument(
    '-ut', '--updateTimes', default=20, type=int,
    help='Number of hyper-parameter update periods')
parser.add_argument(
    '-mc', '--memoryCapacity', default=50000, type=int,
    help='Replay buffer size')
parser.add_argument(
    '-cp', '--checkPeriod', default=20000, type=int,
    help='Evaluation checkpoint period')
parser.add_argument(
    '-ms', '--maxSteps', default=300, type=int,
    help='Max steps per episode (300 = 30s at dt=0.1)')

# hyper-parameters
parser.add_argument(
    '-a', '--annealing', action='store_true', help='Enable gamma annealing')
parser.add_argument(
    '-arc', '--architecture', default=[256, 256], nargs='+', type=int,
    help='Hidden layer sizes')
parser.add_argument(
    '-lr', '--learningRate', default=1e-3, type=float, help='Learning rate')
parser.add_argument(
    '-g', '--gamma', default=0.9999, type=float,
    help='Discount / contraction coefficient')
parser.add_argument(
    '-act', '--actType', default='Tanh', type=str,
    help='Activation function: Tanh, ReLU, Sin')

# RL type
parser.add_argument('-m', '--mode', default='RA', type=str, help='RL mode')
parser.add_argument(
    '-tt', '--terminalType', default='g', type=str,
    help='Terminal value type for RA')

# output
parser.add_argument(
    '-st', '--showTime', action='store_true', help='Append timestamp to name')
parser.add_argument('-n', '--name', default='', type=str, help='Run name')
parser.add_argument(
    '-of', '--outFolder', default='experiments', type=str,
    help='Output directory root')
parser.add_argument(
    '-sf', '--storeFigure', action='store_true', help='Save figures')
parser.add_argument(
    '-pf', '--plotFigure', action='store_true', help='Display figures')

# evaluation
parser.add_argument(
    '--numRndTraj', default=200, type=int,
    help='Number of random trajectories for periodic evaluation')

# importance sampling
parser.add_argument(
    '--importance_sampling', action='store_true',
    help='Enable goal-centric importance sampling for episode ICs')
parser.add_argument(
    '--target_ratio', type=float, default=None,
    help='Fraction of targeted ICs (default: 0.1 for 6D, 0.05 for 13D)')

# IC range
parser.add_argument(
    '--tight_ic', action='store_true',
    help='Use tighter IC sampling range (closer to goal)')

args = parser.parse_args()
print(args)

# == CONFIGURATION ==
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = args.maxSteps

fn = f"docking-{args.dynamics}-DDQN"
run_name = args.name + '-' + args.doneType
if args.showTime:
    run_name = run_name + '-' + timestr
outFolder = os.path.join(args.outFolder, fn, run_name)
print(f"Output: {outFolder}")
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

# == ENVIRONMENT ==
print("\n== Environment ==")
if args.doneType == 'toEnd':
    sample_inside_obs = True
else:
    sample_inside_obs = False

if args.dynamics == '6d':
    from envs.docking6d_env import Docking6DEnv
    env = Docking6DEnv(
        device=device, mode=args.mode, doneType=args.doneType,
        sample_inside_obs=sample_inside_obs, dt=args.timestep)
elif args.dynamics == '13d':
    from envs.docking13d_env import Docking13DEnv
    env = Docking13DEnv(
        device=device, mode=args.mode, doneType=args.doneType,
        sample_inside_obs=sample_inside_obs, dt=args.timestep)
else:
    raise ValueError(f"Unknown dynamics: {args.dynamics}")

stateDim = env.state.shape[0]
actionNum = env.action_space.n
actionList = np.arange(actionNum)
print(f"State dim: {stateDim}, Action num: {actionNum}")
env.set_seed(args.randomSeed)

# == TIGHT IC SAMPLING ==
if args.tight_ic:
    if args.dynamics == '6d':
        tight_range = np.array([
            [-5.0, 5.0], [-5.0, 5.0],
            [-0.5, 0.5], [-0.5, 0.5],
            [-np.pi, np.pi], [-0.5, 0.5]])
    else:
        tight_range = {'pos': (-5.0, 5.0), 'vel': (-0.1, 0.1), 'omega': (-0.2, 0.2)}
    env.set_sampling_range(tight_range)
    print(f"Tight IC sampling enabled")

# == IMPORTANCE SAMPLING ==
if args.importance_sampling:
    from envs.importance_sampler import ImportanceSampler
    target_ratio = args.target_ratio
    if target_ratio is None:
        target_ratio = 0.1 if args.dynamics == '6d' else 0.05
    env.importance_sampler = ImportanceSampler(
        dynamics=env.dynamics,
        target_ratio=target_ratio,
        seed=args.randomSeed,
    )
    print(f"Importance sampling enabled: target_ratio={target_ratio}")

# == AGENT CONFIG ==
print("\n== Agent ==")
if args.annealing:
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=f"docking_{args.dynamics}", SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50,
)

dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingle(
    CONFIG, actionNum, actionList, dimList=dimList, mode=args.mode,
    terminalType=args.terminalType)
print(f"Device: {device}, Q-network on CUDA: "
      f"{next(agent.Q_network.parameters()).is_cuda}")
print(f"Architecture: {dimList}")

# == WARMUP ==
vmin, vmax = -1, 1
if args.warmup:
    print("\n== Warmup Q ==")
    lossList = agent.initQ(
        env, args.warmupIter, outFolder, num_warmup_samples=200,
        vmin=vmin, vmax=vmax, plotFigure=args.plotFigure,
        storeFigure=args.storeFigure)

    if args.storeFigure:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        start = min(500, args.warmupIter - 1)
        ax.plot(np.arange(start, args.warmupIter), lossList[start:], 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig(os.path.join(figureFolder, 'initQ_Loss.png'))
        plt.close(fig)

# == TRAINING ==
print("\n== Training ==")
trainRecords, trainProgress = agent.learn(
    env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
    doneTerminate=True, vmin=vmin, vmax=vmax, showBool=False,
    checkPeriod=args.checkPeriod, outFolder=outFolder,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure,
    numRndTraj=args.numRndTraj)

# == SAVE ==
trainDict = {
    'trainRecords': trainRecords,
    'trainProgress': trainProgress,
    'args': vars(args),
    'dimList': dimList,
}
if hasattr(env, 'importance_sampler') and env.importance_sampler is not None:
    stats = env.importance_sampler.stats
    print(f"IC sampling stats: {stats}")
    trainDict['importance_sampling_stats'] = stats
save_obj(trainDict, os.path.join(outFolder, 'train'))

# == POST-TRAINING PLOTS ==
if args.storeFigure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(trainRecords, 'b:', alpha=0.5)
    ax.set_xlabel('Gradient update')
    ax.set_title('Loss')
    ax.set_xlim(0, maxUpdates)

    ax = axes[1]
    data = trainProgress[:, 0]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Checkpoint')
    ax.set_title('Success Rate')
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(figureFolder, 'train_loss_success.png'))
    plt.close(fig)

print(f"\nTraining complete. Results saved to {outFolder}")
if trainProgress.shape[0] > 0:
    best_idx = np.argmax(trainProgress[:, 0])
    print(f"Best success rate: {trainProgress[best_idx, 0]:.3f} "
          f"at checkpoint {(best_idx + 1) * args.checkPeriod}")
