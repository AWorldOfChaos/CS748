import argparse
from subprocess import run, PIPE
from matplotlib import pyplot as plt

seeds = [13, 69, 420, 666, 42, 0, 2012, 7, 747, 8]
ap = argparse.ArgumentParser()
ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
ap.add_argument('-A', '--algo', type = str, help = 'Enter algorithm', required = False, default = 'ucb')
args = ap.parse_args()

H = args.horizon.strip().split(',')
R = []
for h in H:
    print(f'Horizon: {h}')
    r = 0.0
    for s in seeds:
        print(f'Running for seed: {s}')
        r += float(run(['python3', 'sim.py', '-T', str(h), '-S', str(s), '-A', args.algo], stdout = PIPE, check = False).stdout.decode('utf-8'))
    R.append(r / len(seeds))
    print(f'Regret: {R[-1]}')

print('Generating plot')
plt.plot(H,R)
plt.savefig(f'figures/{args.algo}.png')
plt.show()