import argparse
from subprocess import run, PIPE
from matplotlib import pyplot as plt

seeds = [13, 69, 420, 666, 42, 0, 2012, 7, 747, 8]
ap = argparse.ArgumentParser()
ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)', required = True)
ap.add_argument('-N', '--name', type = str, help = 'Enter plot name', required = True)
args = ap.parse_args()
name = args.name

H = args.horizon.strip().split(',')
R = []
for h in H:
    print(f'Horizon: {h}')
    r = 0.0
    for s in seeds:
        print(f'Running for seed: {s}')
        r += float(run(['python3', 'env.py', '-T', str(h), '-S', str(s)], stdout = PIPE).stdout.decode('utf-8'))
    R.append(r / len(seeds))
    print(f'Regret: {R[-1]}')

print('Generating plot')
plt.plot(H,R)
plt.savefig(f'{name}.png')
plt.show()