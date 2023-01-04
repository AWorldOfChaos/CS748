import argparse
from subprocess import run, PIPE
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-H', '--horizon', type = str, help = 'Enter horizons (comma-separated)')
ap.add_argument('-N', '--name', type = str, help = 'Enter plot name')
args = ap.parse_args()
name = args.name

H = args.horizon.strip().split(',')
R = []
for h in H:
    R.append(float(run(['python3', 'env.py', '-T', str(h)], stdout = PIPE).stdout.decode('utf-8')))
    print(f'Horizon: {h}, Regret: {R[-1]}')

print('Generating plot')
plt.plot(H,R)
plt.savefig(f'{name}.png')
plt.show()