"""Generate reproducible BasicDFN/Chen2020 reference fixtures.

Run: .venv/bin/python tutorials/battery/generate_dfn_reference.py --mesh 5
PyBaMM is only a development dependency.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tutorials.battery.dfn_common import reference


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--mesh', type=int, default=5)
    cli.add_argument('--output', type=Path, default=Path('tests/reference'))
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {
        'discharge': (5., None, np.arange(0., 4001., 50.)),
        'rest': (0., None, np.arange(0., 301., 50.)),
        'charge': (-2.5, {
            'initial_n': 15522.211404441303,
            'initial_p': 35269.45346021563
        }, np.arange(0., 1201., 50.))
    }
    for name, (current, initial, times) in cases.items():
        data = reference(args.mesh, times, current, initial)
        stem = args.output / f'chen2020_{name}_{args.mesh}'
        np.savez_compressed(stem.with_suffix('.npz'),
                            times=data.pop('times'),
                            differential=data.pop('differential'),
                            voltage=data.pop('voltage'))
        data.update(current=current, initial=initial)
        stem.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
        print(stem)


if __name__ == '__main__': main()
