import os
from glob import glob

import pandas as pd


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/3d"


def _get_results_per_dataset():
    for bpath in glob(os.path.join(ROOT, "*")):
        backbone = os.path.basename(bpath)
        for mpath in glob(os.path.join(bpath, "*")):
            model_name = os.path.basename(mpath)
            for dpath in glob(os.path.join(mpath, "*")):
                dataset = os.path.basename(dpath)
                for res_path in glob(os.path.join(dpath, "results", "**", "*.csv")):
                    res = pd.read_csv(res_path)

                    breakpoint()


def main():
    _get_results_per_dataset()


if __name__ == "__main__":
    main()
