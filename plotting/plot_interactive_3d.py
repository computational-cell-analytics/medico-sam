import os
from glob import glob
from pathlib import Path

import pandas as pd


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/3d"


def _get_sam2_results_per_dataset(dataset_name):
    res_list = []
    for bpath in glob(os.path.join(ROOT, "*")):
        backbone = os.path.basename(bpath)
        for res_path in glob(os.path.join(bpath, "hvit_b", dataset_name, "results", "**", "*.csv")):
            res = pd.read_csv(res_path)
            roi = res.columns[-1]
            res_dict = {
                "backbone": backbone,
                "prompt_choice": Path(res_path).stem.split("_")[-1],
                "type": roi,
                "score": res.iloc[0][roi]
            }
            res_list.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(res_list, ignore_index=True)
    print(res_df)

    breakpoint()


def main():
    _get_sam2_results_per_dataset("lgg_mri")


if __name__ == "__main__":
    main()
