import os
import datetime
from glob import glob
from tqdm import tqdm


from micro_sam.util import _load_checkpoint


def main(path, name):
	checkpoint_paths = glob(os.path.join(path, "**", name), recursive=True)
	
	if not os.path.exists(checkpoint_paths[0]):
		raise FileNotFoundError(f"No checkpoitns found at '{path}'.")
	metrics = []
	metric_paths = []
	# Load the state and verify the time taken for getting the best model
	for checkpoint_path in checkpoint_paths:
     
		state, _ = _load_checkpoint(checkpoint_path)
		time_in_seconds = state["train_time"]
		best_metric = state["best_metric"]

		minutes, seconds = divmod(time_in_seconds, 60)
		hours, minutes = divmod(minutes, 60)
		print(checkpoint_path)
		#print("The time taken to achieve the best model -", "%d:%02d:%02d" % (hours, minutes, seconds))
		print(f"with best metric: {best_metric}")



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
        "--path",
        default="/scratch-grete/usr/nimlufre/medico_sam/mito_segmentation_lrs/"
    )
	parser.add_argument(
        "--name",
        default="best.pt"
    )
	args = parser.parse_args()
	main(args.path, args.name)