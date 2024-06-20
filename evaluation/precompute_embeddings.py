import os

from micro_sam.util import get_sam_model
from micro_sam.evaluation import precompute_all_embeddings

from util import get_dataset_paths, get_default_arguments


def main():
    args = get_default_arguments()

    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint)
    embedding_dir = os.path.join(args.experiment_folder, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    # getting the embeddings for the test set
    image_paths, _, _ = get_dataset_paths(args.dataset, "test")
    precompute_all_embeddings(predictor, image_paths, embedding_dir)


if __name__ == "__main__":
    main()
