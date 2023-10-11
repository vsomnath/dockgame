import argparse

from dockgame.data.parser import ProteinParser
from dockgame.data.featurize import construct_featurizer
from dockgame.data import BaseDataset


def parse_preprocess_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="db5", help="Dataset to process")
    parser.add_argument("--esm_embeddings_path", default=None, type=str)
    parser.add_argument("--complex_dir", type=str, default="complexes")
    parser.add_argument("--complex_list_file", type=str, default="complexes.txt")

    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--progress_every", default=500, type=int)
    parser.add_argument("--max_complexes", default=None, type=int)
    parser.add_argument("--center_complex", action='store_true')
    parser.add_argument("--size_sorted", action='store_true')

    parser.add_argument("--agent_type", default="protein", type=str, choices=["chain", "protein"])
    parser.add_argument("--resolution", default="c_alpha", choices=["all", "bb", "c_alpha"])
    parser.add_argument("--featurizer", default=None, type=str, choices=["base", "pifold", None])

    args = parser.parse_args()
    return args


def main():
    args = parse_preprocess_args()

    featurizer = construct_featurizer(args=args)
    parser = ProteinParser()
    
    dataset = BaseDataset(
        root=args.data_dir, transform=None, parser=parser,
        featurizer=featurizer, complex_list_file=args.complex_list_file,
        dataset=args.dataset, complex_dir=args.complex_dir,
        mode=None, resolution=args.resolution, agent_type=args.agent_type,
        num_workers=args.num_workers, 
        esm_embeddings_path=args.esm_embeddings_path,
        center_complex=args.center_complex,
        size_sorted=args.size_sorted
    )

    dataset.preprocess_complexes()


if __name__ == "__main__":
    main()
