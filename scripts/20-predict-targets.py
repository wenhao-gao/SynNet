"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""  # TODO: Clean up + dont hardcode file paths
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from syn_net.config import DATA_PREPROCESS_DIR, DATA_RESULT_DIR, MAX_PROCESSES
from syn_net.data_generation.preprocessing import BuildingBlockFileHandler
from syn_net.encoding.distances import cosine_distance
from syn_net.models.mlp import load_mlp_from_ckpt
from syn_net.MolEmbedder import MolEmbedder
from syn_net.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet
from syn_net.utils.predict_utils import mol_fp, synthetic_tree_decoder_greedy_search

logger = logging.getLogger(__name__)


def _fetch_data_chembl(name: str) -> list[str]:
    raise NotImplementedError
    df = pd.read_csv(f"{DATA_DIR}/chembl_20k.csv")
    smis_query = df.smiles.to_list()
    return smis_query


def _fetch_data_from_file(name: str) -> list[str]:
    with open(name, "rt") as f:
        smis_query = [line.strip() for line in f]
    return smis_query


def _fetch_data(name: str) -> list[str]:
    if args.data in ["train", "valid", "test"]:
        file = (
            Path(DATA_PREPROCESS_DIR) / "syntrees" / f"synthetic-trees-filtered-{args.data}.json.gz"
        )
        logger.info(f"Reading data from {file}")
        syntree_collection = SyntheticTreeSet().load(file)
        smiles = [syntree.root.smiles for syntree in syntree_collection]
    elif args.data in ["chembl"]:
        smiles = _fetch_data_chembl(name)
    else:  # Hopefully got a filename instead
        smiles = _fetch_data_from_file(name)
    return smiles


def find_best_model_ckpt(path: str) -> Union[Path, None]:  # TODO: move to utils.py
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                         ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000 # ~ math.inf
    for file in ckpts:
        stem = file.stem
        val_loss = float(stem.split("val_loss=")[-1])
        if val_loss < lowest_loss:
            best_model_ckpt = file
            lowest_loss = val_loss
    return best_model_ckpt


def _load_pretrained_model(path_to_checkpoints: list[Path]):
    """Wrapper to load modules from checkpoint."""
    # Define paths to pretrained models.
    act_path, rt1_path, rxn_path, rt2_path = path_to_checkpoints

    # Load the pre-trained models.
    act_net = load_mlp_from_ckpt(act_path)
    rt1_net = load_mlp_from_ckpt(rt1_path)
    rxn_net = load_mlp_from_ckpt(rxn_path)
    rt2_net = load_mlp_from_ckpt(rt2_path)
    return act_net, rt1_net, rxn_net, rt2_net


def wrapper_decoder(smiles: str) -> Tuple[str, float, SyntheticTree]:
    """Generate a synthetic tree for the input molecular embedding."""
    emb = mol_fp(smiles)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_greedy_search(
            z_target=emb,
            building_blocks=bblocks,
            bb_dict=bblocks_dict,
            reaction_templates=rxns,
            mol_embedder=bblocks_molembedder.kdtree,  # TODO: fix this, currently misused
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template="hb",  # TODO: Do not hard code
            n_bits=4096,  # TODO: Do not hard code
            beam_width=3,
            max_step=15,
        )
    except Exception as e:
        logger.error(e, exc_info=e)
        action = -1

    if action != 3:  # aka tree has not been properly ended
        smi = None
        similarity = 0
        tree = None

    return smi, similarity, tree


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    # File I/O
    parser.add_argument(
        "--building-blocks-file",
        type=str,
        help="Input file with SMILES strings (First row `SMILES`, then one per line).",
    )
    parser.add_argument(
        "--rxns-collection-file",
        type=str,
        help="Input file for the collection of reactions matched with building-blocks.",
    )
    parser.add_argument(
        "--embeddings-knn-file",
        type=str,
        help="Input file for the pre-computed embeddings (*.npy).",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, help="Directory with checkpoints for {act,rt1,rxn,rt2}-model."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DATA_RESULT_DIR, help="Directory to save output."
    )
    # Parameters
    parser.add_argument("--num", type=int, default=-1, help="Number of molecules to predict.")
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    # Load data ...
    logger.info("Start loading data...")
    # ... query molecules (i.e. molecules to decode)
    targets = _fetch_data(args.data)
    if args.num > 0:  # Select only n queries
        targets = targets[: args.num]

    # ... building blocks
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    # A dict is used as lookup table for 2nd reactant during inference:
    bblocks_dict = {block: i for i, block in enumerate(bblocks)}
    logger.info(f"Successfully read {args.building_blocks_file}.")

    # ... reaction templates
    rxns = ReactionSet().load(args.rxns_collection_file).rxns
    logger.info(f"Successfully read {args.rxns_collection_file}.")

    # ... building block embedding
    bblocks_molembedder = (
        MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
    )
    bb_emb = bblocks_molembedder.get_embeddings()
    logger.info(f"Successfully read {args.embeddings_knn_file} and initialized BallTree.")
    logger.info("...loading data completed.")

    # ... models
    logger.info("Start loading models from checkpoints...")
    path = Path(args.ckpt_dir)
    paths = [find_best_model_ckpt(path / model) for model in "act rt1 rxn rt2".split()]
    act_net, rt1_net, rxn_net, rt2_net = _load_pretrained_model(paths)
    logger.info("...loading models completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(targets)} target molecules.")
    if args.ncpu == 1:
        results = [wrapper_decoder(smi) for smi in targets]
    else:
        with mp.Pool(processes=args.ncpu) as pool:
            logger.info(f"Starting MP with ncpu={args.ncpu}")
            results = pool.map(wrapper_decoder, targets)
    logger.info("Finished decoding.")

    # Print some results from the prediction
    # Note: If a syntree cannot be decoded within `max_depth` steps (15),
    #       we will count it as unsuccessful. The similarity will be 0.
    decoded = [smi for smi, _, _ in results ]
    similarities = [sim for _, sim, _ in results ]
    trees = [tree for _, _, tree in results ]

    recovery_rate = (np.asfarray(similarities) == 1.0).sum() / len(similarities)
    avg_similarity = np.mean(similarities)
    n_successful = sum([syntree is not None for syntree in trees])
    logger.info(f"For {args.data}:")
    logger.info(f"  Total number of attempted  reconstructions: {len(targets)}")
    logger.info(f"  Total number of successful reconstructions: {n_successful}")
    logger.info(f"  {recovery_rate=}")
    logger.info(f"  {avg_similarity=}")

    # Save to local dir
    # 1. Dataframe with targets, decoded, smilarities
    # 2. Synthetic trees of the decoded SMILES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir} ...")

    df = pd.DataFrame({"targets": targets, "decoded": decoded, "similarity": similarities})
    df.to_csv(f"{output_dir}/decoded_results.csv.gz", compression="gzip", index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f"{output_dir}/decoded_syntrees.json.gz")

    logger.info("Completed.")
