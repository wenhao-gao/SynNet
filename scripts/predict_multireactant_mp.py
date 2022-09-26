"""
Generate synthetic trees for a set of specified query molecules. Multiprocessing.
"""
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd

from syn_net.config import (CHECKPOINTS_DIR, DATA_EMBEDDINGS_DIR, DATA_PREPARED_DIR,
                            DATA_PREPROCESS_DIR, DATA_RESULT_DIR)
from syn_net.data_generation.preprocessing import (BuildingBlockFileHandler,
                                                   ReactionTemplateFileHandler)
from syn_net.models.chkpt_loader import load_modules_from_checkpoint
from syn_net.utils.data_utils import SyntheticTree, SyntheticTreeSet
from syn_net.utils.predict_utils import mol_fp, synthetic_tree_decoder_beam_search

Path(DATA_RESULT_DIR).mkdir(exist_ok=True)
from syn_net.MolEmbedder import MolEmbedder


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
        file = Path(DATA_PREPARED_DIR) / f"synthetic-trees-{args.data}.json.gz"
        logger.info(f"Reading data from {file}")
        sts = SyntheticTreeSet()
        sts.load(file)
        smis_query = [st.root.smiles for st in sts.sts]
    elif args.data in ["chembl"]:
        smis_query = _fetch_data_chembl(name)
    else:  # Hopefully got a filename instead
        smis_query = _fetch_data_from_file(name)
    return smis_query

def find_best_model_ckpt(path: str) -> Union[Path, None]:  # TODO: move to utils.py
    """Find checkpoint with lowest val_loss.

    Poor man's regex:
    somepath/act/ckpts.epoch=70-val_loss=0.03.ckpt
                                     ^^^^--extract this as float
    """
    ckpts = Path(path).rglob("*.ckpt")
    best_model_ckpt = None
    lowest_loss = 10_000
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
    path_to_act, path_to_rt1, path_to_rxn, path_to_rt2 = path_to_checkpoints

    # Load the pre-trained models.
    act_net, rt1_net, rxn_net, rt2_net = load_modules_from_checkpoint(
        path_to_act=path_to_act,
        path_to_rt1=path_to_rt1,
        path_to_rxn=path_to_rxn,
        path_to_rt2=path_to_rt2,
        featurize=args.featurize,
        rxn_template=args.rxn_template,
        out_dim=out_dim,
        nbits=nbits,
        ncpu=args.ncpu,
    )
    return act_net, rt1_net, rxn_net, rt2_net


def func(smiles: str) -> Tuple[str,float,SyntheticTree]:
    """Generate a synthetic tree for the input molecular embedding."""
    emb = mol_fp(smiles)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_beam_search(
            z_target=emb,
            building_blocks=building_blocks,
            bb_dict=building_blocks_dict,
            reaction_templates=rxns,
            mol_embedder=mol_fp,
            action_net=act_net,
            reactant1_net=rt1_net,
            rxn_net=rxn_net,
            reactant2_net=rt2_net,
            bb_emb=bb_emb,
            rxn_template=args.rxn_template,
            n_bits=nbits,
            beam_width=3,
            max_step=15,
        )
    except Exception as e:
        logger.error(e,exc_info=e)
        action = -1

    if action != 3:  # aka tree has not been properly ended
        smi = None
        similarity = 0
        tree = None

    return smi, similarity, tree

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--featurize", type=str, default="fp", help="Choose from ['fp', 'gin']"
    )
    parser.add_argument("--radius", type=int, default=2, help="Radius for Morgan Fingerprint")
    parser.add_argument(
        "-b", "--nbits", type=int, default=4096, help="Number of Bits for Morgan Fingerprint"
    )
    parser.add_argument(
        "-r", "--rxn_template", type=str, default="hb", help="Choose from ['hb', 'pis']"
    )
    parser.add_argument("--ncpu", type=int, default=32, help="Number of cpus")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of molecules to predict.")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="test",
        help="Choose from ['train', 'valid', 'test', 'chembl'] or provide a file with one SMILES per line.",
    )
    parser.add_argument(
        "-o",
        "--outputembedding",
        type=str,
        default="fp_256",
        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logger.info(f"Args: {vars(args)}")

    nbits = args.nbits
    out_dim = args.outputembedding.split("_")[-1]  # <=> morgan fingerprint with 256 bits
    building_blocks_id = "enamine_us-2021-smiles"
    param_dir = f"{args.rxn_template}_{args.featurize}_{args.radius}_{nbits}_{out_dim}"

    # Load data ...
    logger.info("Stat loading data...")
    # ... query molecules (i.e. molecules to decode)
    smiles_queries = _fetch_data(args.data)
    if args.num > 0:  # Select only n queries
        smiles_queries = smiles_queries[:args.num]

    # ... building blocks
    file = Path(DATA_PREPROCESS_DIR) / f"{args.rxn_template}-{building_blocks_id}-matched.csv.gz"

    building_blocks = BuildingBlockFileHandler().load(file)
    building_blocks_dict = {
        block: i for i, block in enumerate(building_blocks)
    }  # dict is used as lookup table for 2nd reactant during inference

    # ... reaction templates
    file = Path(DATA_PREPROCESS_DIR) / f"reaction-sets_{args.rxn_template}_{building_blocks_id}.json.gz"
    rxns = ReactionTemplateFileHandler().load(file)

    # ... building block embedding
    file = Path(DATA_EMBEDDINGS_DIR) / f"{args.rxn_template}-{building_blocks_id}-embeddings.npy"
    bb_emb = MolEmbedder.load(file)
    logger.info("...loading data completed.")

    # ... models
    logger.info("Start loading models from checkpoints...")
    path = Path(CHECKPOINTS_DIR) / f"{param_dir}"
    paths = [
        find_best_model_ckpt("results/logs/hb_fp_2_4096/" + model)
        for model in "act rt1 rxn rt2".split()
    ]
    act_net, rt1_net, rxn_net, rt2_net = _load_pretrained_model(paths)
    logger.info("...loading models completed.")

    # Decode queries, i.e. the target molecules.
    logger.info(f"Start to decode {len(smiles_queries)} target molecules.")
    with mp.Pool(processes=args.ncpu) as pool:
        results = pool.map(func, smiles_queries)
    logger.info("Finished decoding.")

    # Print some results from the prediction
    smis_decoded = [r[0] for r in results]
    similarities = [r[1] for r in results]
    trees = [r[2] for r in results]

    recovery_rate = (np.asfarray(similarities) == 1.0).sum() / len(similarities)
    avg_similarity = np.mean(similarities)
    logger.info(f"For {args.data}:")
    logger.info(f"  {len(smiles_queries)=}")
    logger.info(f"  {recovery_rate=}")
    logger.info(f"  {avg_similarity=}")

    # Save to local dir
    output_dir = DATA_RESULT_DIR if args.output_dir is None else args.output_dir
    logger.info("Saving results to {output_dir} ...")
    df = pd.DataFrame(
        {"query SMILES": smiles_queries, "decode SMILES": smis_decoded, "similarity": similarities}
    )
    df.to_csv(f"{output_dir}/decode_result_{args.data}.csv.gz", compression="gzip", index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f"{output_dir}/decoded_st_{args.data}.json.gz")

    logger.info("Finish!")
