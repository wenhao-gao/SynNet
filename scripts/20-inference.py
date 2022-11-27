import json
import logging
from functools import partial
from pathlib import Path
from time import time
from typing import Union

import pandas as pd
from rdkit import RDLogger

from synnet.config import MAX_PROCESSES
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.data_generation.syntrees import Encoder, MorganFingerprintEncoder
from synnet.decoding.decoder import HelperDataloader, SynTreeDecoder, SynTreeDecoderGreedy
from synnet.encoding.distances import cosine_distance, tanimoto_similarity
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet
from synnet.utils.parallel import chunked_parallel

logger = logging.getLogger(__file__)


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
    parser.add_argument("--output-dir", type=str, help="Directory to save output.")
    # Parameters
    parser.add_argument("--num", type=int, default=-1, help="Number of molecules to predict.")
    parser.add_argument(
        "--data",
        type=str,
        help="File with molecules to decode.",
    )
    # Processing
    parser.add_argument("--ncpu", type=int, default=MAX_PROCESSES, help="Number of cpus")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


def wrapper(
    target: str,
    *,
    syntree_decoder: Union[SynTreeDecoder, SynTreeDecoderGreedy],
    mol_encoder: Encoder,
    **kwargs,
) -> dict[str, Union[float, SyntheticTree]]:
    """Wrapper function to decode targets into `SyntheticTree` & catch Exceptions.

    Info:
        Always return a Dict *with* a `SyntheticTree`.
        This allows easily keep everything in order when saving to a file via `SyntheticTreeSet` and
        allows to call its method `is_valid()` on the entire list."""
    # Encode target
    try:
        z_target = mol_encoder.encode(target)
    except Exception as e:
        logger.error(f"Failed to encode {target}", exc_info=1)
        return {"syntree": SyntheticTree()}

    # Decode target
    try:
        res = syntree_decoder.decode(z_target, **kwargs)

    except Exception as e:
        logger.error(f"Failed to encode {target}", exc_info=1)
        return {"syntree": SyntheticTree()}
    return res


def print_stats(df) -> None:
    n_valid = df["is_valid"].sum()
    n_recovered = (df["max_similarity"] == 1.0).sum()
    recovery_rate = n_recovered / n_valid
    avg_similarity = df["max_similarity"][df["is_valid"]].mean()

    # logger.info(f"For {args.data}:")
    logger.info(f"For {args.data}:")
    logger.info(f"  Total number of targets: {len(df)}")
    logger.info(f"  Total number of valid reconstructions: {n_valid}")
    logger.info(f"  Total number of successful reconstructions: {n_recovered}")
    logger.info(f"  {recovery_rate=}  ({recovery_rate :.4%})")
    logger.info(f"  {avg_similarity=} ({avg_similarity:.4%})")
    return None


def postprocess_results(results: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(results)
    df["is_valid"] = df["syntree"].apply(lambda x: x.is_valid)

    df["decoded_smiles"] = df["syntree"].apply(lambda st: st.root.smiles if st.is_valid else None)
    df["decoded_depth"] = df["syntree"].apply(lambda st: st.depth if st.is_valid else None)

    return df


def save_results(output_dir: str, df: pd.DataFrame):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save df
    syntrees = df.pop("syntree").to_list()
    df.to_csv(output_dir / "results.csv", index=False)

    # Save generated trees
    syntree_collection = SyntheticTreeSet(sts=syntrees)
    syntree_collection.save(output_dir / "decoded_syntrees.json.gz")
    logger.info(f"Saved results to {output_dir}.")


def _setup_loggers():
    if args.verbose:
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        logger.addHandler(
            logging.FileHandler(
                filename=Path(args.output_dir) / ".log",
                mode="w",
            )
        )

    if args.debug:
        l = logging.getLogger("synnet.decoding.decoder")
        l.setLevel("DEBUG")
    else:
        RDLogger.DisableLog("rdApp.*")


if __name__ == "__main__":
    logger.info("Start.")
    t0 = time()

    # Parse input args
    args = get_args()

    _setup_loggers()

    logger.info(f"Arguments: {json.dumps(args.__dict__,indent=2)}")

    # region-dataloading
    # Load molecules to decode
    targets_all = HelperDataloader().fetch_data(args.data)
    if args.num > 0:  # Select only n queries
        targets = targets_all[: args.num]
    else:
        targets = targets_all
    logger.info(f"Number of targets, i.e. mols to decode, : {len(targets)}")

    # Load assets
    bblocks = BuildingBlockFileHandler().load(args.building_blocks_file)
    reaction_collection = ReactionSet().load(args.rxns_collection_file)

    # Load and init building blocks embedder (kdtree)
    bblocks_molembedder = (
        MolEmbedder().load_precomputed(args.embeddings_knn_file).init_balltree(cosine_distance)
    )

    # Load models
    logger.info("Start loading models from checkpoints...")
    ckpt_dir = Path(args.ckpt_dir)

    ckpt_files = [find_best_model_ckpt(ckpt_dir / model) for model in "act rt1 rxn rt2".split()]
    act_net, rt1_net, rxn_net, rt2_net = [load_mlp_from_ckpt(file) for file in ckpt_files]
    logger.info("...loading models completed.")
    # endregion-dataloading

    # Simple Encoder
    stdecoder = SynTreeDecoder(
        building_blocks=bblocks,
        reaction_collection=reaction_collection,
        action_net=act_net,
        reactant1_net=rt1_net,
        rxn_net=rxn_net,
        reactant2_net=rt2_net,
        building_blocks_embeddings=bblocks_molembedder.get_embeddings(),
        balltree=bblocks_molembedder.kdtree,
        similarity_fct=tanimoto_similarity,
    )
    # Greedy decoder
    stdecoder_greedy = SynTreeDecoderGreedy(decoder=stdecoder)

    # Decode targets
    _wrapper = partial(
        wrapper,
        syntree_decoder=stdecoder_greedy,
        mol_encoder=MorganFingerprintEncoder(2, 4096),
    )

    logger.info(f"Start decoding {len(targets)} targets.")

    results = chunked_parallel(targets, _wrapper, max_cpu=args.ncpu, verbose=args.verbose)

    logger.info("Completed decoding.")
    logger.info(f"Elapsed: {time()-t0:.0f}s")

    # Convert results to df
    df = postprocess_results(results)

    # Print some stats
    print_stats(df)

    # Save results
    save_results(args.output_dir, df)
    logger.info("Completed.")
