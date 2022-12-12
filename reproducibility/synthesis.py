import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from synnet.encoding.distances import cosine_distance
from synnet.MolEmbedder import MolEmbedder
from synnet.utils.data_utils import ReactionSet, SyntheticTree, SyntheticTreeSet
from synnet.utils.predict_utils import mol_fp, synthetic_tree_decoder_greedy_search


def wrapper_decoder(smiles: str, **kwargs) -> Tuple[str, float, SyntheticTree]:
    """Generate a synthetic tree for the input molecular embedding."""
    smi = None
    similarity = 0
    tree = None

    emb = mol_fp(smiles)
    try:
        smi, similarity, tree, action = synthetic_tree_decoder_greedy_search(
            z_target=emb,
            **kwargs
        )
    except Exception as e:
        print(e, file=sys.stderr)
        action = -1

    if action != 3:  # aka tree has not been properly ended
        smi = None
        similarity = 0
        tree = None

    print(".", end='')
    return smi, similarity, tree


def synthesis(targets: list[str],
              bblocks: list[str],
              checkpoints: list,
              rxn_collection: ReactionSet,
              mol_embedder: MolEmbedder,
              output_dir: Path,
              cpu_cores: int):
    print("Start.")

    # A dict is used as lookup table for 2nd reactant during inference:
    bblocks_dict = {block: i for i, block in enumerate(bblocks)}

    # building block embedding
    bblocks_mol_embedder = mol_embedder.init_balltree(cosine_distance)
    bb_emb = bblocks_mol_embedder.get_embeddings()

    # checkpoints
    act_net, rt1_net, rxn_net, rt2_net = checkpoints

    # Wrapper func
    wrapper_func = partial(wrapper_decoder,
                           building_blocks=bblocks,
                           bb_dict=bblocks_dict,
                           reaction_templates=rxn_collection.rxns,
                           mol_embedder=bblocks_mol_embedder.kdtree,
                           action_net=act_net,
                           reactant1_net=rt1_net,
                           rxn_net=rxn_net,
                           reactant2_net=rt2_net,
                           bb_emb=bb_emb,
                           rxn_template="hb",
                           n_bits=4096,
                           beam_width=3,
                           max_step=15)

    # Decode queries, i.e. the target molecules.
    print(f"Start to decode {len(targets)} target molecules.")
    if cpu_cores == 1:
        results = [wrapper_func(smi) for smi in targets]
    else:
        with mp.Pool(processes=cpu_cores) as pool:
            print(f"Starting MP with ncpu={cpu_cores}")
            results = pool.map(wrapper_func, targets)
    print("\nFinished decoding.")

    # Print some results from the prediction
    # Note: If a syntree cannot be decoded within `max_depth` steps (15),
    #       we will count it as unsuccessful. The similarity will be 0.
    decoded = [smi for smi, _, _ in results]
    similarities = [sim for _, sim, _ in results]
    trees = [tree for _, _, tree in results]

    recovery_rate = (np.asfarray(similarities) == 1.0).sum() / len(similarities)
    avg_similarity = np.mean(similarities)
    n_successful = sum([syntree is not None for syntree in trees])

    print(f"  Total number of attempted  reconstructions: {len(targets)}")
    print(f"  Total number of successful reconstructions: {n_successful}")
    print(f"  {recovery_rate=}")
    print(f"  {avg_similarity=}")

    # Save to local dir
    # 1. Dataframe with targets, decoded, similarities
    # 2. Synthetic trees of the decoded SMILES
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir} ...")

    df = pd.DataFrame({"targets": targets, "decoded": decoded, "similarity": similarities})
    df.to_csv(f"{output_dir}/decoded_results.csv.gz", compression="gzip", index=False)

    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(f"{output_dir}/decoded_syntrees.json.gz")

    print("Completed.")
