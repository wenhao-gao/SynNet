"""
Generates synthetic trees where the root molecule optimizes for a specific objective
based on Therapeutics Data Commons (TDC) oracle functions.
Uses a genetic algorithm to optimize embeddings before decoding.
"""  # TODO: Refactor/Consolidate with generic inference script
import json
import multiprocessing as mp
import time
from functools import partial

import numpy as np
from tdc import Oracle

from synnet.encoding.distances import cosine_distance
from synnet.utils.ga_utils import crossover, mutation
from synnet.utils.predict_utils import synthetic_tree_decoder, tanimoto_similarity


def processor(emb, **kwargs):
    """
    Generates the synthetic tree for the input molecular embedding.

    Args:
        emb (np.ndarray): Molecular embedding to decode.

    Returns:
        str: SMILES for the final chemical node in the tree.
        SyntheticTree: The generated synthetic tree.
    """
    emb = emb.reshape((1, -1))
    try:
        tree, action = synthetic_tree_decoder(
            z_target=emb,
            **kwargs
        )
    except Exception as e:
        print(e)
        action = -1
    if action != 3:
        return None, None
    else:
        scores = np.array(tanimoto_similarity(emb, [node.smiles for node in tree.chemicals]))
        max_score_idx = np.where(scores == np.max(scores))[0][0]
        return tree.chemicals[max_score_idx].smiles, tree


def dock_drd3(smi):
    """
    Returns the docking score for the DRD3 target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the DRD3 target.
    """
    # define the oracle function from the TDC
    _drd3 = Oracle(name="drd3_docking")

    if smi is None:
        return 0.0
    else:
        try:
            return -_drd3(smi)
        except:
            return 0.0


def dock_7l11(smi):
    """
    Returns the docking score for the 7L11 target.

    Args:
        smi (str): SMILES for the molecule to predict the docking score of.

    Returns:
        float: Predicted docking score against the 7L11 target.
    """
    # define the oracle function from the TDC
    _7l11 = Oracle(name="7l11_docking")
    if smi is None:
        return 0.0
    else:
        try:
            return -_7l11(smi)
        except:
            return 0.0


def fitness(embs, _pool, obj, func):
    """
    Returns the scores for the root molecules in synthetic trees generated by the
    input molecular embeddings.

    Args:
        embs (list): Contains molecular embeddings (vectors).
        _pool (mp.Pool): A pool object, which represents a pool of workers (used
            for multiprocessing).
        obj (str): The objective function to use to compute the fitness.

    Raises:
        ValueError: Raised if the specified objective function is not implemented.

    Returns:
        scores (list): Contains the scores for the root molecules in the
            generated trees.
        smiles (list): Contains the root molecules encoded as SMILES strings.
        trees (list): Contains the synthetic trees generated from the input
            embeddings.
    """
    results = _pool.map(func, embs)
    smiles = [r[0] for r in results]
    trees = [r[1] for r in results]

    if obj == "qed":
        # define the oracle function from the TDC
        qed = Oracle(name="QED")
        scores = [qed(smi) for smi in smiles]
    elif obj == "logp":
        # define the oracle function from the TDC
        logp = Oracle(name="LogP")
        scores = [logp(smi) for smi in smiles]
    elif obj == "jnk":
        # define the oracle function from the TDC
        jnk = Oracle(name="JNK3")
        scores = [jnk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "gsk":
        # define the oracle function from the TDC
        gsk = Oracle(name="GSK3B")
        scores = [gsk(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "drd2":
        # define the oracle function from the TDC
        drd2 = Oracle(name="DRD2")
        scores = [drd2(smi) if smi is not None else 0.0 for smi in smiles]
    elif obj == "7l11":
        scores = [dock_7l11(smi) for smi in smiles]
    elif obj == "drd3":
        scores = [dock_drd3(smi) for smi in smiles]
    else:
        raise ValueError("Objective function not implemneted")
    return scores, smiles, trees


def distribution_schedule(n, total):
    """
    Determines the type of probability to use in the `crossover` function, based
    on the number of generations which have occured.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        str: Describes a type of probability distribution.
    """
    if n < 4 * total / 5:
        return "linear"
    else:
        return "softmax_linear"


def num_mut_per_ele_scheduler(n, total):
    """
    Determines the number of bits to mutate in each vector, based on the number
    of elapsed generations.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        int: Number of bits to mutate.
    """
    # if n < total/2:
    #     return 256
    # else:
    #     return 512
    return 24


def mut_probability_scheduler(n, total):
    """
    Determines the probability of mutating a vector, based on the number of elapsed
    generations.

    Args:
        n (int): Number of elapsed generations.
        total (int): Total number of expected generations.

    Returns:
        float: The probability of mutation.
    """
    if n < total / 2:
        return 0.5
    else:
        return 0.5


def optimize(population,
             bblocks,
             rxns_collection,
             checkpoints,
             mol_embedder,
             output_dir,
             cpu_cores,
             num_gen,
             objective,
             num_offspring):
    # define some constants (here, for the Hartenfeller-Button test set)
    nbits = 4096
    rxn_template = "hb"

    # load the purchasable building block embeddings
    bblocks_mol_embedder = mol_embedder.init_balltree(cosine_distance)
    bb_emb = bblocks_mol_embedder.get_embeddings()

    # load the purchasable building block SMILES to a dictionary
    # A dict is used as lookup table for 2nd reactant during inference:
    bb_dict = {block: i for i, block in enumerate(bblocks)}

    # load the reaction templates as a ReactionSet object

    # load the pre-trained modules
    act_net, rt1_net, rxn_net, rt2_net = checkpoints

    func = partial(processor,
                   building_blocks=bblocks,
                   bb_dict=bb_dict,
                   reaction_templates=rxns_collection.rxns,
                   mol_embedder=bblocks_mol_embedder.kdtree,  # TODO: fix this, currently misused,
                   action_net=act_net,
                   reactant1_net=rt1_net,
                   rxn_net=rxn_net,
                   reactant2_net=rt2_net,
                   bb_emb=bb_emb,
                   rxn_template=rxn_template,
                   n_bits=nbits,
                   max_step=15)

    # Evaluation initial population
    with mp.Pool(processes=cpu_cores) as pool:
        scores, mols, trees = fitness(embs=population, _pool=pool, obj=objective, func=func)

    scores = np.array(scores)
    score_x = np.argsort(scores)
    population = population[score_x[::-1]]
    mols = [mols[i] for i in score_x[::-1]]
    scores = scores[score_x[::-1]]
    print(f"Initial: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"Scores: {scores}")
    print(f"Top-3 Smiles: {mols[:3]}")

    # Genetic Algorithm: loop over generations
    recent_scores = []
    for n in range(num_gen):
        print(f"Starting generation {n}")
        t = time.time()

        dist_ = distribution_schedule(n, num_gen)
        num_mut_per_ele_ = num_mut_per_ele_scheduler(n, num_gen)
        mut_probability_ = mut_probability_scheduler(n, num_gen)

        offspring = crossover(
            parents=population, offspring_size=num_offspring, distribution=dist_
        )
        offspring = mutation(
            offspring_crossover=offspring,
            num_mut_per_ele=num_mut_per_ele_,
            mut_probability=mut_probability_,
        )
        new_population = np.unique(np.concatenate([population, offspring], axis=0), axis=0)
        with mp.Pool(processes=cpu_cores) as pool:
            new_scores, new_mols, trees = fitness(new_population, pool, objective, func=func)
        new_scores = np.array(new_scores)
        scores = []
        mols = []

        parent_idx = 0
        indices_to_print = []
        while parent_idx < len(population):
            max_score_idx = np.where(new_scores == np.max(new_scores))[0][0]
            if new_mols[max_score_idx] not in mols:
                indices_to_print.append(max_score_idx)
                scores.append(new_scores[max_score_idx])
                mols.append(new_mols[max_score_idx])
                population[parent_idx, :] = new_population[max_score_idx, :]
                new_scores[max_score_idx] = -999999
                parent_idx += 1
            else:
                new_scores[max_score_idx] = -999999

        scores = np.array(scores)
        print(f"Generation {n + 1}: {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"Scores: {scores}")
        print(f"Top-3 Smiles: {mols[:3]}")
        print(f"Consumed time: {(time.time() - t):.3f} s")
        print()
        for i in range(3):
            trees[indices_to_print[i]]._print()
        print()

        recent_scores.append(scores.mean())
        if len(recent_scores) > 10:
            del recent_scores[0]

        np.save("population_" + objective + "_" + str(n + 1) + ".npy", population)

        data = {
            "objective": objective,
            "top1": np.mean(scores[:1]),
            "top10": np.mean(scores[:10]),
            "top100": np.mean(scores[:100]),
            "smiles": mols,
            "scores": scores.tolist(),
        }
        with open("opt_" + objective + ".json", "w") as f:
            json.dump(data, f)

        if n > 30 and recent_scores[-1] - recent_scores[0] < 0.01:
            print("Early Stop!")
            break

    # Save results
    data = {
        "objective": objective,
        "top1": np.mean(scores[:1]),
        "top10": np.mean(scores[:10]),
        "top100": np.mean(scores[:100]),
        "smiles": mols,
        "scores": scores.tolist(),
    }
    with (output_dir / ("opt_" + objective + ".json")).open("w") as f:
        json.dump(data, f)

    np.save("population_" + objective + ".npy", population)
