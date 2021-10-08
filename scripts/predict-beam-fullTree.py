"""
This file contains the code to decode synthetic trees from fingerprints, with 
networks.
"""
import os
from typing import Tuple, Union, Callable
import pandas as pd
import numpy as np
import rdkit
from tqdm import tqdm
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from synth_net.utils.data_utils import Reaction, ReactionSet, SyntheticTree, SyntheticTreeSet
from sklearn.neighbors import KDTree

import dgl
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from tdc.chem_utils import MolConvert
from synth_net.models.mlp import MLP

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("-v", "--version", type=int, default=1,
                        help="Version")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=1024,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=300,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=16,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width to use for Reactant1 search")
    parser.add_argument("-n", "--num", type=int, default=-1,
                        help="Number of molecules to decode.")
    parser.add_argument("-d", "--data", type=str, default='test',
                        help="Choose from ['train', 'valid', 'test']")
    args = parser.parse_args()

    def softmax(x : Union[np.ndarray, list]) -> Union[np.ndarray, list]:
        """Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def can_react(state : np.ndarray, rxns : list) -> Tuple[np.ndarray, np.ndarray]:
        """Determines if two molecules can react using any of the input reactions.
    
        Parameters
        ----------
        state : np.ndarray
            The current state in the synthetic tree.
        rxns : list of Reaction objects
            Contains available reaction templates.
    
        Returns
        -------
        np.ndarray
            The sum of the reaction mask tells us how many reactions are viable for 
            the two molecules.
        np.ndarray
            The reaction mask. Masks out reactions which are not viable for the two 
            molecules.
        """
        mol1 = state.pop()
        mol2 = state.pop()
        reaction_mask = [int(rxn.run_reaction([mol1, mol2]) is not None) for rxn in rxns]
        return sum(reaction_mask), reaction_mask
    
    def get_action_mask(state : np.ndarray, rxns : list) -> np.ndarray:
        """Determines which actions can apply to a given state in the synthetic 
        tree and returns a mask for which actions can apply.
    
        Parameters
        ----------
        state : np.ndarray
            The current state in the synthetic tree.
        rxns : list of Reaction objects
            Contains available reaction templates.
    
        Returns
        -------
        np.ndarray
            The action mask. Masks out unviable actions from the current state using
            0s, with 1s at the positions corresponding to viable actions.
        """
        # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
        if len(state) == 0:
            return np.array([1, 0, 0, 0])
        elif len(state) == 1:
            return np.array([1, 1, 0, 1])
        elif len(state) == 2:
            can_react_, _ = can_react(state, rxns)
            if can_react_:
                return np.array([0, 1, 1, 0])
            else:
                return np.array([0, 1, 0, 0])
        else:
            raise ValueError('Problem with state.')
    
    def get_reaction_mask(smi : str, rxns : list) -> Tuple[Union[list, None], Union[list, None]]:
        """Determines which reaction templates can apply to the input molecule.
    
        Parameters
        ----------
        smi : str
            The SMILES string corresponding to the molecule in question.
        rxns : list of Reaction objects
            Contains available reaction templates.
    
        Returns
        -------
        reaction_mask : list of ints, or None
            The reaction template mask. Masks out reaction templates which are not 
            viable for the input molecule. If there are no viable reaction templates
            identified, is simply None.
        available_list : list of lists, or None
            Contains available reactants if at least one viable reaction  template is 
            identified. Else is simply None.
        """
        reaction_mask = [int(rxn.is_reactant(smi)) for rxn in rxns]
    
        if sum(reaction_mask) == 0:
            return None, None
        available_list = []
        mol = rdkit.Chem.MolFromSmiles(smi)
        for i, rxn in enumerate(rxns):
            if reaction_mask[i] and rxn.num_reactant == 2:
    
                if rxn.is_reactant_first(mol):
                    available_list.append(rxn.available_reactants[1])
                elif rxn.is_reactant_second(mol):
                    available_list.append(rxn.available_reactants[0])
                else:
                    raise ValueError('Check the reactants')
    
                if len(available_list[-1]) == 0:
                    reaction_mask[i] = 0
    
            else:
                available_list.append([])

        return reaction_mask, available_list
    
    def graph_construction_and_featurization(smiles : list) -> Tuple[list, list]:
        """Constructs graphs from SMILES and featurizes them.
    
        Parameters
        ----------
        smiles : list of str
            SMILES of molecules to embed.
    
        Returns
        -------
        graphs : list of DGLGraph
            List of featurized graphs which were constructed.
        success : list of bool
            Indicators for whether the SMILES string can be parsed by RDKit.
        """
        graphs = []
        success = []
        for smi in tqdm(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    success.append(False)
                    continue
                g = mol_to_bigraph(mol, add_self_loop=True,
                                   node_featurizer=PretrainAtomFeaturizer(),
                                   edge_featurizer=PretrainBondFeaturizer(),
                                   canonical_atom_order=False)
                graphs.append(g)
                success.append(True)
            except:
                success.append(False)
    
        return graphs, success
        
    def one_hot_encoder(dim : int, space : int) -> np.ndarray:
        """Returns one-hot encoded vector of length=`space` containing a one at 
        the specified `dim` and zero elsewhere.
        """
        vec = np.zeros((1, space))
        vec[0, dim] = 1
        return vec
    
    readout = AvgPooling()  # TODO try SumPooling
    model_type = 'gin_supervised_contextpred'
    device = 'cuda:0'
    mol_embedder = load_pretrained(model_type).to(device)
    mol_embedder.eval()

    def mol_embedding(smi : str, device : str='cuda:0') -> torch.Tensor:  # TODO check how is this different from function below, might be duplicate
        """Computes the molecular graph embedding for the input SMILES.

        Parameters
        ----------
        smi : str
            SMILES of molecule to embed.
        device : str
            Indicates the device to run on. Default 'cuda:0'.

        Returns
        -------
        torch.Tensor
            The graph embedding.
        """
        if smi is None:
            return np.zeros(args.out_dim)
        else:
            mol = Chem.MolFromSmiles(smi)
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            bg = g.to(device)
            nfeats = [bg.ndata.pop('atomic_number').to(device),
                      bg.ndata.pop('chirality_type').to(device)]
            efeats = [bg.edata.pop('bond_type').to(device),
                      bg.edata.pop('bond_direction_type').to(device)]
            with torch.no_grad():
                node_repr = mol_embedder(bg, nfeats, efeats)
            return readout(bg, node_repr).detach().cpu().numpy().reshape(1, -1)
    
    def get_mol_embedding(smi : str, 
                          model : Callable=mol_embedder, 
                          device : str='cuda:0', 
                          readout : Callable=readout) -> torch.Tensor:
        """Computes the molecular graph embedding for the input SMILES.
    
        Parameters
        ----------
        smi : str
            SMILES of molecule to embed.
        model : dgllife.model
            Pre-trained NN model to use for computing the embedding. Default GIN.
        device : str
            Indicates the device to run on. Default 'cuda:0'.
        readout : dgl.nn.pytorch.glob
            Readout function to use for computing the graph embedding.
    
        Returns
        -------
        torch.Tensor
            Learned embedding for input molecule.
        """
        mol = Chem.MolFromSmiles(smi)
        g = mol_to_bigraph(mol, add_self_loop=True,
                       node_featurizer=PretrainAtomFeaturizer(),
                       edge_featurizer=PretrainBondFeaturizer(),
                       canonical_atom_order=False)
        bg = g.to(device)
        nfeats = [bg.ndata.pop('atomic_number').to(device),
                  bg.ndata.pop('chirality_type').to(device)]
        efeats = [bg.edata.pop('bond_type').to(device),
                  bg.edata.pop('bond_direction_type').to(device)]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        return readout(bg, node_repr).detach().cpu().numpy()
    
    def mol_fp(smi : str, radius : int=2, nBits : int=args.nbits) -> np.ndarray:
        """Computes the Morgan fingerprint for the input SMILES.

        Parameters
        ----------
        smi : str
            SMILES for molecule to compute fingerprint for.
        radius : int
            Fingerprint radius to use.
        nBits : int
            Number of bits to use for fingerprint (i.e. length of fingerprint).
        
        Returns
        -------
        features : np.ndarray
            For valid SMILES, this is the fingerprint. Otherwise, if the input
            SMILES is bad, this can be a purely zero vector.
        """
        if smi is None:
            features = np.zeros(nBits)
            return features
        else:
            mol = Chem.MolFromSmiles(smi)
            try:
                features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            except:
                features = np.zeros(nBits)
                return features
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
            return features
    
    bb_emb = np.load('/pool001/whgao/data/synth_net/st_' + args.rxn_template + '/enamine_us_emb.npy')
    kdtree = KDTree(bb_emb, metric='euclidean')
    def nn_search(_e : np.ndarray, _tree : "KDTree"=kdtree, _k : int=1) -> \
                  Tuple[Union[float, np.ndarray], Union[int, np.ndarray]]:
        """Conducts a nearest-neighbors search.

        Parameters
        ----------
        _e : np.ndarray
            A specific point in the dataset.
        _tree : sklearn.neighbors._kd_tree.KDTree
            A k-dimensional tree.
        _k : int
            Indicates how many nearest neighbors to get (k-nearest).

        Returns
        -------
        float or np.ndarray of floats
            The distances to the nearest neighbors.
        int or np.ndarray of ints
            The indices of the nearest neighbors.
        """
        dist, ind = _tree.query(_e, k=_k)
        return dist[0], ind[0]
    
    def set_embedding(z_target : np.ndarray, 
                      state : list,
                      _mol_embedding : Callable=get_mol_embedding) -> np.ndarray:
        """Computes embeddings for all molecules in input state.

        Parameters
        ----------
        z_target : np.ndarray
            Embedding for the target molecule
        state : list
            Contains molecules in the current state, if not the initial state
        _mol_embedding : Callable
            Function to use for computing the embeddings of the first and second
            molecules in the state (e.g. Morgan fingerprint)

        Returns
        -------
        np.ndarray
            Embedding consisting of the concatenation of the target molecule
            with the current molecules (if available) in the input state.
        """
        if len(state) == 0:
            z_target = np.expand_dims(z_target, axis=0)
            return np.concatenate([np.zeros((1, 2 * args.nbits)), z_target], axis=1)
        else:
            e1 = _mol_embedding(state[0])
            e1 = np.expand_dims(e1, axis=0)
            if len(state) == 1:
                e2 = np.zeros((1, args.nbits))
            else:
                e2 = _mol_embedding(state[1])
                e2 = np.expand_dims(e2, axis=0)
            z_target = np.expand_dims(z_target, axis=0)
            return np.concatenate([e1, e2, z_target], axis=1)
    
    def synthetic_tree_decoder(z_target : np.ndarray, 
                               building_blocks : list, 
                               bb_dict : dict,
                               reaction_templates : list, 
                               mol_embedder : "GIN", 
                               action_net : "MLP", 
                               reactant1_net : "MLP", 
                               rxn_net : "MLP", 
                               reactant2_net : "MLP", 
                               max_step : int=15) -> Tuple["KDTree", int]:
        """Computes the synthetic tree given an input molecule embedding, using
        the Action, Reaction, Reactant1, and Reactant2 networks and a greedy search.

        Parameters
        ----------
        z_target : np.ndarray
            Embedding for the target molecule.
        building_blocks : list of str
            Contains available building blocks.
        bb_dict : dict
            Building block dictionary.
        reaction_templates : list of Reactions
            Contains reaction templates.
        mol_embedder : dgllife.model.gnn.gin.GIN
            GNN to use for obtaining molecular embeddings.
        action_net : synth_net.models.mlp.MLP
            The action network.
        reactant1_net : synth_net.models.mlp.MLP
            The reactant1 network.
        rxn_net : synth_net.models.mlp.MLP
            The reaction network.
        reactant2_net : synth_net.models.mlp.MLP
            The reactant2 network.
        max_steps : int
            Maximum number of steps to include in the synthetic tree.

        Returns
        -------
        tree : sklearn.neighbors._kd_tree.KDTree
            The final synthetic tree.
        act : int
            The final action (to know if the tree was "properly" terminated).
        """
        # Initialization
        tree = SyntheticTree()
        mol_recent = None
    
        # Start iteration
        # try:
        for i in range(max_step):
            # Encode current state
            # from ipdb import set_trace; set_trace(context=11)
            state = tree.get_state() # a set
            z_state = set_embedding(z_target, state, mol_fp)
    
            # Predict action type, masked selection
            # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
            action_proba = action_net(torch.Tensor(z_state)) 
            action_proba = action_proba.squeeze().detach().numpy()
            action_mask = get_action_mask(tree.get_state(), reaction_templates)
            act = np.argmax(action_proba * action_mask)
    
            #import ipdb; ipdb.set_trace(context=9)
            
            z_mol1 = reactant1_net(torch.Tensor(np.concatenate([z_state, one_hot_encoder(act, 4)], axis=1)))
            z_mol1 = z_mol1.detach().numpy()
    
            # Select first molecule
            if act == 3:
                # End
                mol1_nlls = [0.0]
                break
            elif act == 0:
                # Add
                # **don't try to sample more points than there are in the tree
                # beam search for mol1 candidates
                dist, ind = nn_search(z_mol1, _k=min(len(bb_emb), args.beam_width))
                try:
                    mol1_probas = softmax(- 0.1 * dist)
                    mol1_nlls = -np.log(mol1_probas)
                except:  # exception for beam search of length 1
                    mol1_nlls = [-np.log(0.5)]
                mol1_list = [building_blocks[idx] for idx in ind]
            else:
                # Expand or Merge
                mol1_list = [mol_recent]
                mol1_nlls = [-np.log(0.5)]

            action_tuples = []  # list of action tuples created by beam search
            act_list      = [act] * args.beam_width
            for mol1_idx, mol1 in enumerate(mol1_list):

                # z_mol1 = get_mol_embedding(mol1, mol_embedder)
                z_mol1 = mol_fp(mol1)
                act = act_list[mol1_idx]
    
                # Select reaction
                z_mol1 = np.expand_dims(z_mol1, axis=0)
                reaction_proba = rxn_net(torch.Tensor(np.concatenate([z_state, z_mol1], axis=1)))
                reaction_proba = reaction_proba.squeeze().detach().numpy()
    
                if act != 2:
                    reaction_mask, available_list = get_reaction_mask(mol1, reaction_templates)
                else:
                    _, reaction_mask = can_react(tree.get_state(), reaction_templates)
                    available_list = [[] for rxn in reaction_templates]
    
                if reaction_mask is None:
                    if len(state) == 1:
                        act = 3
                        mol1_nlls[mol1_idx] += -np.log(action_proba * reaction_mask)[act]  # correct the NLL
                        act_list[mol1_idx] = act
                        #                     nll,                 act, mol1, rxn, rxn_id, mol2
                        action_tuples.append([mol1_nlls[mol1_idx], act, mol1, None, None, None])
                        continue
                    else:
                        #act = 3
                        #nlls[mol1_idx] += -np.log(action_proba * reaction_mask)[act]  # correct the NLL
                        act_list[mol1_idx] = act
                        #                     nll,                 act, mol1, rxn, rxn_id, mol2
                        action_tuples.append([mol1_nlls[mol1_idx], act, mol1, None, None, None])
                        continue

                rxn_ids = np.argsort(-reaction_proba * reaction_mask)[:args.beam_width]
                rxn_nlls = mol1_nlls[mol1_idx] - np.log(reaction_proba * reaction_mask)

                for rxn_id in rxn_ids:
                    rxn = reaction_templates[rxn_id]
                    rxn_nll = rxn_nlls[rxn_id]

                    if np.isinf(rxn_nll):
                        #                     nll,     act, mol1, rxn, rxn_id, mol2
                        action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, None])
                        continue
                    elif rxn.num_reactant == 2:
                        # Select second molecule
                        if act == 2:
                            # Merge
                            temp = set(state) - set([mol1])
                            mol2 = temp.pop()
                            #                     nll,     act, mol1, rxn, rxn_id, mol2
                            action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, mol2])
                        else:
                            # Add or Expand
                            if args.rxn_template == 'hb':
                                z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 91)], axis=1)))
                            elif args.rxn_template == 'pis':
                                z_mol2 = reactant2_net(torch.Tensor(np.concatenate([z_state, z_mol1, one_hot_encoder(rxn_id, 4700)], axis=1)))
                            z_mol2 = z_mol2.detach().numpy()
                            available = available_list[rxn_id]
                            available = [bb_dict[available[i]] for i in range(len(available))]
                            temp_emb = bb_emb[available]
                            available_tree = KDTree(temp_emb, metric='euclidean')
                            dist, ind = nn_search(z_mol2, _tree=available_tree, _k=min(len(temp_emb), args.beam_width))
                            try:
                                mol2_probas = softmax(-dist)
                                mol2_nlls = rxn_nll - np.log(mol2_probas)
                            except:
                                mol2_nlls = [rxn_nll + 0.0]
                            mol2_list = [building_blocks[available[idc]] for idc in ind]
                            for mol2_idx, mol2 in enumerate(mol2_list):
                                #                     nll,     act, mol1, rxn, rxn_id, mol2
                                action_tuples.append([mol2_nlls[mol2_idx], act, mol1, rxn, rxn_id, mol2])
                    else:
                        #                     nll,     act, mol1, rxn, rxn_id, mol2
                        action_tuples.append([rxn_nll, act, mol1, rxn, rxn_id, None])
                
            # Run reaction until get a valid (non-None) product
            for i in range(0, len(action_tuples)):
                nlls = list(zip(*action_tuples))[0]
                best_idx = np.argsort(nlls)[i]
                act      = action_tuples[best_idx][1]
                mol1     = action_tuples[best_idx][2]
                rxn      = action_tuples[best_idx][3]
                rxn_id   = action_tuples[best_idx][4]
                mol2     = action_tuples[best_idx][5]
                try:
                    mol_product = rxn.run_reaction([mol1, mol2])
                except:
                    mol_product = None
                else:
                    if mol_product is None:
                        continue
                    else:
                        break
            
            if mol_product is None or Chem.MolFromSmiles(mol_product) is None:
                if len(tree.get_state()) == 1:
                    act = 3
                    break
                else:
                    break

            # Update
            tree.update(act, int(rxn_id), mol1, mol2, mol_product)
            mol_recent = mol_product
        # except Exception as e:
        #     print(e)
        #     act = -1
        #     tree = None

        if act != 3:
            tree = tree
        else:
            tree.update(act, None, None, None, None)
        
        return tree, act
    
    path_to_reaction_file   = ('/pool001/whgao/data/synth_net/st_' + args.rxn_template 
                               + '/reactions_' + args.rxn_template + '.json.gz')
    path_to_building_blocks = ('/pool001/whgao/data/synth_net/st_' + args.rxn_template
                               + '/enamine_us_matched.csv.gz')
    
    param_path  = (f"/home/rociomer/SynthNet/pre-trained-models/{args.rxn_template}" 
                   f"_{args.featurize}_{args.radius}_{args.nbits}_v{args.version}/")
    path_to_act = param_path + 'act.ckpt'
    path_to_rt1 = param_path + 'rt1.ckpt'
    path_to_rxn = param_path + 'rxn.ckpt'
    path_to_rt2 = param_path + 'rt2.ckpt'
    
    np.random.seed(6)
    
    building_blocks = pd.read_csv(path_to_building_blocks, compression='gzip')['SMILES'].tolist()
    bb_dict = {building_blocks[i]: i for i in range(len(building_blocks))}
    
    rxn_set = ReactionSet()
    rxn_set.load(path_to_reaction_file)
    rxns = rxn_set.rxns
    
    ncpu = 16
    if args.featurize == 'fp':

        act_net = MLP.load_from_checkpoint(
            path_to_act,
            input_dim=int(3 * args.nbits),
            output_dim=4,
            hidden_dim=1000,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task='classification',
            loss='cross_entropy',
            valid_loss='accuracy',
            optimizer='adam',
            learning_rate=1e-4,
            ncpu=ncpu
        )

        rt1_net = MLP.load_from_checkpoint(
            path_to_rt1,
            input_dim=int(3 * args.nbits + 4),
            output_dim=args.out_dim,
            hidden_dim=1200,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task='regression',
            loss='mse',
            valid_loss='mse',
            optimizer='adam',
            learning_rate=1e-4,
            ncpu=ncpu
        )

        if args.rxn_template == 'hb':

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * args.nbits),
                output_dim=91,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )
            
            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(4 * args.nbits + 91),
                output_dim=args.out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='regression',
                loss='mse',
                valid_loss='mse',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

        elif args.rxn_template == 'pis':
            
            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * args.nbits),
                output_dim=4700,
                hidden_dim=4500,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )
            
            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(4 * args.nbits + 4700),
                output_dim=args.out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='regression',
                loss='mse',
                valid_loss='mse',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

    elif args.featurize == 'gin':

        act_net = MLP.load_from_checkpoint(
            path_to_act,
            input_dim=int(2 * args.nbits + args.out_dim),
            output_dim=4,
            hidden_dim=1000,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task='classification',
            loss='cross_entropy',
            valid_loss='accuracy',
            optimizer='adam',
            learning_rate=1e-4,
            ncpu=ncpu
        )
    
        rt1_net = MLP.load_from_checkpoint(
            path_to_rt1,
            input_dim=int(2 * args.nbits + args.out_dim + 4),
            output_dim=args.out_dim,
            hidden_dim=1200,
            num_layers=5,
            dropout=0.5,
            num_dropout_layers=1,
            task='regression',
            loss='mse',
            valid_loss='mse',
            optimizer='adam',
            learning_rate=1e-4,
            ncpu=ncpu
        )

        if args.rxn_template == 'hb':

            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(3 * args.nbits + args.out_dim),
                output_dim=91,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )
            
            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(3 * args.nbits + args.out_dim + 91),
                output_dim=args.out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='regression',
                loss='mse',
                valid_loss='mse',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

        elif args.rxn_template == 'pis':
            
            rxn_net = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(3 * args.nbits + args.out_dim),
                output_dim=4700,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )
            
            rt2_net = MLP.load_from_checkpoint(
                path_to_rt2,
                input_dim=int(3 * args.nbits + args.out_dim + 4700),
                output_dim=args.out_dim,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='regression',
                loss='mse',
                valid_loss='mse',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

    act_net.eval()
    rt1_net.eval()
    rxn_net.eval()
    rt2_net.eval()
    
    def decode_one_molecule(query_smi):
        if args.featurize == 'fp':
            z_target = mol_fp(query_smi, args.radius, args.nbits)
        elif args.featurize == 'gin':
            z_target = mol_embedding(query_smi)
        tree, action = synthetic_tree_decoder(z_target,
                                              building_blocks,
                                              bb_dict,
                                              rxns, 
                                              mol_embedder, 
                                              act_net, 
                                              rt1_net, 
                                              rxn_net, 
                                              rt2_net, 
                                              max_step=15)
        return tree, action


    path_to_data = '/pool001/whgao/data/synth_net/st_' + args.rxn_template + '/st_' + args.data +'.json.gz'
    print('Reading data from ', path_to_data)
    sts = SyntheticTreeSet()
    sts.load(path_to_data)
    query_smis = [st.root.smiles for st in sts.sts]  # TODO here could filter "~"s
    if args.num == -1:
        pass
    else:
        query_smis = query_smis[:args.num]

    output_smis = []
    similaritys = []
    trees = []
    num_finish = 0
    num_unfinish = 0

    print('Start to decode!')
    for smi in tqdm(query_smis):

        try:
            tree, action = decode_one_molecule(smi)
        except Exception as e:
            print(e)
            action = 1
            tree = None

        if action != 3:
            num_unfinish += 1
            output_smis.append(None)
            similaritys.append(None)
            trees.append(None)
        else:
            num_finish += 1
            output_smis.append(tree.root.smiles)
            ms = [Chem.MolFromSmiles(sm) for sm in [smi, tree.root.smiles]]
            fps = [Chem.RDKFingerprint(x) for x in ms]
            similaritys.append(DataStructs.FingerprintSimilarity(fps[0],fps[1]))
            trees.append(tree)

    print('Saving ......')
    save_path = '../results/' + args.rxn_template + '_' + args.featurize + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame({'query SMILES': query_smis, 'decode SMILES': output_smis, 'similarity': similaritys})
    print("mean similarities", df['similarity'].mean(), df['similarity'].std())
    print("NAs", df.isna().sum())
    df.to_csv(save_path + 'decode_result_' + args.data + '_bw_' + str(args.beam_width) + '.csv.gz', compression='gzip', index=False)
    
    synthetic_tree_set = SyntheticTreeSet(sts=trees)
    synthetic_tree_set.save(save_path + 'decoded_st_bw_' + str(args.beam_width) + '_' + args.data + '.json.gz')

    print('Finish!')
