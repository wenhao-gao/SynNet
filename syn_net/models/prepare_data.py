"""
Prepares the training, testing, and validation data by reading in the states
and steps for the reaction data and re-writing it as separate one-hot encoded
action, reactant 1, reactant 2, and reaction files.
"""
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

# TODO add comments

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--outputembedding", type=str, default='gin',
                        help="Choose from ['fp_4096', 'fp_256', 'gin', 'rdkit2d']")
    args = parser.parse_args()
    rxn_template = args.rxn_template
    featurize = args.featurize
    output_emb = args.outputembedding

    main_dir = '/pool001/whgao/data/synth_net/' + rxn_template + '_' + featurize + '_' + str(args.radius) + '_' + str(args.nbits) + '_' + str(args.outputembedding) + '/'
    if rxn_template == 'hb':
        num_rxn = 91
    elif rxn_template == 'pis':
        num_rxn = 4700

    if output_emb == 'gin':
        out_dim = 300
    elif output_emb == 'rdkit2d':
        out_dim = 200
    elif output_emb == 'fp_4096':
        out_dim = 4096
    elif output_emb == 'fp_256':
        out_dim = 256


    for dataset in ['train', 'valid', 'test']:
    # for dataset in ['valid']:

        print('Reading ' + dataset + ' data ......')
        states_list = []
        steps_list = []
        for i in range(1):
            states_list.append(sparse.load_npz(main_dir + 'states_' + str(i) + '_' + dataset + '.npz'))
            steps_list.append(sparse.load_npz(main_dir + 'steps_' + str(i) + '_' + dataset + '.npz'))

        states = sparse.csc_matrix(sparse.vstack(states_list))
        steps = sparse.csc_matrix(sparse.vstack(steps_list))

        X = states
        y = steps[:, 0]

        sparse.save_npz(main_dir + 'X_act_' + dataset + '.npz', X)
        sparse.save_npz(main_dir + 'y_act_' + dataset + '.npz', y)

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 3).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 3).reshape(-1, )])

        X = sparse.hstack([states, steps[:, (2 * out_dim + 2):]])
        y = steps[:, out_dim + 1]

        sparse.save_npz(main_dir + 'X_rxn_' + dataset + '.npz', X)
        sparse.save_npz(main_dir + 'y_rxn_' + dataset + '.npz', y)

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 2).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 2).reshape(-1, )])

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit([[i] for i in range(num_rxn)])
        # import ipdb; ipdb.set_trace(context=9)
        X = sparse.hstack([states, steps[:, (2 * out_dim + 2):], sparse.csc_matrix(enc.transform(steps[:, out_dim+1].A.reshape((-1, 1))).toarray())])
        y = steps[:, (out_dim+2): (2 * out_dim + 2)]

        sparse.save_npz(main_dir + 'X_rt2_' + dataset + '.npz', X)
        sparse.save_npz(main_dir + 'y_rt2_' + dataset + '.npz', y)

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 1).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 1).reshape(-1, )])

        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit([[i] for i in range(4)])

        # X = sparse.hstack([states, sparse.csc_matrix(enc.transform(steps[:, 0].A.reshape((-1, 1))).toarray())])
        X = states
        y = steps[:, 1: (out_dim+1)]

        sparse.save_npz(main_dir + 'X_rt1_' + dataset + '.npz', X)
        sparse.save_npz(main_dir + 'y_rt1_' + dataset + '.npz', y)

    print('Finish!')
