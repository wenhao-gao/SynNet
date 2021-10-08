from tdc import Evaluator
import pandas as pd
import numpy as np

kl_divergence = Evaluator(name = 'KL_Divergence')
fcd_distance = Evaluator(name = 'FCD_Distance')
novelty = Evaluator(name = 'Novelty')
validity = Evaluator(name = 'Validity')
uniqueness = Evaluator(name = 'Uniqueness')

if __name__ == '__main__':
    result_train = pd.read_csv('../results/decode_result_test_processed_property.csv.gz', compression='gzip')
    # result_test_unrecover = result_train[result_train['recovered sa'] != -1][result_train['similarity'] != 1.0]
    result_test_unrecover = result_train[result_train['recovered sa'] != -1]
    # print(f"Novelty: {novelty(result_test_unrecover['query SMILES'].tolist(), result_test_unrecover['decode SMILES'].tolist())}")
    # print(f"Validity: {validity(result_test_unrecover['decode SMILES'].tolist())}")
    # print(f"Uniqueness: {uniqueness(result_test_unrecover['decode SMILES'].tolist())}")
    print(f"FCD: {fcd_distance(result_test_unrecover['query SMILES'].tolist(), result_test_unrecover['decode SMILES'].tolist())}")
    print(f"KL: {kl_divergence(result_test_unrecover['query SMILES'].tolist()[:10000], result_test_unrecover['decode SMILES'].tolist()[:10000])}")

