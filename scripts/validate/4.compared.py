import pandas as pd


if __name__ == '__main__':
    reweighting = pd.read_csv(
        'benchmark/scripts/prime_persistent/best_val.csv')
    reweighting['metal'] = reweighting['metal'].fillna('NA')
    hard_mining = pd.read_csv(
        'benchmark/scripts/prime_persistent/hard_mining.csv')
    hard_mining['metal'] = hard_mining['metal'].fillna('NA')
    f1_hardmining = {}
    for i, row in hard_mining.iterrows():
        f1_hardmining[row['metal']] = row['test_F1']
    f1_reweighting = {}
    for i, row in reweighting.iterrows():
        f1_reweighting[row['metal']] = row['best_test_F1']

    print('Hard Mining vs Reweighting')
    for metal in f1_hardmining.keys():
        print(
            f"Metal: {metal},\tHard Mining F1: {f1_hardmining[metal]:.4f},\tReweighting F1: {f1_reweighting.get(metal, 0):4f}")
