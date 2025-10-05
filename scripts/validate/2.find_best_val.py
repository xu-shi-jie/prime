import pandas as pd
# fmt: off
import sys
sys.path.append('.')
from models.dataset import considered_metals
# fmt: on

if __name__ == '__main__':
    df = pd.read_csv('benchmark/scripts/prime_persistent/all_metals.csv',)
    df['metal'] = df['metal'].fillna('NA')

    print(f'Found {len(df["metal"].unique())} metals in all_metals.csv')
    patience = 5
    df_test = df[df['dataset'] == 'test']
    df_val = df[df['dataset'] == 'val']
    df_test = df_test.rename(columns={
        'TP': 'test_TP',
        'FP': 'test_FP',
        'FN': 'test_FN',
        'F1': 'test_F1',
        'RMSE': 'test_RMSE'
    })
    df_val = df_val.rename(columns={
        'TP': 'val_TP',
        'FP': 'val_FP',
        'FN': 'val_FN',
        'F1': 'val_F1',
        'RMSE': 'val_RMSE'
    })
    df_test.drop(columns=['dataset'], inplace=True)
    df_val.drop(columns=['dataset'], inplace=True)
    df = pd.merge(
        df_test, df_val, on=['file', 'metal', 'cnn_layers', 'pretrained', 'loss_type', 'epoch'])
    df['val_pre'] = df['val_TP'] / (df['val_TP'] + df['val_FP'])
    df['test_pre'] = df['test_TP'] / (df['test_TP'] + df['test_FP'])
    df['val_rec'] = df['val_TP'] / (df['val_TP'] + df['val_FN'])
    df['test_rec'] = df['test_TP'] / (df['test_TP'] + df['test_FN'])
    df['rank'] = df['val_F1']-df['val_RMSE']/10
    # remove rows with NaN values
    df = df.dropna()
    df = df.sort_values(by=['metal', 'rank'], ascending=False)
    df.to_csv('benchmark/scripts/prime_persistent/sort_by_val.csv', index=False)
    df = df.groupby(['metal'])
    best_df = []
    for metal, group in df:
        group = group.reset_index(drop=True)
        best_row = group.iloc[0].copy()
        best_row['max_epoch'] = group['epoch'].max()
        best_df.append(best_row)
    best_df = pd.DataFrame(best_df)
    best_df = best_df.rename(columns={
        'val_TP': 'best_val_TP',
        'val_FP': 'best_val_FP',
        'val_FN': 'best_val_FN',
        'val_F1': 'best_val_F1',
        'val_RMSE': 'best_val_RMSE',
        'val_pre': 'best_val_pre',
        'val_rec': 'best_val_rec',
        'test_TP': 'best_test_TP',
        'test_FP': 'best_test_FP',
        'test_FN': 'best_test_FN',
        'test_F1': 'best_test_F1',
        'test_RMSE': 'best_test_RMSE',
        'test_pre': 'best_test_pre',
        'test_rec': 'best_test_rec',
        'rank': 'best_rank'
    })
    best_df = best_df.drop(
        columns=['file', 'cnn_layers', 'pretrained', 'loss_type'])
    # move max_epoch to the beginning
    best_df = best_df[['metal', 'epoch', 'max_epoch'] +
                      [col for col in best_df.columns if col not in ['metal', 'epoch', 'max_epoch']]]
    best_df.to_csv(
        'benchmark/scripts/prime_persistent/best_val.csv', index=False)
    finished_metals = best_df['metal'].unique()
    undoing_metals = set(considered_metals) - set(finished_metals)
    print(
        f'Finished metals: {finished_metals}, Undoing metals: {undoing_metals}')
