import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from m4_feats_functions import getEssays

# POLARS
def normalise_up_down_times(train_logs, test_logs):
    new_logs = []
    for logs in [train_logs, test_logs]:
        min_down_time = logs.group_by('id').agg(pl.min('down_time').alias('min_down_time'))
        logs = logs.join(min_down_time, on='id', how='left')
        logs = logs.with_columns([
            (pl.col('down_time') - pl.col('min_down_time')).alias('normalised_down_time'),
            (pl.col('up_time') + pl.col('action_time')).alias('normalised_up_time')
        ])
        logs = logs.drop(['min_down_time', 'down_time', 'up_time'])
        logs = logs.rename({'normalised_down_time': 'down_time', 'normalised_up_time': 'up_time'})
        new_logs.append(logs)
    return new_logs[0], new_logs[1]

# Helper functions
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

def countvectorize_one_one(train_logs, test_logs):

    data = []

    for logs in [train_logs, test_logs]:

        ids = logs.id.unique()
        essays = getEssays(logs)
        c_vect = CountVectorizer(ngram_range=(1, 1))
        toks = c_vect.fit_transform(essays['essay']).todense()
        toks = toks[:,:16]
        toks_df = pd.DataFrame(columns = [f'tok_{i}' for i in range(toks.shape[1])], data=toks)
        toks_df['id'] = ids
        toks_df.reset_index(drop=True, inplace=True)
        data.append(toks_df)

    return data[0], data[1]

def down_time_padding(train_logs, test_logs, time_agg):

    padded = []

    for logs in [train_logs, test_logs]:

        # add time bins to original logs
        logs_binned = logs.copy()
        logs_binned['down_time_sec'] = logs_binned['down_time'] / 1000
        logs_binned['time_bin'] = logs_binned['down_time_sec'].apply(lambda x: time_agg * (x // time_agg))

        # bin logs without padding
        grp_binned = logs_binned.groupby(['id', 'time_bin'])['word_count'].max().reset_index() 

        # upper bound for padding
        logs_pad = logs.copy()
        max_down_time = logs_pad.groupby('id')['down_time'].max()
        max_down_time /= 1000

        pad_df = pd.DataFrame()
        for id, max_time in max_down_time.items():
            time_steps = list(range(0, int(max_time) + time_agg, time_agg))
            padding_df = pd.DataFrame({'id': id, 'time_bin': time_steps})
            pad_df = pd.concat([pad_df, padding_df])
            pad_df.append(pad_df)

        # right join padded bins and ffill
        grp_df = grp_binned.merge(pad_df, on=['id', 'time_bin'], how='right')
        grp_df = grp_df.ffill()
        padded.append(grp_df)

    return padded[0], padded[1]

# need to convert to polars
def rate_of_change_feats(train_logs, test_logs):

    AGGREGATIONS = ['count', 'mean', 'std', 'sum', 'max', q1, 'median', q3, 'skew', pd.DataFrame.kurt] #, ] 'min', ,
    feats = []
    for logs in [train_logs, test_logs]:

        # df = logs.copy()
        # max_down_time = df.groupby('id')['down_time'].max()
        # max_down_time /= 1000
# 
        # df = logs[['id', 'down_time', 'word_count']].copy()
        # df['down_time_sec'] = df['down_time'] / 1000
        # df['time_bin'] = df['down_time_sec'].apply(lambda x: time_agg * (x // time_agg))
        # grp_df = df.groupby(['id', 'time_bin'])['word_count'].max().reset_index()
# 
        word_count_diff = logs.groupby('id')['word_count'].diff().fillna(0)
        time_bin_diff = logs.groupby('id')['time_bin'].diff().fillna(0)
        logs['rate_of_change'] = (word_count_diff / time_bin_diff).fillna(0)

        stats = logs[['id','rate_of_change']].groupby(['id']).agg(AGGREGATIONS)
        stats.columns = ['_'.join(x) for x in stats.columns]
        feats.append(stats)

    return feats[0], feats[1]

def wc_acceleration_feats(train_logs, test_logs, time_agg):

    AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', q1, 'median', q3, 'skew', pd.DataFrame.kurt, 'sum']
    feats = []
    for logs in [train_logs, test_logs]:

        df = logs[['id', 'down_time', 'word_count']].copy()
        df['down_time_sec'] = df['down_time'] / 1000
        df['time_bin'] = df['down_time_sec'].apply(lambda x: time_agg * (x // time_agg))
        grp_df = df.groupby(['id', 'time_bin'])['word_count'].max().reset_index()

        word_count_diff = grp_df.groupby('id')['word_count'].diff().fillna(0)
        time_bin_diff = grp_df.groupby('id')['time_bin'].diff().fillna(0)
        grp_df['rate_of_change'] = (word_count_diff / time_bin_diff).fillna(0)
        rate_of_change_diff = grp_df.groupby('id')['rate_of_change'].diff().fillna(0)
        grp_df['acceleration'] = (rate_of_change_diff / time_bin_diff).fillna(0)

        stats = grp_df[['id','acceleration']].groupby(['id']).agg(AGGREGATIONS)
        stats.columns = ['_'.join(x) for x in stats.columns]
        feats.append(stats)

    return feats[0], feats[1]