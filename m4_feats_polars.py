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

#POLARS
def down_time_padding(train_logs, test_logs, time_agg):

    data = []
    for logs in [train_logs, test_logs]:
    # bin original logs
        logs_binned = logs.clone()
        logs_binned = logs_binned.with_columns((pl.col('down_time') / 1000).alias('down_time_sec'))
        logs_binned = logs_binned.with_columns(((pl.col('down_time_sec') // time_agg) * time_agg).alias('time_bin'))

        grp_binned = logs_binned.group_by(['id', 'time_bin']).agg(pl.max('word_count'))
        grp_binned = grp_binned.with_columns(pl.col('time_bin').cast(pl.Int64))
        grp_binned = grp_binned.sort([pl.col('id'), pl.col('time_bin')])

        # get max down_time value from logs
        max_logs = logs.clone()
        max_down_time = max_logs.group_by(['id']).agg(pl.max('down_time') / 1000)
        max_down_time = max_down_time.with_columns([pl.col('down_time').cast(pl.Int64)])

        padding_dataframes = []
        max_down_time = max_down_time.collect()

        # Iterate over each row in max_down_time_df
        for row in max_down_time.rows():
            id_value, max_time_value = row[0], row[1]  # Access by index

            # Generate time steps
            time_steps = list(range(0, max_time_value + time_agg, time_agg))

            # Create padding DataFrame with the correct types
            padding_df = pl.DataFrame({
                'id': [str(id_value)] * len(time_steps),
                'time_bin': time_steps
            })

            padding_dataframes.append(padding_df)

        pad_df = pl.concat(padding_dataframes).lazy()
        grp_df = pad_df.join(grp_binned.lazy(), on=['id', 'time_bin'], how='left')
        grp_df = grp_df.sort([pl.col('id'), pl.col('time_bin')])
        grp_df = grp_df.with_columns(pl.col('word_count').fill_null(strategy="forward").over('id'))
        data.append(grp_df.collect())

    return data[0].lazy(), data[1].lazy()

# POLARS
def rate_of_change_feats(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:

        logs = data.clone()
        logs = logs.sort('id')
        logs = logs.with_columns([
            pl.col('word_count').diff().over('id').alias('word_count_diff'),
            pl.col('time_bin').diff().over('id').alias('time_bin_diff')
        ]).fill_nan(0)

        logs = logs.with_columns(
            (pl.col('word_count_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')).fill_nan(0)

        # Aggregating
        stats = logs.group_by('id').agg([
            pl.col('rate_of_change').count().alias('roc_count'),
            pl.col('rate_of_change').mean().alias('roc_mean'),
            pl.col('rate_of_change').std().alias('roc_std'),
            pl.col('rate_of_change').sum().alias('roc_sum'),
            pl.col('rate_of_change').max().alias('roc_max'),
            pl.col('rate_of_change').quantile(0.25).alias('roc_q1'),
            pl.col('rate_of_change').median().alias('roc_median'),
            pl.col('rate_of_change').quantile(0.75).alias('roc_q3'),
            pl.col('rate_of_change').kurtosis().alias('roc_kurt'),
            pl.col('rate_of_change').skew().alias('roc_skew'),
        ])

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