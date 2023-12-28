import pandas as pd
import polars as pl
import numpy as np
import re
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import skew, kurtosis

pl.set_random_seed(42)

def normalise_up_down_times(train_logs, test_logs):
    new_logs = []
    for logs in [train_logs, test_logs]:
        min_down_time = logs.group_by('id').agg(pl.min('down_time').alias('min_down_time'))
        logs = logs.join(min_down_time, on='id', how='left')
        logs = logs.with_columns([(pl.col('down_time') - pl.col('min_down_time')).alias('normalised_down_time')])
        logs = logs.with_columns([(pl.col('normalised_down_time') + pl.col('action_time')).alias('normalised_up_time')])
        logs = logs.drop(['min_down_time', 'down_time', 'up_time'])
        logs = logs.rename({'normalised_down_time': 'down_time', 'normalised_up_time': 'up_time'})
        new_logs.append(logs.sort(['id','event_id'], descending=[True,True]))
    return new_logs[0], new_logs[1]

# Helper functions
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

def down_time_padding(train_logs, test_logs, time_agg):

    data = []
    for logs in [train_logs, test_logs]:
    # bin original logs
        logs_binned = logs.clone()
        logs_binned = logs_binned.with_columns((pl.col('down_time') / 1000).alias('down_time_sec'))
        logs_binned = logs_binned.with_columns(((pl.col('down_time_sec') // time_agg) * time_agg).alias('time_bin'))

        grp_binned = logs_binned.group_by(['id', 'time_bin']).agg(pl.max('word_count'),
                                                                  pl.count('event_id'),
                                                                  pl.max('cursor_position'))
        grp_binned = grp_binned.with_columns(pl.col('time_bin').cast(pl.Float64))
        grp_binned = grp_binned.sort([pl.col('id'), pl.col('time_bin')])

        # get max down_time value from logs
        max_logs = logs.clone()
        max_down_time = max_logs.group_by(['id']).agg(pl.max('down_time') / 1000)
        # max_down_time = max_down_time.with_columns([pl.col('down_time').cast(pl.Int64)])

        padding_dataframes = []
        max_down_time = max_down_time.collect()

        # Iterate over each row in max_down_time_df
        for row in max_down_time.rows():
            id_value, max_time_value = row[0], row[1]  # Access by index
            time_steps = list(np.arange(0, max_time_value + time_agg, time_agg))

            # Create padding DataFrame with the correct types
            padding_df = pl.DataFrame({
                'id': [str(id_value)] * len(time_steps),
                'time_bin': time_steps
            })

            padding_dataframes.append(padding_df)

        pad_df = pl.concat(padding_dataframes).lazy()
        grp_df = pad_df.join(grp_binned.lazy(), on=['id', 'time_bin'], how='left')
        grp_df = grp_df.sort([pl.col('id'), pl.col('time_bin')])
        grp_df = grp_df.with_columns(pl.col(['word_count','event_id','cursor_position']).fill_null(strategy="forward").over('id'))
        data.append(grp_df)

    return data[0], data[1]

def countvectorize_one_one(train_essays, test_essays):
    print("< Count vectorize one-grams >")
    
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(1, 1))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,:8]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_one{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_one_two(train_essays, test_essays):
    print("< Count vectorize one-grams >")
    
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(1, 1))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,9:16]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_two{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_one_three(train_essays, test_essays):
    print("< Count vectorize one-grams >")
    
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(1, 1))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,17:24]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_three{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_one_four(train_essays, test_essays):
    print("< Count vectorize one-grams >")
    
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(1, 1))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,25:]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_four{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_two_one(train_essays, test_essays):
    print("< Count vectorize bi-grams >")
    data = []
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(2, 2))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,:4]
    toks_df = pd.DataFrame(columns = [f'bigram_tok_one{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_two_one(train_essays, test_essays):
    print("< Count vectorize bi-grams >")
    data = []
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(2, 2))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,5:8]
    toks_df = pd.DataFrame(columns = [f'bigram_tok_two{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def countvectorize_two_two(train_essays, test_essays):
    print("< Count vectorize bi-grams >")
    data = []
    tr_len = train_essays.shape[0]
    ids = pd.concat([train_essays.id, test_essays.id], axis=0).reset_index(drop=True)
    essays = pd.concat([train_essays, test_essays], axis=0).reset_index(drop=True)
    c_vect = CountVectorizer(ngram_range=(2, 2))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks = toks[:,9:12]
    toks_df = pd.DataFrame(columns = [f'bigram_tok_three{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

def down_events_counts_one(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .slice(offset=0, length=10).collect()
            .select('down_event')
            .to_series().to_list())

    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    event_stats = (data
                    .filter(pl.col('down_event').is_in(events))
                    .group_by(['id', 'down_event'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='down_event')
                    ).fill_null(0).lazy()
    
    # Rename columns to a generic format
    cols = event_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        event_stats = event_stats.rename({col: f'down_event_one_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats, ts_feats

def down_events_counts_two(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .slice(offset=10, length=10).collect()
            .select('down_event')
            .to_series().to_list())

    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    event_stats = (data
                    .filter(pl.col('down_event').is_in(events))
                    .group_by(['id', 'down_event'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='down_event')
                    ).fill_null(0).lazy()
    
    # Rename columns to a generic format
    cols = event_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        event_stats = event_stats.rename({col: f'down_event_two_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats, ts_feats

def down_events_counts_three(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .slice(offset=20, length=10).collect()
            .select('down_event')
            .to_series().to_list())

    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    event_stats = (data
                    .filter(pl.col('down_event').is_in(events))
                    .group_by(['id', 'down_event'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='down_event')
                    ).fill_null(0).lazy()
    
    # Rename columns to a generic format
    cols = event_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        event_stats = event_stats.rename({col: f'down_event_three_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats, ts_feats

def down_events_counts_four(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .slice(offset=30, length=10).collect()
            .select('down_event')
            .to_series().to_list())

    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    event_stats = (data
                    .filter(pl.col('down_event').is_in(events))
                    .group_by(['id', 'down_event'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='down_event')
                    ).fill_null(0).lazy()
    
    # Rename columns to a generic format
    cols = event_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        event_stats = event_stats.rename({col: f'down_event_four_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats, ts_feats

def create_pauses(train_logs, test_logs):

    print("< Idle time features >")
    feats = []
    for logs in [train_logs, test_logs]:
        df = logs.clone()
        temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        temp = temp.group_by("id").agg(inter_key_largest_lantency = pl.max('time_diff'),
                                        inter_key_median_lantency = pl.median('time_diff'),
                                        mean_pause_time = pl.mean('time_diff'),
                                        std_pause_time = pl.std('time_diff'),
                                        total_pause_time = pl.sum('time_diff'),
                                        pauses_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 0.5) & (pl.col('time_diff') < 1)).count(),
                                        pauses_1_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1) & (pl.col('time_diff') < 1.5)).count(),
                                        pauses_1_half_sec = pl.col('time_diff').filter((pl.col('time_diff') > 1.5) & (pl.col('time_diff') < 2)).count(),
                                        pauses_2_sec = pl.col('time_diff').filter((pl.col('time_diff') > 2) & (pl.col('time_diff') < 3)).count(),
                                        pauses_3_sec = pl.col('time_diff').filter(pl.col('time_diff') > 3).count(),)
        feats.append(temp)
    return feats[0], feats[1]

def essay_sents_per_par_basic(df):
    AGGREGATIONS = ['count', 'mean', 'std', 'median']
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['sent_per_par'] = df['paragraph'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent_per_par')
    df['sent_per_par'] = df['sent_per_par'].apply(lambda x: x.replace('\n','').strip())
    df = df.groupby(['id','paragraph'])['sent_per_par'].count().reset_index()
    df = df[df['paragraph'].str.strip() != ''].drop('paragraph', axis=1)

    par_sent_df = df[['id','sent_per_par']].groupby(['id']).agg(AGGREGATIONS)
    par_sent_df.columns = ['_'.join(x) for x in par_sent_df.columns]
    par_sent_df['id'] = par_sent_df.index
    par_sent_df = par_sent_df.reset_index(drop=True)
    par_sent_df = par_sent_df.rename(columns={"paragraph_len_count":"paragraph_count"})

    return par_sent_df

def essay_sents_per_par_adv(df):
    AGGREGATIONS = ['max','min', 'first', 'last', q1, q3]
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['sent_per_par'] = df['paragraph'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent_per_par')
    df['sent_per_par'] = df['sent_per_par'].apply(lambda x: x.replace('\n','').strip())
    df = df.groupby(['id','paragraph'])['sent_per_par'].count().reset_index()
    df = df[df['paragraph'].str.strip() != ''].drop('paragraph', axis=1)

    par_sent_df = df[['id','sent_per_par']].groupby(['id']).agg(AGGREGATIONS)
    par_sent_df.columns = ['_'.join(x) for x in par_sent_df.columns]
    par_sent_df['id'] = par_sent_df.index
    par_sent_df = par_sent_df.reset_index(drop=True)
    par_sent_df = par_sent_df.rename(columns={"paragraph_len_count":"paragraph_count"})

    return par_sent_df

def cursor_pos_acceleration_basic(train_logs, test_logs, time_agg=8):
    print("< cursor position acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('cursor_position').diff().over('id').fill_null(0).alias('cursor_position_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('cursor_position_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            cursor_pos_acc_zero = pl.col('acceleration').filter(pl.col('acceleration') == 0).count(),
            cursor_pos_acc_pst = pl.col('acceleration').filter(pl.col('acceleration') > 0).count(),
            cursor_pos_acc_neg = pl.col('acceleration').filter(pl.col('acceleration') < 0).count(),
            cursor_pos_acc_sum = pl.col('acceleration').sum(),
            cursor_pos_acc_mean = pl.col('acceleration').mean(),
            cursor_pos_acc_std = pl.col('acceleration').std(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def cursor_pos_acceleration_adv(train_logs, test_logs, time_agg=8):
    print("< cursor position acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('cursor_position').diff().over('id').fill_null(0).alias('cursor_position_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('cursor_position_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            cursor_pos_acc_max = pl.col('acceleration').max(),
            cursor_pos_acc_q1 = pl.col('acceleration').quantile(0.25),
            cursor_pos_acc_median = pl.col('acceleration').median(),
            cursor_pos_acc_q3 = pl.col('acceleration').quantile(0.75),
            cursor_pos_acc_kurt = pl.col('acceleration').kurtosis(),
            cursor_pos_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def p_burst_feats_basic(train_logs, test_logs, time_agg=2):
    print("< P-burst features >")    
    feats=[]
    original_test_ids = test_logs.select('id').unique()  
    for logs in [train_logs, test_logs]:
        df=logs.clone()

        temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        # temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        temp = temp.filter(pl.col('activity').is_in(['Input']))

        temp = temp.with_columns(pl.col('time_diff')< time_agg)

        rle_grp = temp.with_columns(
            id_runs = pl.struct('time_diff','id').
            rle_id()).filter(pl.col('time_diff'))

        p_burst = rle_grp.group_by(['id','id_runs']).count()

        p_burst = p_burst.group_by(['id']).agg(
            p_burst_count = pl.col('count').count(),
            p_burst_mean = pl.col('count').mean(),
            p_burst_sum = pl.col('count').sum(),
            p_burst_std = pl.col('count').std(),
            p_burst_median = pl.col('count').median(),
        )
        feats.append(p_burst)

    # Check if the second dataframe (test_logs) is empty and fill with zeros if so
    if feats[1].collect().height == 0:
        zero_filled_df = original_test_ids.with_columns([pl.lit(0).alias(col) for col in feats[0].columns if col != 'id'])
        feats[1] = zero_filled_df

    [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)
            
    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)

    return feats[0], feats[1]

def p_burst_feats_adv(train_logs, test_logs, time_agg=2):
    print("< P-burst features >")    
    feats=[]
    original_test_ids = test_logs.select('id').unique()  
    for logs in [train_logs, test_logs]:
        df=logs.clone()

        temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        # temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        temp = temp.filter(pl.col('activity').is_in(['Input']))

        temp = temp.with_columns(pl.col('time_diff')< time_agg)

        rle_grp = temp.with_columns(
            id_runs = pl.struct('time_diff','id').
            rle_id()).filter(pl.col('time_diff'))

        p_burst = rle_grp.group_by(['id','id_runs']).count()

        p_burst = p_burst.group_by(['id']).agg(
            p_burst_max = pl.col('count').max(),
            p_burst_min = pl.col('count').min(),
            burst_skew = pl.col('count').skew(),
            burst_kurt = pl.col('count').kurtosis(),
            burst_q1 = pl.col('count').quantile(0.25),
            burst_q3 = pl.col('count').quantile(0.75),

        )
        feats.append(p_burst)

    # Check if the second dataframe (test_logs) is empty and fill with zeros if so
    if feats[1].collect().height == 0:
        zero_filled_df = original_test_ids.with_columns([pl.lit(0).alias(col) for col in feats[0].columns if col != 'id'])
        feats[1] = zero_filled_df

    [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)
            
    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)

    return feats[0], feats[1]

def r_burst_feats_basic(train_logs, test_logs):
    print("< R-burst features >")    
    feats = []
    original_test_ids = test_logs.select('id').unique()  

    for logs in [train_logs, test_logs]:
        df = logs.clone()
        temp = df.with_columns(pl.col('activity').is_in(['Remove/Cut']))
        rle_grp = temp.with_columns(
            id_runs = pl.struct('activity', 'id').rle_id()
        ).filter(pl.col('activity'))

        r_burst = rle_grp.group_by(['id', 'id_runs']).count()
        r_burst = r_burst.group_by(['id']).agg(
            r_burst_count = pl.col('count').count(),
            r_burst_mean = pl.col('count').mean(),
            r_burst_sum = pl.col('count').sum(),
            r_burst_std = pl.col('count').std(),
            r_burst_median = pl.col('count').median(),
        )
        feats.append(r_burst)

    # Check if the second dataframe (test_logs) is empty and fill with zeros if so
    if feats[1].collect().height == 0:
        zero_filled_df = original_test_ids.with_columns([pl.lit(0).alias(col) for col in feats[0].columns if col != 'id'])
        feats[1] = zero_filled_df

    [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)
            
    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)

    return feats[0], feats[1]

def r_burst_feats_adv(train_logs, test_logs):
    print("< R-burst features >")    
    feats = []
    original_test_ids = test_logs.select('id').unique()  

    for logs in [train_logs, test_logs]:
        df = logs.clone()
        temp = df.with_columns(pl.col('activity').is_in(['Remove/Cut']))
        rle_grp = temp.with_columns(
            id_runs = pl.struct('activity', 'id').rle_id()
        ).filter(pl.col('activity'))

        r_burst = rle_grp.group_by(['id', 'id_runs']).count()
        r_burst = r_burst.group_by(['id']).agg(
            r_burst_max = pl.col('count').max(),
            r_burst_min = pl.col('count').min(),
            r_burst_skew = pl.col('count').skew(),
            r_burst_kurt = pl.col('count').kurtosis(),
            r_burst_q1 = pl.col('count').quantile(0.25),
            r_burst_q3 = pl.col('count').quantile(0.75),
        )
        feats.append(r_burst)

    # Check if the second dataframe (test_logs) is empty and fill with zeros if so
    if feats[1].collect().height == 0:
        zero_filled_df = original_test_ids.with_columns([pl.lit(0).alias(col) for col in feats[0].columns if col != 'id'])
        feats[1] = zero_filled_df

    [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)
            
    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)

    return feats[0], feats[1]

def events_counts_acceleration_basic(train_logs, test_logs, time_agg=4):
    print("< events counts acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('event_id').diff().over('id').fill_null(0).alias('event_id_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('event_id_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            evid_acc_zero = pl.col('acceleration').filter(pl.col('acceleration') == 0).count(),
            evid_acc_pst = pl.col('acceleration').filter(pl.col('acceleration') > 0).count(),
            evid_acc_neg = pl.col('acceleration').filter(pl.col('acceleration') < 0).count(),
            evid_acc_sum = pl.col('acceleration').sum(),
            evid_acc_mean = pl.col('acceleration').mean(),
            evid_acc_std = pl.col('acceleration').std(),
            evid_acc_median = pl.col('acceleration').median(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def events_counts_acceleration_adv(train_logs, test_logs, time_agg=4):
    print("< events counts acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('event_id').diff().over('id').fill_null(0).alias('event_id_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('event_id_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            evid_acc_max = pl.col('acceleration').max(),
            evid_acc_q1 = pl.col('acceleration').quantile(0.25),
            evid_acc_q3 = pl.col('acceleration').quantile(0.75),
            evid_acc_kurt = pl.col('acceleration').kurtosis(),
            evid_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def word_count_acceleration_basic(train_logs, test_logs, time_agg=8):
    print("< word count acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('word_count').diff().over('id').fill_null(0).alias('word_count_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('word_count_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            word_count_acc_zero = pl.col('acceleration').filter(pl.col('acceleration') == 0).count(),
            word_count_acc_pst = pl.col('acceleration').filter(pl.col('acceleration') > 0).count(),
            word_count_acc_neg = pl.col('acceleration').filter(pl.col('acceleration') < 0).count(),
            word_count_acc_sum = pl.col('acceleration').sum(),
            word_count_acc_mean = pl.col('acceleration').mean(),
            word_count_acc_median = pl.col('acceleration').median(),
            word_count_acc_std = pl.col('acceleration').std(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def word_count_acceleration_adv(train_logs, test_logs, time_agg=8):
    print("< word count acceleration >")    

    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for logs in [tr_pad, ts_pad]:

        grp_df = logs.clone()
        grp_df = grp_df.sort(['id', 'time_bin'])

        grp_df = grp_df.with_columns([
            pl.col('word_count').diff().over('id').fill_null(0).alias('word_count_diff'),
            pl.col('time_bin').diff().over('id').fill_null(0).alias('time_bin_diff'),
        ])

        grp_df = grp_df.with_columns(
            (pl.col('word_count_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')
        )

        grp_df = grp_df.with_columns(
            pl.col('rate_of_change').diff().over('id').fill_nan(0).alias('rate_of_change_diff')
        )

        grp_df = grp_df.with_columns(
            (pl.col('rate_of_change_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('acceleration')
        )
        grp_df = grp_df.select(pl.col(['id', 'acceleration']))

        stats = grp_df.group_by('id').agg(
            word_count_acc_max = pl.col('acceleration').max(),
            word_count_acc_q1 = pl.col('acceleration').quantile(0.25),
            word_count_acc_q3 = pl.col('acceleration').quantile(0.75),
            word_count_acc_kurt = pl.col('acceleration').kurtosis(),
            word_count_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def essay_sent_words(df):
    AGGREGATIONS = ['count', 'mean', 'max', 'first', q1, 'median', q3, 'sum']
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))

    sent_agg_df = df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    return sent_agg_df

def essay_sent_length(df):
    AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

    print("< Essays sentences feats >")    
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def essay_par_length(df):
    AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

    print("< Essays paragraphs feats >")    
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS)
                                 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def essay_par_words(df):
    AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
    print("< Essays paragraphs feats >")    
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    
    paragraph_agg_df = df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def essay_sents_per_par(df):
    AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3]
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df['sent_per_par'] = df['paragraph'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent_per_par')
    df['sent_per_par'] = df['sent_per_par'].apply(lambda x: x.replace('\n','').strip())
    df = df.groupby(['id','paragraph'])['sent_per_par'].count().reset_index()
    df = df[df['paragraph'].str.strip() != ''].drop('paragraph', axis=1)

    par_sent_df = df[['id','sent_per_par']].groupby(['id']).agg(AGGREGATIONS)
    par_sent_df.columns = ['_'.join(x) for x in par_sent_df.columns]
    par_sent_df['id'] = par_sent_df.index
    par_sent_df = par_sent_df.reset_index(drop=True)
    par_sent_df = par_sent_df.rename(columns={"paragraph_len_count":"paragraph_count"})

    return par_sent_df

def add_word_pauses_basic(train_logs, test_logs):
    print("< added words pauses basic")    
    feats = []

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)

    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(pl.col('word_count')
                    .diff()
                    .over('id')
                    .fill_null(1)
                    .alias('word_diff'))

        logs = logs.with_columns(pl.col('down_time')
                    .diff()
                    .over('id')
                    .fill_null(0)
                    .alias('down_time_diff')) 

        word_pause = logs.filter(pl.col('word_diff')>0)
        word_pause = word_pause.group_by(['id']).agg(
                add_words_pause_count = pl.col('down_time_diff').count(),
                add_words_pause_mean = pl.col('down_time_diff').mean(),
                add_words_pause_sum = pl.col('down_time_diff').sum(),
                add_words_pause_std = pl.col('down_time_diff').std(),
                add_words_pause_median = pl.col('down_time_diff').median(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]


def add_word_pauses_adv(train_logs, test_logs):
    print("< added words pauses advanced")    
    feats = []

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)

    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(pl.col('word_count')
                    .diff()
                    .over('id')
                    .fill_null(1)
                    .alias('word_diff'))

        logs = logs.with_columns(pl.col('down_time')
                    .diff()
                    .over('id')
                    .fill_null(0)
                    .alias('down_time_diff')) 

        word_pause = logs.filter(pl.col('word_diff')>0)
        word_pause = word_pause.group_by(['id']).agg(
                add_words_pause_max = pl.col('down_time_diff').max(),
                add_words_pause_q1 = pl.col('down_time_diff').quantile(0.25),
                add_words_pause_q3 = pl.col('down_time_diff').quantile(0.75),
                add_words_pause_kurt = pl.col('down_time_diff').kurtosis(),
                add_words_pause_skew = pl.col('down_time_diff').skew(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]

def remove_word_pauses_basic(train_logs, test_logs):
    print("< removed words pauses basic")    
    feats = []

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)

    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(pl.col('word_count')
                    .diff()
                    .over('id')
                    .fill_null(1)
                    .alias('word_diff'))

        logs = logs.with_columns(pl.col('down_time')
                    .diff()
                    .over('id')
                    .fill_null(0)
                    .alias('down_time_diff')) 

        word_pause = logs.filter(pl.col('word_diff')<0)
        word_pause = word_pause.group_by(['id']).agg(
                rmv_words_pause_count = pl.col('down_time_diff').count(),
                rmv_words_pause_mean = pl.col('down_time_diff').mean(),
                rmv_words_pause_sum = pl.col('down_time_diff').sum(),
                rmv_words_pause_std = pl.col('down_time_diff').std(),
                rmv_words_pause_median = pl.col('down_time_diff').median(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]


def remove_word_pauses_adv(train_logs, test_logs):
    print("< removed words pauses advanced")    
    feats = []

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)

    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(pl.col('word_count')
                    .diff()
                    .over('id')
                    .fill_null(1)
                    .alias('word_diff'))

        logs = logs.with_columns(pl.col('down_time')
                    .diff()
                    .over('id')
                    .fill_null(0)
                    .alias('down_time_diff')) 

        word_pause = logs.filter(pl.col('word_diff')<0)
        word_pause = word_pause.group_by(['id']).agg(
                rmv_words_pause_max = pl.col('down_time_diff').max(),
                rmv_words_pause_q1 = pl.col('down_time_diff').quantile(0.25),
                rmv_words_pause_q3 = pl.col('down_time_diff').quantile(0.75),
                rmv_words_pause_kurt = pl.col('down_time_diff').kurtosis(),
                rmv_words_pause_skew = pl.col('down_time_diff').skew(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]

def word_timings_basic(train_logs, test_logs):
    print("< word timings advanced")    
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    for data in [tr_logs, ts_logs]:

        logs = data.clone()
        logs = logs.sort(['id', 'event_id'])
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(
            pl.cum_sum('action_time')
            .over(['id','word_count'])
            .alias('cum_sum_action_time_per_word')
            )

        logs = logs.group_by(['id','word_count']).agg(
            pl.max('cum_sum_action_time_per_word')
            .alias('time_per_word'))

        word_timings = logs.group_by(['id']).agg(
            word_timings_mean = pl.col('time_per_word').mean(),
            word_timings_sum = pl.col('time_per_word').sum(),
            word_timings_std = pl.col('time_per_word').std(),
            word_timings_median = pl.col('time_per_word').median(),
        )
        feats.append(word_timings)
    return feats[0], feats[1]

def word_timings_adv(train_logs, test_logs):
    print("< word timings advanced")    
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    for data in [tr_logs, ts_logs]:

        logs = data.clone()
        logs = logs.sort(['id', 'event_id'])
        logs = logs.select(pl.col(['id','event_id','word_count','down_time','up_time','action_time']))
        logs = logs.with_columns(
            pl.cum_sum('action_time')
            .over(['id','word_count'])
            .alias('cum_sum_action_time_per_word')
            )

        logs = logs.group_by(['id','word_count']).agg(
            pl.max('cum_sum_action_time_per_word')
            .alias('time_per_word'))

        word_timings = logs.group_by(['id']).agg(
            words_timings_max = pl.col('time_per_word').max(),
            words_timings_q1 = pl.col('time_per_word').quantile(0.25),
            words_timings_q3 = pl.col('time_per_word').quantile(0.75),
            words_timings_kurt = pl.col('time_per_word').kurtosis(),
            words_timings_skew = pl.col('time_per_word').skew(),
        )
        feats.append(word_timings)
    return feats[0], feats[1]

def sentences_timing(train_logs, test_logs):
    print("< sentences timing >")    
    feats = []
    for data in [train_logs, test_logs]:
        
        logs = data.clone()
        logs = logs.select(
            pl.col(['id','event_id','down_event','action_time'])).sort('id','event_id')
            
        logs = logs.with_columns(
            pl.when(pl.col('down_event')==".")
            .then(0)
            .when(pl.col('down_event')=="Backspace")
            .then(-1)
            .otherwise(1)
            .alias('removed_sent_interm')
        )

        logs = logs.with_columns((pl.col('down_event') == '.').cum_sum().alias('sentence_number'))
        logs = logs.with_columns(pl.col('down_event').is_in(['.','?','!']).alias('is_sent'))
        logs = logs.with_columns(pl.col('removed_sent_interm').cum_sum().over('id','sentence_number'))

        # FIND REMOVED "." WITH CONSECUTIVE BACKSPACES > removed_sent_interm will be neg
        removed_stops = logs.groupby('id','sentence_number').agg(
            (pl.col('removed_sent_interm') < 0)
            .any()
            .alias('has_negative')
        )

        logs = logs.join(removed_stops, on=('id', 'sentence_number'), how='left')

        logs = logs.with_columns(
            pl.when(pl.col('has_negative') & (pl.col('is_sent')))
            .then(False)
            .otherwise(pl.col('is_sent'))
            .alias('is_sent')
        )

        logs = logs.drop('has_negative')
        # RE-STABLISH SENTENCES STARTING POINT
        logs = logs.with_columns((pl.col('is_sent')).cum_sum().alias('sentence_number'))

        logs = logs.with_columns(pl.col('sentence_number').shift(1))
        logs = logs.with_columns(
                    sent_time = pl.cum_sum('action_time').over('id','sentence_number').fill_null(0)
                )

        sentences = logs.group_by('id','sentence_number').agg(
            pl.max('sent_time')
            .alias('total_sentence_time')
        ).sort('id','sentence_number')

        sentences = sentences.group_by(['id']).agg(
                        sent_timings_mean = pl.col('total_sentence_time').mean(),
                        sent_timings_sum = pl.col('total_sentence_time').sum(),
                        sent_timings_std = pl.col('total_sentence_time').std(),
                        sent_timings_max = pl.col('total_sentence_time').max(),
                        sent_timings_min = pl.col('total_sentence_time').min(),
                        sent_timings_median = pl.col('total_sentence_time').median(),
                        sent_timingse_q1 = pl.col('total_sentence_time').quantile(0.25),
                        sent_timingse_q3 = pl.col('total_sentence_time').quantile(0.75),
                        sent_timingse_kurt = pl.col('total_sentence_time').kurtosis(),
                        sent_timingse_skew = pl.col('total_sentence_time').skew(),
        )
        feats.append(sentences)

    return feats[0], feats[1]

def categorical_nunique(train_logs, test_logs):
    print("< Categorical # unique values features >")    
    feats = []

    for logs in [train_logs, test_logs]:
        data = logs.clone()
        temp  = data.group_by("id").agg(
            pl.n_unique(['activity', 'down_event', 'text_change'])
            .name.suffix('_nunique'))
        feats.append(temp)
    
    return feats[0], feats[1]

added_feats_list = ['train_down_events_counts.pkl', 'train_vector_one_gram.pkl', 
                    'train_create_pauses.pkl', 'train_sentences_per_paragraph.pkl', 
                    'train_add_word_pauses_basic.pkl', 'train_cursor_pos_acceleration.pkl', 
                    'train_remove_word_pauses_adv.pkl', 'train_paragraph_length.pkl']