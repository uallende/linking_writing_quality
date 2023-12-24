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
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_{i}' for i in range(toks.shape[1])], data=toks)

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
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_{i}' for i in range(toks.shape[1])], data=toks)

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
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_{i}' for i in range(toks.shape[1])], data=toks)

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
    toks = toks[:,25:]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_{i}' for i in range(toks.shape[1])], data=toks)

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
            .slice(offset=0, length=20).collect()
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
        event_stats = event_stats.rename({col: f'down_event_{i+1}'})

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
            .slice(offset=20, length=20).collect()
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
        event_stats = event_stats.rename({col: f'down_event_{i+1}'})

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
            .slice(offset=40, length=20).collect()
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
        event_stats = event_stats.rename({col: f'down_event_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats, ts_feats

def down_events_counts_one(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .slice(offset=60, length=20).collect()
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
        event_stats = event_stats.rename({col: f'down_event_{i+1}'})

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

added_feats_list = ['train_down_events_counts.pkl', 'train_vector_one_gram.pkl', 
                    'train_create_pauses.pkl', 'train_sentences_per_paragraph.pkl', 
                    'train_add_word_pauses_basic.pkl', 'train_cursor_pos_acceleration.pkl', 
                    'train_remove_word_pauses_adv.pkl', 'train_paragraph_length.pkl']