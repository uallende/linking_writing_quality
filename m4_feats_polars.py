import pandas as pd
import polars as pl
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from m4_feats_functions import getEssays
from scipy.stats import skew, kurtosis

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
        data.append(toks_df.fillna(0))

    return data[0], data[1]

#POLARS
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
        grp_df = grp_df.with_columns(pl.col(['word_count','event_id']).fill_null(strategy="forward").over('id'))
        data.append(grp_df)

    return data[0], data[1]

def input_text_change_feats(train_logs, test_logs):

    print("< Input text change features >") # character level - not word level as opposed to essays feats
    feats = []
    for data in [train_logs, test_logs]:
        df = data.clone()
        temp = df.filter((~pl.col('text_change').str.contains('=>')) & (pl.col('text_change') != 'NoChange'))
        temp = temp.group_by('id').agg(pl.col('text_change').str.concat('').str.extract_all(r'q+'))
        temp = temp.with_columns(
                                    input_text_count = pl.col('text_change').list.len(),
                                    input_text_len_mean = pl.col('text_change').map_elements(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0)),
                                    input_text_len_max = pl.col('text_change').map_elements(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0)),
                                    input_text_len_std = pl.col('text_change').map_elements(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0)),
                                    input_text_len_median = pl.col('text_change').map_elements(lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0)),
                                    input_text_len_skew = pl.col('text_change').map_elements(lambda x: skew([len(i) for i in x] if len(x) > 0 else 0)))
        temp = temp.drop('text_change')
        feats.append(temp)

    feats = [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)

    for col in missing_cols:
        nan_series = pl.repeat(np.nan, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(nan_series)

    return feats[0].lazy(), feats[1].lazy()

def count_of_activities(train_logs, test_logs):
    def count_by_values(df, colname, values):
        fts = df.select(pl.col('id').unique(maintain_order=True))
        for i, value in enumerate(values):
            tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
            fts  = fts.join(tmp_df, on='id', how='left') 
        return fts

    feats = []
    activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']

    for logs in [train_logs, test_logs]:
        counts = logs.clone()
        counts = count_by_values(counts, 'activity', activities)
        feats.append(counts)

    return feats[0], feats[1]

def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts

def events_counts(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    feats = []
    events = (train_logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .head(n_events).collect()
            .select('down_event')
            .to_series().to_list())
    
    for logs in [train_logs, test_logs]:
        data = logs.clone()
        event_stats = (data
                       .filter(pl.col('down_event').is_in(events))
                       .group_by(['id', 'down_event'])
                       .agg(pl.count()).collect()
                       .pivot(values='count', index='id', columns='down_event')
                      )

        # Rename columns to a generic format
        cols = event_stats.columns[1:]  # Skip the 'id' column
        for i, col in enumerate(cols):
            event_stats = event_stats.rename({col: f'event_{i+1}'})

        feats.append(event_stats.fill_null(0).lazy())

    # Ensure that feats are evaluated LazyFrames
    feats = [feat.collect() for feat in feats]
    missing_cols = set(feats[0].columns) - set(feats[1].columns)

    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)

    return feats[0].lazy(), feats[1].lazy()

# POLARS
def rate_of_change_feats(train_logs, test_logs, time_agg=5):
    print("< Word counts rate of change features >")
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for data in [tr_pad, ts_pad]:
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
            pl.col('rate_of_change').filter(pl.col('rate_of_change') == 0).count().alias('roc_zro_count'),
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

def words_stats_feats(train_logs, test_logs, time_agg=12):
    print("< Count of events feats >")
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for data in [tr_pad, ts_pad]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            word_count_sum = pl.col('word_count').sum(),
            word_count_mean = pl.col('word_count').mean(),
            word_count_std = pl.col('word_count').std(),
            word_count_max = pl.col('word_count').max(),
            word_count_q1 = pl.col('word_count').quantile(0.25),
            word_count_median = pl.col('word_count').median(),
            word_count_q3 = pl.col('word_count').quantile(0.75),
            word_count_kurt = pl.col('word_count').kurtosis(),
            word_count_skew = pl.col('word_count').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def events_stats_feats(train_logs, test_logs, time_agg=5):
    print("< Count of events feats >")
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for data in [tr_pad, ts_pad]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            eid_stats_sum = pl.col('event_id').sum(),
            eid_stats_mean = pl.col('event_id').mean(),
            eid_stats_std = pl.col('event_id').std(),
            eid_stats_max = pl.col('event_id').max(),
            eid_stats_q1 = pl.col('event_id').quantile(0.25),
            eid_stats_median = pl.col('event_id').median(),
            eid_stats_q3 = pl.col('event_id').quantile(0.75),
            eid_stats_kurt = pl.col('event_id').kurtosis(),
            eid_stats_skew = pl.col('event_id').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def action_time_feats(train_logs, test_logs):
    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            action_time_mean = pl.col('action_time').mean(),
            action_time_std = pl.col('action_time').std(),
            action_time_max = pl.col('action_time').max(),
           # action_time_q1 = pl.col('action_time').quantile(0.25),
           # action_time_median = pl.col('action_time').median(),
           # action_time_q3 = pl.col('action_time').quantile(0.75),
           # action_time_kurt = pl.col('action_time').kurtosis(),
           # action_time_skew = pl.col('action_time').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def cursor_stats_feats(train_logs, test_logs):
    print("< Cursor changes features >")
    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            action_time_mean = pl.col('cursor_position').mean(),
            action_time_std = pl.col('cursor_position').std(),
            action_time_max = pl.col('cursor_position').max(),
            action_time_q1 = pl.col('cursor_position').quantile(0.25),
            action_time_median = pl.col('cursor_position').median(),
            action_time_q3 = pl.col('cursor_position').quantile(0.75),
            action_time_kurt = pl.col('cursor_position').kurtosis(),
            action_time_skew = pl.col('cursor_position').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def time_based_cursor_stats_feats(train_logs, test_logs, time_agg=30):
    print("< Cursor changes features >")
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)
    feats = []
    for data in [tr_pad, ts_pad]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            cursor_pos_mean = pl.col('cursor_position').mean(),
            cursor_pos_std = pl.col('cursor_position').std(),
            cursor_pos_max = pl.col('cursor_position').max(),
            cursor_pos_q1 = pl.col('cursor_position').quantile(0.25),
            cursor_pos_median = pl.col('cursor_position').median(),
            cursor_pos_q3 = pl.col('cursor_position').quantile(0.75),
            cursor_pos_kurt = pl.col('cursor_position').kurtosis(),
            cursor_pos_skew = pl.col('cursor_position').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def p_burst_feats(train_logs, test_logs):
    print("< P-burst features >")    
    feats=[]
    for logs in [train_logs, test_logs]:
        df=logs.clone()

        temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        temp = temp.with_columns(pl.col('time_diff')<2)

        rle_grp = temp.with_columns(
            id_runs = pl.struct('time_diff','id').
            rle_id()).filter(pl.col('time_diff'))

        p_burst = rle_grp.group_by(['id','id_runs']).count()

        p_burst = p_burst.group_by(['id']).agg(
            burst_count = pl.col('count').count(),
            burst_mean = pl.col('count').mean(),
            burst_sum = pl.col('count').sum(),
            burst_std = pl.col('count').std(),
            burst_max = pl.col('count').max(),
            burst_min = pl.col('count').min(),
            burst_median = pl.col('count').median(),
            # burst_skew = pl.col('count').skew(),
            # burst_kurt = pl.col('count').kurtosis(),
            # burst_q1 = pl.col('count').quantile(0.25),
            # burst_q3 = pl.col('count').quantile(0.75),

        )
        feats.append(p_burst)

    return feats[0], feats[1]

def r_burst_feats(train_logs, test_logs):
    print("< R-burst features >")    
    feats=[]
    for logs in [train_logs, test_logs]:
        df=logs.clone()
        temp = df.with_columns(pl.col('activity').is_in(['Remove/Cut']))
        rle_grp = temp.with_columns(
            id_runs = pl.struct('activity','id').
            rle_id()).filter(pl.col('activity'))

        r_burst = rle_grp.group_by(['id','id_runs']).count()
        r_burst = r_burst.group_by(['id']).agg(
            r_burst_count = pl.col('count').count(),
            r_burst_mean = pl.col('count').mean(),
            r_burst_sum = pl.col('count').sum(),
            r_burst_std = pl.col('count').std(),
            r_burst_max = pl.col('count').max(),
            r_burst_min = pl.col('count').min(),
            r_burst_median = pl.col('count').median(),
            # burst_skew = pl.col('count').skew(),
            # burst_kurt = pl.col('count').kurtosis(),
            # burst_q1 = pl.col('count').quantile(0.25),
            # burst_q3 = pl.col('count').quantile(0.75),
        )
        feats.append(r_burst)
    return feats[0], feats[1]

# POLARS
def rate_of_change_events(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:

        logs = data.clone()
        logs = logs.sort('id')
        logs = logs.with_columns([
            pl.col('event_id').diff().over('id').alias('event_id_diff'),
            pl.col('time_bin').diff().over('id').alias('time_bin_diff')
        ]).fill_nan(0)

        logs = logs.with_columns(
            (pl.col('event_id_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')).fill_nan(0)

        # Aggregating
        stats = logs.group_by('id').agg(
            eid_roc_count_zr = pl.col('rate_of_change').filter(pl.col('rate_of_change') == 0).count(),
            eid_roc_count = pl.col('rate_of_change').count(),
            eid_roc_mean = pl.col('rate_of_change').mean(),
            eid_roc_std = pl.col('rate_of_change').std(),
            eid_roc_max = pl.col('rate_of_change').max(),
            eid_roc_q1 = pl.col('rate_of_change').quantile(0.25),
            eid_roc_median = pl.col('rate_of_change').median(),
            eid_roc_q3 = pl.col('rate_of_change').quantile(0.75),
            eid_roc_kurt = pl.col('rate_of_change').kurtosis(),
            eid_roc_skew = pl.col('rate_of_change').skew(),
        )

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

def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == 'Replace':
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(' To ')
            valueArr = [item.split(', ') for item in splitTxt]
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def get_essay_df(df):
    df       = df[df.activity != 'Nonproduction']
    temp     = df.groupby('id').apply(lambda x: reconstruct_essay(x[['activity', 'cursor_position', 'text_change']]))
    essay_df = pd.DataFrame({'id': df['id'].unique().tolist()})
    essay_df = essay_df.merge(temp.rename('essay'), on='id')
    return essay_df


def word_feats(df):
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]
    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)
    return word_agg_df


def sent_feats(df):
    df['sent'] = df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent')
    df['sent'] = df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    df['sent_len'] = df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    df['sent_word_count'] = df['sent'].apply(lambda x: len(x.split(' ')))
    df = df[df.sent_len!=0].reset_index(drop=True)

    sent_agg_df = pd.concat([df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), 
                             df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df


def parag_feats(df):
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    # Number of characters in paragraphs
    df['paragraph_len'] = df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    df['paragraph_word_count'] = df['paragraph'].apply(lambda x: len(x.split(' ')))
    df = df[df.paragraph_len!=0].reset_index(drop=True)
    
    paragraph_agg_df = pd.concat([df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), 
                                  df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

def product_to_keys(logs, essays):
    essays['product_len'] = essays.essay.str.len()
    tmp_df = logs[logs.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
    essays = essays.merge(tmp_df, on='id', how='left')
    essays['product_to_keys'] = essays['product_len'] / essays['keys_pressed']
    return essays[['id', 'product_to_keys']]

def get_keys_pressed_per_second(logs):
    temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
    temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
    temp_df = temp_df.merge(temp_df_2, on='id', how='left')
    temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
    return temp_df[['id', 'keys_per_second']]

def create_pauses(train_logs, test_logs):

    feats = []
    print("< Idle time features >")

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