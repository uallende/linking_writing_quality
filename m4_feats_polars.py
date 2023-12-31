import pandas as pd
import polars as pl
import numpy as np
import re
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import skew, kurtosis

pl.set_random_seed(42)

def amend_event_id_order(train_logs, test_logs):
    fixed_logs = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        logs = logs.sort(pl.col(['id', 'down_time']))
        logs = logs.with_columns(pl.col('event_id').cumcount().over('id'))
        fixed_logs.append(logs)

    return fixed_logs[0], fixed_logs[1]

def normalise_up_down_times(train_logs, test_logs):
    new_logs = []
    for logs in [train_logs, test_logs]:
        logs = logs.sort('id','event_id')
        min_down_time = logs.group_by('id').agg(pl.min('down_time').alias('min_down_time'))
        logs = logs.join(min_down_time, on='id', how='left')
        logs = logs.with_columns([(pl.col('down_time') - pl.col('min_down_time')).alias('normalised_down_time')])
        logs = logs.with_columns([(pl.col('normalised_down_time') + pl.col('action_time')).alias('normalised_up_time')])
        logs = logs.drop(['min_down_time', 'down_time', 'up_time'])
        logs = logs.rename({'normalised_down_time': 'down_time', 'normalised_up_time': 'up_time'})
        new_logs.append(logs.sort(['id','event_id'], descending=[False,False]))
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
    toks = toks[:,:16]
    toks_df = pd.DataFrame(columns = [f'one_gram_tok_{i}' for i in range(toks.shape[1])], data=toks)

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
    toks = toks[:,:16]
    toks_df = pd.DataFrame(columns = [f'bigram_tok_{i}' for i in range(toks.shape[1])], data=toks)

    feats = pd.concat([ids, toks_df], axis=1)

    tr_feats = feats.loc[:tr_len-1]
    ts_feats = feats.loc[tr_len:]

    tr_feats = pl.DataFrame(tr_feats).lazy()
    ts_feats = pl.DataFrame(ts_feats).lazy()
    return tr_feats, ts_feats

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

        counts = counts.with_columns((pl.col('activity_0_cnt')/pl.col('activity_1_cnt')).alias('ratio_1_2'))
        counts = counts.with_columns((pl.col('activity_0_cnt')/pl.col('activity_2_cnt')).alias('ratio_1_3'))

        feats.append(counts)

    return feats[0], feats[1]

def count_by_values(df, colname, values):
    fts = df.select(pl.col('id').unique(maintain_order=True))
    for i, value in enumerate(values):
        tmp_df = df.group_by('id').agg(pl.col(colname).is_in([value]).sum().alias(f'{colname}_{i}_cnt'))
        fts  = fts.join(tmp_df, on='id', how='left') 
    return fts

def action_time_by_activity(train_logs, test_logs):
    print("< Action time by activities >")
    feats = []
    activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
    for data in [train_logs, test_logs]:
        logs = data.clone()

        stats = logs.filter(pl.col('activity').is_in(activities)) \
            .group_by(['id', 'activity']) \
            .agg(pl.sum('action_time').name.suffix('_sum')) \
            .collect() \
            .pivot(values='action_time_sum', index='id', columns='activity') \
            .fill_null(0)

        feats.append(stats)

    missing_cols = set(feats[0].columns) - set(feats[1].columns)

    for col in missing_cols:
        zero_series = pl.repeat(0, n=len(feats[1])).alias(col)
        feats[1] = feats[1].with_columns(zero_series)
    return feats[0].lazy(), feats[1].lazy()

def down_events_counts(train_logs, test_logs, n_events=20):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = (logs
            .group_by(['down_event'])
            .agg(pl.count())
            .sort('count', descending=True)
            .head(n_events).collect()
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

def word_count_stats_baseline(train_logs, test_logs):
    print("< word changes stats baseline>")
    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            word_count_baseline_sum = pl.col('word_count').sum(),
            word_count_baseline_mean = pl.col('word_count').mean(),
            word_count_baseline_std = pl.col('word_count').std(),
            word_count_baseline_max = pl.col('word_count').max(),
            word_count_baseline_q1 = pl.col('word_count').quantile(0.25),
            word_count_baseline_median = pl.col('word_count').median(),
            word_count_baseline_q3 = pl.col('word_count').quantile(0.75),
            word_count_baseline_kurt = pl.col('word_count').kurtosis(),
            word_count_baseline_skew = pl.col('word_count').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def word_count_time_based(train_logs, test_logs, time_agg=12):
    print("< word changes stats time based>")
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

def word_counts_rate_of_change(train_logs, test_logs, time_agg=5):
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
            pl.col('rate_of_change').filter(pl.col('rate_of_change') > 0).count().alias('pos_change_count'),
            pl.col('rate_of_change').filter(pl.col('rate_of_change') < 0).count().alias('neg_change_count'),
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

def word_count_acceleration(train_logs, test_logs, time_agg=8):
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
            word_count_acc_std = pl.col('acceleration').std(),
            word_count_acc_max = pl.col('acceleration').max(),
            word_count_acc_q1 = pl.col('acceleration').quantile(0.25),
            word_count_acc_median = pl.col('acceleration').median(),
            word_count_acc_q3 = pl.col('acceleration').quantile(0.75),
            word_count_acc_kurt = pl.col('acceleration').kurtosis(),
            word_count_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def events_counts_time_based(train_logs, test_logs, time_agg=5):
    print("< Count of events time based feats >")
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)

    for data in [tr_pad, ts_pad]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            eid_timeb_sum = pl.col('event_id').sum(),
            eid_timeb_mean = pl.col('event_id').mean(),
            eid_timeb_std = pl.col('event_id').std(),
            eid_timeb_max = pl.col('event_id').max(),
            eid_timeb_q1 = pl.col('event_id').quantile(0.25),
            eid_timeb_median = pl.col('event_id').median(),
            eid_timeb_q3 = pl.col('event_id').quantile(0.75),
            eid_timeb_kurt = pl.col('event_id').kurtosis(),
            eid_timeb_skew = pl.col('event_id').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def events_counts_baseline(train_logs, test_logs):
    print("< Count of events baseline feats >")
    feats = []

    for data in [train_logs, test_logs]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            eid_bline_sum = pl.col('event_id').sum(),
            eid_bline_mean = pl.col('event_id').mean(),
            eid_bline_std = pl.col('event_id').std(),
            eid_bline_max = pl.col('event_id').max(),
            eid_bline_q1 = pl.col('event_id').quantile(0.25),
            eid_bline_median = pl.col('event_id').median(),
            eid_bline_q3 = pl.col('event_id').quantile(0.75),
            eid_bline_kurt = pl.col('event_id').kurtosis(),
            eid_bline_skew = pl.col('event_id').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def events_counts_rate_of_change(train_logs, test_logs, time_agg=3):
    print("< event_id rate of change >")    
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)
    for data in [tr_pad, ts_pad]:

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

def events_counts_acceleration(train_logs, test_logs, time_agg=4):
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
            evid_acc_max = pl.col('acceleration').max(),
            evid_acc_q1 = pl.col('acceleration').quantile(0.25),
            evid_acc_median = pl.col('acceleration').median(),
            evid_acc_q3 = pl.col('acceleration').quantile(0.75),
            evid_acc_kurt = pl.col('acceleration').kurtosis(),
            evid_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def action_time_baseline_stats(train_logs, test_logs):
    print("< Action time baseline stats >")
    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        stats = logs.group_by('id').agg(
            action_time_mean = pl.col('action_time').mean(),
            action_time_std = pl.col('action_time').std(),
            action_time_max = pl.col('action_time').max(),
            action_time_q1 = pl.col('action_time').quantile(0.25),
            action_time_median = pl.col('action_time').median(),
            action_time_q3 = pl.col('action_time').quantile(0.75),
            action_time_kurt = pl.col('action_time').kurtosis(),
            action_time_skew = pl.col('action_time').skew(),
        )
        feats.append(stats)
    return feats[0], feats[1]

def cursor_pos_baseline(train_logs, test_logs):
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

def cursor_pos_time_based(train_logs, test_logs, time_agg=3):
    print("< Cursor changes based on time >")
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

def cursor_pos_rate_of_change(train_logs, test_logs, time_agg=10):
    print("< event_id rate of change >")    
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    tr_pad, ts_pad = down_time_padding(tr_logs, ts_logs, time_agg)
    for data in [tr_pad, ts_pad]:

        logs = data.clone()
        logs = logs.sort('id')
        logs = logs.with_columns([
            pl.col('cursor_position').diff().over('id').alias('cursor_pos_diff'),
            pl.col('time_bin').diff().over('id').alias('time_bin_diff')
        ]).fill_nan(0)

        logs = logs.with_columns(
            (pl.col('cursor_pos_diff') / pl.col('time_bin_diff')).fill_nan(0).alias('rate_of_change')).fill_nan(0)

        # Aggregating
        stats = logs.group_by('id').agg(
            cursor_pos_roc_count_zr = pl.col('rate_of_change').filter(pl.col('rate_of_change') == 0).count(),
            cursor_pos_pst_change_count = pl.col('rate_of_change').filter(pl.col('rate_of_change') > 0).count(),
            cursor_pos_neg_change_count = pl.col('rate_of_change').filter(pl.col('rate_of_change') < 0).count(),
            cursor_pos_roc_count = pl.col('rate_of_change').count(),
            cursor_pos_roc_mean = pl.col('rate_of_change').mean(),
            cursor_pos_roc_std = pl.col('rate_of_change').std(),
            cursor_pos_roc_max = pl.col('rate_of_change').max(),
            cursor_pos_roc_q1 = pl.col('rate_of_change').quantile(0.25),
            cursor_pos_roc_median = pl.col('rate_of_change').median(),
            cursor_pos_roc_q3 = pl.col('rate_of_change').quantile(0.75),
            cursor_pos_roc_kurt = pl.col('rate_of_change').kurtosis(),
            cursor_pos_roc_skew = pl.col('rate_of_change').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def cursor_pos_acceleration(train_logs, test_logs, time_agg=8):
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
            cursor_pos_acc_max = pl.col('acceleration').max(),
            cursor_pos_acc_q1 = pl.col('acceleration').quantile(0.25),
            cursor_pos_acc_median = pl.col('acceleration').median(),
            cursor_pos_acc_q3 = pl.col('acceleration').quantile(0.75),
            cursor_pos_acc_kurt = pl.col('acceleration').kurtosis(),
            cursor_pos_acc_skew = pl.col('acceleration').skew(),
        )

        feats.append(stats)
    return feats[0], feats[1]

def create_integrated_iki(logs):

    logs = logs.with_columns(
        pl.col('action_time').diff()
        .over('id')
        .alias('iki')
        .fill_null(0)
    )
    logs = logs.with_columns(
        pl.col('action_time')
        .mean()
        .over('id')
        .alias('action_time_mean')
    )
    logs = logs.with_columns(
        (pl.col('iki') - pl.col('action_time'))
        .alias('mean_centering')
    )
    logs = logs.with_columns(
        pl.col('mean_centering')
        .cum_sum()
        .over('id')
        .alias('iki_integrated')
    )

    logs = logs.select(pl.col(['id','iki_integrated']))
    return logs


def integrated_iki(train_logs, test_logs):
    print("integrated IKI")    
    feats = []

    for data in [train_logs, test_logs]:
        logs = data.clone()
        logs = create_integrated_iki(logs)

        iki_stats = logs.group_by(['id']).agg(
                        iki_stats_count = pl.col('iki_integrated').count(),
                        iki_stats_mean = pl.col('iki_integrated').mean(),
                        iki_stats_sum = pl.col('iki_integrated').sum(),
                        iki_stats_std = pl.col('iki_integrated').std(),
                        iki_stats_max = pl.col('iki_integrated').max(),
                        iki_stats_min = pl.col('iki_integrated').min(),
                        iki_stats_median = pl.col('iki_integrated').median()
        )
        feats.append(iki_stats)
    return feats[0], feats[1]

def calculate_fluctuations(iki_integrated, q, bin_sizes):
    Fq = np.zeros(len(bin_sizes))
    for i, s in enumerate(bin_sizes):
        segments = int(np.floor(len(iki_integrated) / s))
        rms = np.zeros(segments)
        for v in range(segments):
            segment = iki_integrated[v * s: (v + 1) * s]
            trend = np.polyfit(np.arange(s), segment, 1)  # linear fit (trend)
            detrended = segment - np.polyval(trend, np.arange(s))
            rms[v] = np.sqrt(np.mean(detrended ** 2))
        Fq[i] = (np.mean(rms ** q)) ** (1 / q) if q != 0 else np.exp(0.5 * np.mean(np.log(rms ** 2)))
    return Fq

def mfdfla_for_series(series, q_values, bin_sizes):
    results = []
    for q in q_values:
        Fq_values = calculate_fluctuations(series, q, bin_sizes)
        results.extend(Fq_values)
    return results

def process_group(series, q_values, bin_sizes):
    return mfdfla_for_series(series, q_values, bin_sizes)

def calculate_selected_fluctuations_parallel(iki_integrated_df, q_values, bin_sizes, n_jobs=-1):
    grouped = iki_integrated_df.groupby('id')['iki_integrated']
    results = Parallel(n_jobs=n_jobs)(delayed(process_group)(group, q_values, bin_sizes) for name, group in grouped)
    feats = pd.DataFrame(results, index=[name for name, group in grouped])
    feats.reset_index(inplace=True)
    columns = ['id'] + [f'Fq_q{q}_bin{s}' for q in q_values for s in bin_sizes]
    feats.columns = columns
    return feats

def fractal_stats(train_logs, test_logs):

    feats = []
    q_values = np.linspace(-15, 15, 2)
    bin_sizes = [1500, 2000, 2500]
    for data in [train_logs, test_logs]:
        
        logs = data.clone()
        iki_df = create_integrated_iki(logs)
        iki_df = iki_df.collect().to_pandas()
        stats = calculate_selected_fluctuations_parallel(iki_df, q_values, bin_sizes)
        stats = pl.DataFrame(stats).lazy()
        feats.append(stats)

    return feats[0], feats[1]

def p_burst_feats(train_logs, test_logs, time_agg=2):
    print("< P-burst features >")    
    feats=[]
    original_test_ids = test_logs.select('id').unique()  
    for logs in [train_logs, test_logs]:
        df=logs.clone()

        temp = df.with_columns(pl.col('up_time').shift().over('id').alias('up_time_lagged'))
        temp = temp.with_columns((abs(pl.col('down_time') - pl.col('up_time_lagged')) / 1000).fill_null(0).alias('time_diff'))
        temp = temp.filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
        #temp = temp.filter(pl.col('activity').is_in(['Input']))

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
            p_burst_max = pl.col('count').max(),
            p_burst_min = pl.col('count').min(),
            p_burst_median = pl.col('count').median(),
            p_burst_skew = pl.col('count').skew(),
            p_burst_kurt = pl.col('count').kurtosis(),
            p_burst_q1 = pl.col('count').quantile(0.25),
            p_burst_q3 = pl.col('count').quantile(0.75),

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

def r_burst_feats(train_logs, test_logs):
    print("< R-burst features >")    
    feats = []
    tr_ids = pl.DataFrame({'id': train_logs.select(pl.col('id')).unique().collect().to_series().to_list()})
    ts_ids = pl.DataFrame({'id': test_logs.select(pl.col('id')).unique().collect().to_series().to_list()})


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
            r_burst_max = pl.col('count').max(),
            r_burst_min = pl.col('count').min(),
            r_burst_median = pl.col('count').median()
        )
        feats.append(r_burst.collect())
    feats[0] = tr_ids.join(feats[0], on='id', how='left').fill_null(0)
    feats[1] = ts_ids.join(feats[1], on='id', how='left').fill_null(0)
    return feats[0].lazy(), feats[1].lazy()

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
    print("< Essays word feats >")    
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

def word_long_word_count(df):
    print("< Essays word feats >")    
    essay_df = df
    df['word'] = df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    df = df.explode('word')
    df['word_len'] = df['word'].apply(lambda x: len(x))
    df = df[df['word_len'] != 0]
    word_agg_df = df[['id','word_len']].groupby(['id']).agg(AGGREGATIONS)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    word_agg_df = word_agg_df.reset_index(drop=True)

    for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = df[df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)

    cols = ['id'] + [f'word_len_ge_{word_l}_count' for word_l in [5, 6, 7, 8, 9, 10, 11, 12]]

    return word_agg_df[cols]


def sent_feats(df):
    print("< Essays sentences feats >")    
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
    print("< Essays paragraphs feats >")    
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
    print('< product to keys >')
    feats = []
    for log, essay in zip(logs, essays):
        essay['product_len'] = essay.essay.str.len()
        tmp_df = log[log.activity.isin(['Input', 'Remove/Cut'])].groupby(['id']).agg({'activity': 'count'}).reset_index().rename(columns={'activity': 'keys_pressed'})
        essay = essay.merge(tmp_df, on='id', how='left')
        essay['product_to_keys'] = essay['product_len'] / essay['keys_pressed']
        feats.append(essay[['id', 'product_to_keys']])
        
    tr_feats = pl.DataFrame(feats[0]).lazy()
    ts_feats = pl.DataFrame(feats[1]).lazy()
    return tr_feats, ts_feats
    

def get_keys_pressed_per_second(train_logs, test_logs):
    print('< get keys pressed per second >')
    feats = []
    for data in [train_logs, test_logs]:
        logs = data.copy()
        temp_df = logs[logs['activity'].isin(['Input', 'Remove/Cut'])].groupby(['id']).agg(keys_pressed=('event_id', 'count')).reset_index()
        temp_df_2 = logs.groupby(['id']).agg(min_down_time=('down_time', 'min'), max_up_time=('up_time', 'max')).reset_index()
        temp_df = temp_df.merge(temp_df_2, on='id', how='left')
        temp_df['keys_per_second'] = temp_df['keys_pressed'] / ((temp_df['max_up_time'] - temp_df['min_down_time']) / 1000)
        feats.append(temp_df[['id', 'keys_per_second']])

    tr_feats = pl.DataFrame(feats[0]).lazy()
    ts_feats = pl.DataFrame(feats[1]).lazy()
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

def essay_sents_per_par(df):
    AGGREGATIONS = ['count', 'mean', 'min', 'max', 'first', 'last', q1, 'median', q3]
    df['paragraph'] = df['essay'].apply(lambda x: x.split('\n'))
    df = df.explode('paragraph')
    df = df[df['paragraph'].str.strip() != '']
    df['sent_per_par'] = df['paragraph'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    df = df.explode('sent_per_par')
    df = df[df['sent_per_par'].str.strip() != '']
    df['sent_per_par'] = df['sent_per_par'].apply(lambda x: x.replace('\n','').strip())
    df = df.groupby(['id','paragraph'])['sent_per_par'].count().reset_index()
    df = df[df['paragraph'].str.strip() != ''].drop('paragraph', axis=1)

    par_sent_df = df[['id','sent_per_par']].groupby(['id']).agg(AGGREGATIONS)
    par_sent_df.columns = ['_'.join(x) for x in par_sent_df.columns]
    par_sent_df['id'] = par_sent_df.index
    par_sent_df = par_sent_df.reset_index(drop=True)
    par_sent_df = par_sent_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return par_sent_df

def add_word_pauses(train_logs, test_logs):
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
                    .over('id','word_count')
                    .fill_null(0)
                    .alias('down_time_diff')) 

        word_pause = logs.filter(pl.col('word_diff')>0)
        word_pause = word_pause.group_by(['id']).agg(
                add_words_pause_count = pl.col('down_time_diff').count(),
                add_words_pause_mean = pl.col('down_time_diff').mean(),
                add_words_pause_sum = pl.col('down_time_diff').sum(),
                add_words_pause_std = pl.col('down_time_diff').std(),
                add_words_pause_median = pl.col('down_time_diff').median(),
                add_words_pause_max = pl.col('down_time_diff').max(),
                add_words_pause_q1 = pl.col('down_time_diff').quantile(0.25),
                add_words_pause_q3 = pl.col('down_time_diff').quantile(0.75),
                add_words_pause_kurt = pl.col('down_time_diff').kurtosis(),
                add_words_pause_skew = pl.col('down_time_diff').skew(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]

def remove_word_pauses(train_logs, test_logs):
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
                rmv_words_pause_max = pl.col('down_time_diff').max(),
                rmv_words_pause_q1 = pl.col('down_time_diff').quantile(0.25),
                rmv_words_pause_q3 = pl.col('down_time_diff').quantile(0.75),
                rmv_words_pause_kurt = pl.col('down_time_diff').kurtosis(),
                rmv_words_pause_skew = pl.col('down_time_diff').skew(),
        )
        feats.append(word_pause)
    return feats[0], feats[1]

def word_timings(train_logs, test_logs):
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
            words_timings_max = pl.col('time_per_word').max(),
            words_timings_q1 = pl.col('time_per_word').quantile(0.25),
            words_timings_q3 = pl.col('time_per_word').quantile(0.75),
            words_timings_kurt = pl.col('time_per_word').kurtosis(),
            words_timings_skew = pl.col('time_per_word').skew(),
        )
        feats.append(word_timings)
    return feats[0], feats[1]

def word_wait_shift(train_logs, test_logs, shift=1):
    print('< word_wait_shift >')
    feats = []
    for data in [train_logs,test_logs]:
        logs = data.clone()
        logs = logs.group_by('id','word_count').agg(
            word_start_time = pl.col('down_time').min()).sort('id','word_count')

        logs = logs.with_columns(pl.col('word_start_time').shift(shift).over('id').alias(f'shifted'))
        logs = logs.with_columns((pl.col('word_start_time') - pl.col('shifted')).alias(f'word_time_diff_{shift}'))

        words_shifted = logs.group_by('id').agg(
                        pl.col(f'word_time_diff_{shift}').count().name.suffix('count'),
                        pl.col(f'word_time_diff_{shift}').mean().name.suffix('mean'),
                        pl.col(f'word_time_diff_{shift}').sum().name.suffix('sum'),
                        pl.col(f'word_time_diff_{shift}').std().name.suffix('std'),
                        pl.col(f'word_time_diff_{shift}').median().name.suffix('median'),
                        pl.col(f'word_time_diff_{shift}').max().name.suffix('max'),
                        pl.col(f'word_time_diff_{shift}').quantile(0.25).name.suffix('quantile_25'),
                        pl.col(f'word_time_diff_{shift}').quantile(0.75).name.suffix('quantile_75'),
                        pl.col(f'word_time_diff_{shift}').kurtosis().name.suffix('kurt'),
                        pl.col(f'word_time_diff_{shift}').skew().name.suffix('skew'),
        )
        feats.append(words_shifted)
    return feats[0], feats[1]

def text_changes_counts(train_logs, test_logs):
    print("< text chaanges counts features >")
    text_changes = ['\n', ':', 'NoChange', '/', ' ', ';', '\\', '=']
    logs = pl.concat([train_logs, test_logs], how = 'vertical')

    all_ids = pl.DataFrame({'id': logs.select(pl.col('id')).unique().collect().to_series().to_list()})
    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    text_changes_stats = (data
                    .filter(pl.col('text_change').is_in(text_changes))
                    .group_by(['id', 'text_change'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='text_change')
                    ).fill_null(0)

    text_changes_stats = all_ids.join(text_changes_stats,on='id',how='left')
    text_changes_stats = text_changes_stats.fill_null(0)  

    # Rename columns to a generic format
    cols = text_changes_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        text_changes_stats = text_changes_stats.rename({col: f'text_change{i+1}'})

    tr_feats = text_changes_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = text_changes_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats.lazy(), ts_feats.lazy()


def punctuations(train_logs, test_logs):
    print("< punctuations features >")
    feats = []
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    punctuations = [':', '#', '%', '<', ')', '>', '+', '/', '(', '^', '_', ';', '@', '!', '$', '&', '*']
    all_ids = pl.DataFrame({'id': logs.select(pl.col('id')).unique().collect().to_series().to_list()})
    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    data = logs.clone()
    event_stats = (data
                    .filter(pl.col('down_event').is_in(punctuations))
                    .group_by(['id', 'down_event'])
                    .agg(pl.count()).collect()
                    .pivot(values='count', index='id', columns='down_event')
                    ).fill_null(0)

    event_stats = all_ids.join(event_stats,on='id',how='left')
    event_stats = event_stats.fill_null(0)

    # Rename columns to a generic format
    cols = event_stats.columns[1:]  # Skip the 'id' column
    for i, col in enumerate(cols):
        event_stats = event_stats.rename({col: f'punctuation_{i+1}'})

    tr_feats = event_stats.filter(pl.col('id').is_in(tr_ids))
    ts_feats = event_stats.filter(pl.col('id').is_in(ts_ids))

    return tr_feats.lazy(), ts_feats.lazy()

def remove_words_time_spent(train_logs, test_logs):
    print(f'< remove_words_time_spent >')
    feats = []
    ts_ids = pl.DataFrame({'id': test_logs.select(pl.col('id')).unique().collect().to_series().to_list()}).lazy()
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)

    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.sort(['id', 'event_id'])
        logs = logs.filter(pl.col('activity') == 'Remove/Cut')

        logs = logs.with_columns(
            pl.col('down_time')
            .diff()
            .over('id')
            .fill_null(0)
            .alias('down_time_diff')
        )

        logs = logs.group_by('id', 'word_count').agg(
            remove_time_spent=pl.col('down_time_diff').sum()
        )

        word_r_stats = logs.group_by('id').agg(
            word_r_stats_count=pl.col('remove_time_spent').count(),
            word_r_stats_mean=pl.col('remove_time_spent').mean(),
            word_r_stats_sum=pl.col('remove_time_spent').sum(),
            word_r_stats_std=pl.col('remove_time_spent').std(),
            word_r_stats_max=pl.col('remove_time_spent').max(),
            word_r_stats_median=pl.col('remove_time_spent').median(),
            word_r_stats_skew=pl.col('remove_time_spent').skew(),
            word_r_stats_kurt=pl.col('remove_time_spent').kurtosis(),
            word_r_stats_q3=pl.col('remove_time_spent').quantile(0.75)
        )
        
        feats.append(word_r_stats)

    feats[1] = ts_ids.join(feats[1], on='id', how='left')

    return feats[0].fill_null(0), feats[1].fill_null(0)

def words_duration_stats(train_logs, test_logs):
    print('< words_duration_stats >')
    
    feats = []
    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    for data in [tr_logs, ts_logs]:
        logs = data.clone()
        logs = logs.sort(['id', 'event_id'])
        logs = logs.select(pl.col(['id', 'event_id', 'word_count', 'down_time', 'up_time', 'action_time']))

        logs = logs.with_columns(
            pl.col('down_time')
            .diff()
            .over('id')
            .fill_null(0)
            .alias('down_time_diff')
        )

        logs = logs.with_columns(
            pl.cum_sum('down_time_diff')
            .over(['id', 'word_count'])
            .alias('time_per_word')
        )

        logs = logs.group_by(['id', 'word_count']).agg(
            pl.last('time_per_word')
        ).sort(['id', 'word_count'])

        words_duration = logs.group_by(['id']).agg(
            words_duration_mean=pl.col('time_per_word').mean(),
            words_duration_sum=pl.col('time_per_word').sum(),
            words_duration_std=pl.col('time_per_word').std(),
            words_duration_max=pl.col('time_per_word').max(),
            words_duration_median=pl.col('time_per_word').median(),
            words_duration_skew=pl.col('time_per_word').skew(),
            words_duration_kurt=pl.col('time_per_word').kurtosis(),
            words_duration_q3=pl.col('time_per_word').quantile(0.75)
        )
        feats.append(words_duration)

    return feats[0], feats[1]

def words_p_burst(train_logs, test_logs, time_agg=2500):
    print('< words_p_burst >')

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    logs = pl.concat([tr_logs, ts_logs], how='vertical')

    all_ids = pl.DataFrame({'id': logs.select(pl.col('id')).unique().collect().to_series().to_list()}).lazy()
    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    logs = logs.sort(['id', 'event_id'])
    logs = logs.select(pl.col(['id', 'event_id', 'word_count', 'down_time', 'up_time', 'action_time']))

    logs = logs.with_columns(
        pl.col('down_time')
        .diff()
        .over('id')
        .fill_null(0)
        .alias('down_time_diff')
    )

    logs = logs.with_columns(
        pl.cum_sum('down_time_diff')
        .over(['id', 'word_count'])
        .alias('time_per_word')
    )

    logs = logs.group_by(['id', 'word_count']).agg(
        pl.last('time_per_word')
    ).sort(['id', 'word_count'])

    temp = logs.with_columns(pl.col('time_per_word') < time_agg)

    rle_grp = temp.with_columns(
        id_runs=pl.struct('time_per_word', 'id').rle_id()
    ).filter(pl.col('time_per_word')).sort('id', 'word_count')

    rle_grp = rle_grp.group_by('id', 'id_runs').count().sort('id', 'id_runs')
    rle_grp = rle_grp.filter(pl.col('count') > 1)

    p_burst = rle_grp.group_by(['id']).agg(
        p_burst_count=pl.col('count').count(),
        p_burst_mean=pl.col('count').mean(),
        p_burst_sum=pl.col('count').sum(),
        p_burst_std=pl.col('count').std(),
        p_burst_max=pl.col('count').max(),
        p_burst_median=pl.col('count').median(),
        p_burst_skew=pl.col('count').skew(),
        p_burst_kurt=pl.col('count').kurtosis(),
        p_burst_q3=pl.col('count').quantile(0.75)
    )

    p_burst = all_ids.join(p_burst, on='id', how='left').fill_null(0)

    tr = p_burst.filter(pl.col('id').is_in(tr_ids))
    ts = p_burst.filter(pl.col('id').is_in(ts_ids))
    return tr, ts

def sentences_timings(train_logs, test_logs):
    print(f'< sentences timings >')
    feats = []
    tr_logs,ts_logs = normalise_up_down_times(train_logs,test_logs)
    for data in [tr_logs, ts_logs]:

        logs = data.clone()

        logs = logs.with_columns(
            pl.col('cursor_position')
            .diff()
            .over('id')
            .fill_null(0)
            .alias('cursor_pos_diff'))
        
        logs = logs.with_columns(
            pl.when(pl.col('cursor_pos_diff')< -1)
            .then(True)
            .otherwise(False)
            .alias('cursor_moved_new_sentence'))

        logs = logs.with_columns(
            pl.col('down_event')
            .is_in(['.','?','!']).alias('is_end_of_sentence_mark'))

        logs = logs.with_columns(
            (pl.col('is_end_of_sentence_mark') | (pl.col('cursor_moved_new_sentence')))
            .cum_sum()
            .over('id')
            .shift(1)
            .fill_null(0)
            .alias('sentence_number')
        )
        replace_end_of_sentence = logs.with_columns(
            pl.col('cursor_pos_diff')
            .cum_sum()
            .over('id','sentence_number')
            .alias('calc_is_sent_removed'))

        replace_end_of_sentence = replace_end_of_sentence.with_columns(
            pl.when(pl.col('calc_is_sent_removed') == -1)
            .then(True)
            .otherwise(False)
            .alias('removed_end_of_sentence'))

        replace_end_of_sentence = replace_end_of_sentence.group_by('id','sentence_number').agg([
            pl.col('calc_is_sent_removed').min().alias('min_value')
        ]).with_columns(
            (pl.col('min_value') == -1).alias('is_min_minus_one')
        ).sort('id','sentence_number')

        replace_end_of_sentence = replace_end_of_sentence.with_columns(pl.col('is_min_minus_one').shift(-1))
        logs = logs.join(replace_end_of_sentence, on=('id','sentence_number'), how='left')
        logs = logs.with_columns(
            pl.when('is_min_minus_one')
            .then(not('cursor_moved_new_sentence'))
            .otherwise(pl.col('cursor_moved_new_sentence')).alias('is_end_of_sentence_mark'))

        logs = logs.drop('event_id','up_time','action_time','up_event','text_change','min_value','is_min_minus_one','word_count','sentence_number')

        ### AFTER RECALCULATING SENTENCES

        logs = logs.with_columns(
            (pl.col('is_end_of_sentence_mark'))
            .cum_sum()
            .over('id')
            .shift(1)
            .fill_null(0)
            .alias('sentence_number')
        )

        logs = logs.with_columns(
            pl.col('down_time')
            .diff()
            .over('id','sentence_number')
            .fill_null(0)
            .alias('down_time_diff'))

        logs = logs.with_columns(pl.col('down_time_diff').cum_sum().over('id','sentence_number').fill_null(0).alias('sentence_duration'))

        sent = logs.group_by('id','sentence_number').agg(
            pl.last('sentence_duration')).sort('id','sentence_number')

        sent = sent.group_by(['id']).agg(
                        sent_timings_mean = pl.col('sentence_duration').mean(),
                        sent_timings_sum = pl.col('sentence_duration').sum(),
                        sent_timings_std = pl.col('sentence_duration').std(),
                        sent_timings_max = pl.col('sentence_duration').max(),
                        sent_timings_median = pl.col('sentence_duration').median(),
                        sent_timingse_q1 = pl.col('sentence_duration').quantile(0.25),
                        sent_timingse_q3 = pl.col('sentence_duration').quantile(0.75),
                        sent_timingse_kurt = pl.col('sentence_duration').kurtosis(),
                        sent_timingse_skew = pl.col('sentence_duration').skew(),
        )

        feats.append(sent)
    return feats[0], feats[1]

def activity_time_on_down_time(train_logs, test_logs):

    print('< activity_time_on_down_time >')
    feats = []

    tr_logs, ts_logs = normalise_up_down_times(train_logs, test_logs)
    logs = pl.concat([tr_logs, ts_logs], how='vertical')

    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    logs = logs.select('id','event_id','down_time','activity').filter(~pl.col('activity').str.contains('Move'))

    logs = logs.with_columns(
        pl.col('down_time')
        .diff()
        .over('id')
        .fill_null(0)
        .alias('down_time_diff'))

    logs = logs.group_by('id','activity').agg(

        mean_cumul_down_time_by_act = pl.col('down_time_diff').mean(),
        median_cumul_down_time_by_act = pl.col('down_time_diff').median(),
        std_cumul_down_time_by_act = pl.col('down_time_diff').std(),
        qfirst_cumul_down_time_by_act = pl.col('down_time_diff').quantile(0.25),
        qthird_cumul_down_time_by_act = pl.col('down_time_diff').quantile(0.275,

    )).collect()

    logs = logs.pivot(values=['mean_cumul_down_time_by_act','median_cumul_down_time_by_act',
                            'std_cumul_down_time_by_act','qfirst_cumul_down_time_by_act',
                            'qthird_cumul_down_time_by_act'],
                            columns='activity', index='id').fill_null(0).sort('id')
        
    
    tr_logs = logs.filter(pl.col('id').is_in(tr_ids)).lazy()
    ts_logs = logs.filter(pl.col('id').is_in(ts_ids)).lazy()

    return tr_logs, ts_logs

def word_pauses_ratios(train_logs,test_logs):

    tr_logs, ts_logs = add_word_pauses(train_logs, test_logs)
    tr_re_logs, ts_re_logs = remove_word_pauses(train_logs, test_logs)
    train_feats = tr_logs.join(tr_re_logs, on='id', how='left')
    test_feats = ts_logs.join(ts_re_logs, on='id', how='left')


    train_feats = train_feats.select('id','add_words_pause_mean','add_words_pause_sum','rmv_words_pause_mean','rmv_words_pause_sum')
    train_feats = train_feats.with_columns((pl.col('add_words_pause_mean')/pl.col('rmv_words_pause_mean')).alias('word_pause_mean_ratio'))
    train_feats = train_feats.with_columns((pl.col('add_words_pause_sum')/pl.col('rmv_words_pause_sum')).alias('word_pause_sum_ratio'))
    train_feats = train_feats.drop('add_words_pause_mean','add_words_pause_sum','rmv_words_pause_mean','rmv_words_pause_sum')

    test_feats = test_feats.select('id','add_words_pause_mean','add_words_pause_sum','rmv_words_pause_mean','rmv_words_pause_sum')
    test_feats = test_feats.with_columns((pl.col('add_words_pause_mean')/pl.col('rmv_words_pause_mean')).alias('word_pause_mean_ratio'))
    test_feats = test_feats.with_columns((pl.col('add_words_pause_sum')/pl.col('rmv_words_pause_sum')).alias('word_pause_sum_ratio'))
    test_feats = test_feats.drop('add_words_pause_mean','add_words_pause_sum','rmv_words_pause_mean','rmv_words_pause_sum')

    return train_feats, test_feats

def words_rem_events_ratio(train_logs, test_logs):
    print('< words_rem_events_ratio >')
    feats = []

    logs = pl.concat([train_logs, test_logs], how='vertical')

    all_ids = pl.DataFrame({'id': logs.select(pl.col('id')).unique().collect().to_series().to_list()}).lazy()
    tr_ids = train_logs.select(pl.col('id')).unique().collect().to_series().to_list()
    ts_ids = test_logs.select(pl.col('id')).unique().collect().to_series().to_list()

    count_of_remove_events = logs.filter(
        pl.col('activity')=='Remove/Cut').group_by('id').agg(
            pl.count()
            .alias('count_of_remove_events'))
    
    last_word_count = logs.group_by('id').agg(
        pl.last('word_count')
        .alias('last_word_count'))

    ratio = count_of_remove_events.join(last_word_count, on='id', how='left')
    ratio = ratio.with_columns((pl.col('last_word_count')/pl.col('count_of_remove_events')).alias('words_rem_events_ratio')).drop('count_of_remove_events','last_word_count')

    ratio = all_ids.join(ratio, on='id', how='left').fill_null(0)

    tr_ratio = ratio.filter(pl.col('id').is_in(tr_ids))
    ts_ratio = ratio.filter(pl.col('id').is_in(ts_ids))

    return tr_ratio, ts_ratio

def split_essays_into_sentences(essay_df):
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
    return essay_df

def split_essays_into_words(essay_df):
    essay_df['word'] = essay_df['essay'].apply(lambda x: re.split(' |\\n|\\.|\\?|\\!',x))
    essay_df = essay_df.explode('word')
    # Word length (number of characters in word)
    essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
    essay_df = essay_df[essay_df['word_len'] != 0]
    return essay_df

def sent_long_word_count(sent_df):
    aggregations = ['mean']
    sent_agg_df = sent_df[['id','sent_len','sent_word_count']].groupby(['id']).agg(aggregations)
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    # New features: computing the # of sentences whose (character) length exceed sent_l
    for sent_l in [50, 60, 75, 100]:
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_df[sent_df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
        sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    cols = ['id'] + [f'sent_len_ge_{sent_l}_count' for sent_l in [50, 60, 75, 100]]

    return sent_agg_df[cols]

def word_long_word_count(word_df):
    aggregations = ['mean']
    word_agg_df = word_df[['id','word_len']].groupby(['id']).agg(aggregations)
    word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    word_agg_df['id'] = word_agg_df.index
    # New features: computing the # of words whose length exceed word_l
    for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
        word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    word_agg_df = word_agg_df.reset_index(drop=True)
    cols = ['id'] + [f'word_len_ge_{word_l}_count' for word_l in [5, 6, 7, 8, 9, 10, 11, 12]]
    return word_agg_df[cols]