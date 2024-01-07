import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re

def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

def down_events_counts(train_logs, test_logs):
    print("< Events counts features >")
    logs = pl.concat([train_logs, test_logs], how = 'vertical')
    events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
        'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']

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

def action_time_gap(train_logs, test_logs):
    feats = []
    print("Engineering time data")

    for data in [train_logs, test_logs]:
        logs = data.clone()
        gaps = [1, 2, 3, 5, 10, 20, 50, 100]

        cols_to_drop = [f'up_time_shift{gap}' for gap in gaps]
        for gap in gaps:
            logs = logs.with_columns(pl.col('up_time').shift(gap).over('id').alias(f'up_time_shift{gap}'))
            logs = logs.with_columns((pl.col('down_time') - pl.col(f'up_time_shift{gap}')).alias(f'action_time_gap{gap}'))

        logs = logs.drop(cols_to_drop)

        for gap in gaps:
            stats = logs.group_by('id').agg(

                        pl.col(f'action_time_gap{gap}').max().alias(f'action_time_gap{gap}_max'),
                        pl.col(f'action_time_gap{gap}').min().alias(f'action_time_gap{gap}_min'),
                        pl.col(f'action_time_gap{gap}').mean().alias(f'action_time_gap{gap}_mean'),
                        pl.col(f'action_time_gap{gap}').std().alias(f'action_time_gap{gap}_std'),
                        pl.col(f'action_time_gap{gap}').quantile(0.5).alias(f'action_time_gap{gap}_quantile'),
                        pl.col(f'action_time_gap{gap}').median().alias(f'action_time_gap{gap}_median'),
                        pl.col(f'action_time_gap{gap}').sum().alias(f'action_time_gap{gap}_sum'),
                        pl.col(f'action_time_gap{gap}').skew().alias(f'action_time_gap{gap}_skew'),
                        pl.col(f'action_time_gap{gap}').kurtosis().alias(f'action_time_gap{gap}_kurt'),
            )

        feats.append(stats)
    return feats[0], feats[1]

def cursor_position_change_gap(train_logs, test_logs):
    feats = []
    print("Engineering cursor position data")

    for data in [train_logs, test_logs]:
        logs = data.clone()
        gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        cols_to_drop = [f'cursor_position_shift{gap}' for gap in gaps]
        for gap in gaps:
            logs = logs.with_columns(pl.col('cursor_position').shift(gap).over('id').alias(f'cursor_position_shift{gap}'))
            logs = logs.with_columns((pl.col('cursor_position') - pl.col(f'cursor_position_shift{gap}')).alias(f'cursor_position_change{gap}'))
            logs = logs.with_columns(pl.col(f'cursor_position_change{gap}').abs().alias(f'cursor_position_abs_change{gap}'))
        
        logs = logs.drop(cols_to_drop) 

        for gap in gaps:
            stats = logs.group_by('id').agg(

                        pl.col(f'cursor_position_abs_change{gap}').max().alias(f'cursor_position_change{gap}_max'),
                        pl.col(f'cursor_position_abs_change{gap}').min().alias(f'cursor_position_change{gap}_min'),
                        pl.col(f'cursor_position_abs_change{gap}').mean().alias(f'cursor_position_change{gap}_mean'),
                        pl.col(f'cursor_position_abs_change{gap}').std().alias(f'cursor_position_change{gap}_std'),
                        pl.col(f'cursor_position_abs_change{gap}').quantile(0.5).alias(f'cursor_position_change{gap}_quantile'),
                        pl.col(f'cursor_position_abs_change{gap}').median().alias(f'cursor_position_change{gap}_median'),
                        pl.col(f'cursor_position_abs_change{gap}').sum().alias(f'cursor_position_change{gap}_sum'),
                        pl.col(f'cursor_position_abs_change{gap}').skew().alias(f'cursor_position_change{gap}_skew'),
                        pl.col(f'cursor_position_abs_change{gap}').kurtosis().alias(f'cursor_position_change{gap}_kurt'),
            )

        feats.append(stats)
    return feats[0], feats[1]

def word_count_change_gap(train_logs, test_logs):

    feats = []
    print("Engineering word count data")

    for data in [train_logs, test_logs]:
        logs = data.clone()
        gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        cols_to_drop = [f'word_count_shift{gap}' for gap in gaps]
        for gap in gaps:
            logs = logs.with_columns(pl.col('word_count').shift(gap).over('id').alias(f'word_count_shift{gap}'))
            logs = logs.with_columns((pl.col('word_count') - pl.col(f'word_count_shift{gap}')).alias(f'word_count_change{gap}'))
            logs = logs.with_columns(pl.col(f'word_count_change{gap}').abs().alias(f'word_count_abs_change{gap}'))
        
        logs = logs.drop(cols_to_drop) 

        for gap in gaps:
            stats = logs.group_by('id').agg(

                        pl.col(f'word_count_abs_change{gap}').max().alias(f'word_count_change{gap}_max'),
                        pl.col(f'word_count_abs_change{gap}').min().alias(f'word_count_change{gap}_min'),
                        pl.col(f'word_count_abs_change{gap}').mean().alias(f'word_count_change{gap}_mean'),
                        pl.col(f'word_count_abs_change{gap}').std().alias(f'word_count_change{gap}_std'),
                        pl.col(f'word_count_abs_change{gap}').quantile(0.5).alias(f'word_count_change{gap}_quantile'),
                        pl.col(f'word_count_abs_change{gap}').median().alias(f'word_count_change{gap}_median'),
                        pl.col(f'word_count_abs_change{gap}').sum().alias(f'word_count_change{gap}_sum'),
                        pl.col(f'word_count_abs_change{gap}').skew().alias(f'word_count_change{gap}_skew'),
                        pl.col(f'word_count_abs_change{gap}').kurtosis().alias(f'word_count_change{gap}_kurt'),
            )

        feats.append(stats)
    return feats[0], feats[1]

def categorical_stats(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        categorical_stats = logs.group_by('id', maintain_order=True).agg(

            pl.col('event_id').max().name.suffix('_max'),
            pl.col('up_time').max().name.suffix('_max'),
            pl.col('activity').n_unique().name.suffix('_n_unique'),
            pl.col('down_event').n_unique().name.suffix('_n_unique'),
            pl.col('up_event').n_unique().name.suffix('_n_unique'),
            pl.col('text_change').n_unique().name.suffix('_n_unique'),
        )

        feats.append(categorical_stats)
    return feats[0], feats[1]
def action_time_stats(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        action_time_stats = logs.group_by('id', maintain_order=True).agg(

            action_time_max = pl.col('action_time').max(),
            action_time_min = pl.col('action_time').min(),
            action_time_mean = pl.col('action_time').mean(),
            action_time_std = pl.col('action_time').std(),
            action_time_quantile = pl.col('action_time').quantile(0.5),
            action_time_median = pl.col('action_time').median(),
            action_time_sum = pl.col('action_time').sum(),
            action_time_skew = pl.col('action_time').skew(),
            action_time_kurt = pl.col('action_time').kurtosis(),
    )

        feats.append(action_time_stats)
    return feats[0], feats[1]
def cursor_position_stats(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        cursor_position_stats = logs.group_by('id', maintain_order=True).agg(

            cursor_position_max = pl.col('cursor_position').max(),
            cursor_position_mean = pl.col('cursor_position').mean(),
            cursor_position_quantile = pl.col('cursor_position').quantile(0.5),
            cursor_position_median = pl.col('cursor_position').median(),
            cursor_position_nunique = pl.col('cursor_position').n_unique()
        )
        feats.append(cursor_position_stats)
    return feats[0], feats[1]

def word_count_stats(train_logs, test_logs):

    feats = []
    for data in [train_logs, test_logs]:
        logs = data.clone()
        word_count_stats = logs.group_by('id', maintain_order=True).agg(

            word_count_max = pl.col('word_count').max(),
            word_count_mean = pl.col('word_count').mean(),
            word_count_quantile = pl.col('word_count').quantile(0.5),
            word_count_median = pl.col('word_count').median(),
            word_count_nunique = pl.col('word_count').n_unique()
        )
        feats.append(word_count_stats)
    return feats[0], feats[1]        


    #     '''
    #     Credit: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features
    #     '''
    #     cnts = ret_tfidf.sum(1)

    #     for col in tfidf_cols:
    #         if col in self.idf.keys():
    #             idf = self.idf[col]
    #         else:
    #             idf = df.shape[0] / (ret_tfidf[col].sum() + 1)
    #             idf = np.log(idf)
    #             self.idf[col] = idf

    #         ret_tfidf[col] = 1 + np.log(ret_tfidf[col] / cnts)
    #         ret_tfidf[col] *= idf
        
    #     ret_agg = pd.concat([ret_tfidf, ret_normal], axis=1)
    #     return ret_agg

    # def event_counts(self, df, colname):
    #     tmp_df = df.groupby('id').agg({colname: list}).reset_index()
    #     ret = list()
    #     for li in tqdm(tmp_df[colname].values):
    #         items = list(Counter(li).items())
    #         di = dict()
    #         for k in self.events:
    #             di[k] = 0
    #         for item in items:
    #             k, v = item[0], item[1]
    #             if k in di:
    #                 di[k] = v
    #         ret.append(di)
            
    #     ret = pd.DataFrame(ret)
    #     # using tfidf
    #     ret_tfidf = pd.DataFrame(ret)
    #     # returning counts as is
    #     ret_normal = pd.DataFrame(ret)
        
    #     tfidf_cols = [f'{colname}_{event}_tfidf_count' for event in ret.columns]
    #     normal_cols = [f'{colname}_{event}_normal_count' for event in ret.columns]
        
    #     ret_tfidf.columns = tfidf_cols
    #     ret_normal.columns = normal_cols
        
    #     '''
    #     Credit: https://www.kaggle.com/code/olyatsimboy/towards-tf-idf-in-logs-features
    #     '''
    #     cnts = ret_tfidf.sum(1)

    #     for col in tfidf_cols:
    #         if col in self.idf.keys():
    #             idf = self.idf[col]
    #         else:
    #             idf = df.shape[0] / (ret_tfidf[col].sum() + 1)
    #             idf = np.log(idf)
    #             self.idf[col] = idf

    #         ret_tfidf[col] = 1 + np.log(ret_tfidf[col] / cnts)
    #         ret_tfidf[col] *= idf
        
    #     ret_agg = pd.concat([ret_tfidf, ret_normal], axis=1)
    #     return ret_agg


    # def compute_sentence_aggregations(self, sent_df):
    #     sent_agg_df = sent_df[['id','sent_len','sent_word_count']].groupby(['id']).agg(self.aggregations)
    #     sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    #     sent_agg_df['id'] = sent_agg_df.index
    #     # New features: computing the # of sentences whose (character) length exceed sent_l
    #     for sent_l in [50, 60, 75, 100]:
    #         sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_df[sent_df['sent_len'] >= sent_l].groupby(['id']).count().iloc[:, 0]
    #         sent_agg_df[f'sent_len_ge_{sent_l}_count'] = sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
    #     sent_agg_df = sent_agg_df.reset_index(drop=True)
    #     return sent_agg_df


    # def compute_word_aggregations(self, word_df):
    #     word_agg_df = word_df[['id','word_len']].groupby(['id']).agg(self.aggregations)
    #     word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
    #     word_agg_df['id'] = word_agg_df.index
    #     # New features: computing the # of words whose length exceed word_l
    #     for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
    #         word_agg_df[f'word_len_ge_{word_l}_count'] = word_df[word_df['word_len'] >= word_l].groupby(['id']).count().iloc[:, 0]
    #         word_agg_df[f'word_len_ge_{word_l}_count'] = word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
    #     word_agg_df = word_agg_df.reset_index(drop=True)
    #     return word_agg_df