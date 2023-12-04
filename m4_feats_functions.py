import pandas as pd
import numpy as np
import copy, re

from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Function to construct essays copied from here (small adjustments): https://www.kaggle.com/code/yuriao/fast-essay-constructor
def processingInputs(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        # Input[0] = activity
        # Input[1] = cursor_position
        # Input[2] = text_change
        # Input[3] = id
        # If activity = Replace
        if Input[0] == 'Replace':
            # splits text_change at ' => '
            replaceTxt = Input[2].split(' => ')
            essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
            continue

        # If activity = Paste    
        if Input[0] == 'Paste':
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
            continue

        # If activity = Remove/Cut
        if Input[0] == 'Remove/Cut':
            essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
            continue

        # If activity = Move...
        if "M" in Input[0]:
            # Gets rid of the "Move from to" text
            croppedTxt = Input[0][10:]              
            # Splits cropped text by ' To '
            splitTxt = croppedTxt.split(' To ')              
            # Splits split text again by ', ' for each item
            valueArr = [item.split(', ') for item in splitTxt]              
            # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
            moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))
            # Skip if someone manages to activiate this by moving to same place
            if moveData[0] != moveData[2]:
                # Check if they move text forward in essay (they are different)
                if moveData[0] < moveData[2]:
                    essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] +\
                    essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                else:
                    essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] +\
                    essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
            continue                

        # If activity = input
        essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
    return essayText


def getEssays(df):
    # Copy required columns
    textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
    # Get rid of text inputs that make no change
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
    
    # Construct essay, fast 
    tqdm.pandas()
    essays = textInputDf.groupby('id')[['activity', 'cursor_position', 'text_change']].progress_apply(lambda x: processingInputs(x))

    # Check if essays is a Series and convert to DataFrame if necessary
    if isinstance(essays, pd.Series):
        essayFrame = essays.to_frame().reset_index()
        essayFrame.columns = ['id', 'essay']
    elif isinstance(essays, pd.DataFrame):
        essayFrame = essays.reset_index()
    else:
        # Handle unexpected output
        essayFrame = pd.DataFrame(columns=['id', 'essay'])

    return essayFrame
# Helper functions
def q1(x):
    return x.quantile(0.25)
def q3(x):
    return x.quantile(0.75)

AGGREGATIONS = ['count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3, 'skew', pd.DataFrame.kurt, 'sum']

def split_essays_into_sentences(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!',x))
    essay_df = essay_df.explode('sent')
    essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n','').strip())
    # Number of characters in sentences
    essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
    # Number of words in sentences
    essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.sent_len!=0].reset_index(drop=True)
    return essay_df

def compute_sentence_aggregations(df):
    sent_agg_df = pd.concat(
        [df[['id','sent_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','sent_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    )
    sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
    sent_agg_df['id'] = sent_agg_df.index
    sent_agg_df = sent_agg_df.reset_index(drop=True)
    sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
    sent_agg_df = sent_agg_df.rename(columns={"sent_len_count":"sent_count"})
    return sent_agg_df

def split_essays_into_paragraphs(df):
    essay_df = df
    #essay_df['id'] = essay_df.index
    essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
    essay_df = essay_df.explode('paragraph')
    # Number of characters in paragraphs
    essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x)) 
    # Number of words in paragraphs
    essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
    essay_df = essay_df[essay_df.paragraph_len!=0].reset_index(drop=True)
    return essay_df

def compute_paragraph_aggregations(df):
    paragraph_agg_df = pd.concat(
        [df[['id','paragraph_len']].groupby(['id']).agg(AGGREGATIONS), df[['id','paragraph_word_count']].groupby(['id']).agg(AGGREGATIONS)], axis=1
    ) 
    paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
    paragraph_agg_df['id'] = paragraph_agg_df.index
    paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
    paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
    paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count":"paragraph_count"})
    return paragraph_agg_df

# The following code comes almost Abdullah's notebook: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
# Abdullah's code is based on work shared in previous notebooks (e.g., https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline)

from collections import defaultdict

# The following code comes almost Abdullah's notebook: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
# Abdullah's code is based on work shared in previous notebooks (e.g., https://www.kaggle.com/code/hengzheng/link-writing-simple-lgbm-baseline)

from collections import defaultdict

class Preprocessor:
    
    def __init__(self, seed):
        self.seed = seed
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        
        self.idf = defaultdict(float)
    
    def activity_counts(self, df):
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.activities:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf
            
        return ret

    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    def make_feats(self, df):
        
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        
        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt])
            ])
        
        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']

        return feats
    
    # Code for additional aggregations comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs

def agg_fe_df(train_logs, test_logs):
    train_agg_fe_df = train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
        ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum']) # what about quantiles, IQR, etc.
    
    train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
    train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
    train_agg_fe_df.reset_index(inplace=True)

    test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(
        ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
    test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
    test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
    test_agg_fe_df.reset_index(inplace=True)

    return train_agg_fe_df, test_agg_fe_df

# Code for creating these features comes from here: https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs
# Idea is based on features introduced in Section 3 of this research paper: https://files.eric.ed.gov/fulltext/ED592674.pdf
# added pause features from https://www.kaggle.com/code/eraikako/from-a-to-z-eda-feature-engineering-modeling 


def pauses_feats(train_logs, test_logs):
    data = []

    for logs in [train_logs, test_logs]:
        logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
        logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

        group = logs.groupby('id')['time_diff']
        largest_lantency = group.max()
        smallest_lantency = group.min()
        median_lantency = group.median()
        initial_pause = logs.groupby('id')['down_time'].first() / 1000
        pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
        pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
        pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
        pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
        pauses_3_sec = group.apply(lambda x: (x > 3).sum())

        data.append(pd.DataFrame({
            'id': logs['id'].unique(),
            'largest_lantency': largest_lantency,
            'smallest_lantency': smallest_lantency,
            'median_lantency': median_lantency,
            'initial_pause': initial_pause,
            'pauses_half_sec': pauses_half_sec,
            'pauses_1_sec': pauses_1_sec,
            'pauses_1_half_sec': pauses_1_half_sec,
            'pauses_2_sec': pauses_2_sec,
            'pauses_3_sec': pauses_3_sec
        }).reset_index(drop=True))

    train_eD592674, test_eD592674 = data
    return train_eD592674, test_eD592674

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# WORD STATS ON ESSAY
def diverse_stats(series, prefix):
    if isinstance(series, list):
        series = pd.Series([len(item) for item in series])

    stats = {
        f'{prefix}_count': series.count(),
        f'{prefix}_mean': series.mean(),
        f'{prefix}_std': series.std(),
        f'{prefix}_max': series.max(),
        f'{prefix}_median': series.median(),
        f'{prefix}_sum': series.sum(),
        f'{prefix}_last': series.iloc[-1] if not series.empty else None,
        f'{prefix}_q1': series.quantile(0.25),
        f'{prefix}_q3': series.quantile(0.75),
        f'{prefix}_iqr': series.quantile(0.75) - series.quantile(0.25),
        f'{prefix}_min': series.min(),
        # f'{prefix}_first': series.iloc[0] if not series.empty else None,
        # f'{prefix}_sem': series.sem(),
        # f'{prefix}_skew': series.skew(),
        # f'{prefix}_kurt': series.kurtosis(),
        # f'{prefix}_range': series.max() - series.min(),
    }
    return pd.Series(stats)

def word_length_stats(text, scope):
    words = text.split()
    word_lengths = [len(word) for word in words]
    word_len_feats = diverse_stats(pd.Series(word_lengths), scope)
    return word_len_feats

def create_word_length_features(df, text_column, id_column, scope):
    columns_to_drop = {'sent', 'paragraph'}
    columns_to_drop = columns_to_drop.intersection(df.columns)
    df = df.drop(columns_to_drop, axis=1)
    df.reset_index(inplace=True, drop=True)
    df.columns = [id_column, text_column]

    grouped = df.groupby(id_column)
    features = {}

    for id, group in grouped:
        concatenated_text = ' '.join(group[text_column].astype(str))
        features[id] = word_length_stats(concatenated_text, scope)

    features_df = pd.DataFrame(features).T.reset_index()
    features_df.rename(columns={'index': id_column}, inplace=True)
    return features_df


time_buckets=[0, 450, 900, 1350, 1805]
def transform_df_into_action_buckets(df, time_buckets=time_buckets):
    time_buckets = [item * 1000 for item in time_buckets]
    bucket_labels = list(np.arange(1, len(time_buckets), 1))

    action_df = df.copy()
    action_df['adj_dt'] = action_df['down_time'] - action_df.groupby('id')['down_time'].transform('min')
    
    max_adj_dt = action_df.groupby('id')['down_time'].transform('min') + (30 * 60 * 1000)
    action_df['adj_dt'] = action_df['adj_dt'].clip(upper=max_adj_dt)

    action_df['adj_ut'] = action_df['adj_dt'] + action_df['action_time']
    action_df['time_all_bckt'] = pd.cut(action_df['down_time'], bins=time_buckets, labels=bucket_labels, right=False)
    action_df = action_df.drop(['up_time', 'down_time'], axis=1)
    
    return action_df

# ACTION_TIME BY BUCKETS
def create_feats_buckets_action_time(comb_df):
    actions_df = transform_df_into_action_buckets(comb_df)
    
    colname, method = ('action_time', ['nunique', 'max', 'mean'])
    df = actions_df.groupby(['id', 'time_all_bckt']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method}'})
    
    # Adjusting column names
    df.columns = ['id', 'time_bucket', 'nunique', 'max', 'mean']
    df = df.reset_index()
    
    feats = df.pivot_table(index='id', 
                                columns='time_bucket', 
                                values=['nunique', 'max', 'mean'],
                                aggfunc='first')

    feats.columns = ['bucket_at_{}_{}'.format(col[1], col[0]) for col in feats.columns]
    feats.reset_index(inplace=True)
    return feats

def action_time_by_bucket_feats(train_logs, test_logs):
    train_feats = create_feats_buckets_action_time(train_logs)
    test_feats = create_feats_buckets_action_time(test_logs)
    return train_feats, test_feats

def process_action_time_activity(train_logs, test_logs):
    def create_action_time_activity_features(train_logs):
        df_logs = train_logs.copy()
        msk = df_logs['activity'].str.contains('Move From')
        df_logs.loc[msk, 'activity'] = 'Move'
        grouped = df_logs.groupby(['id', 'activity'])['action_time']
        action_time_act_stats = grouped.apply(lambda x: diverse_stats(x, 'action_time_act')).reset_index()
        action_time_act_stats['feat_name'] = action_time_act_stats['activity'] + "_" + action_time_act_stats['level_2']
        action_time_act_feats = action_time_act_stats[['id', 'feat_name', 'action_time']].pivot(index=['id'], columns=['feat_name'], values=['action_time']).reset_index()
        new_columns = [col[1] if col[0] == 'id' else '_'.join(col) for col in action_time_act_feats.columns]
        new_columns[0] = 'id'
        action_time_act_feats.columns = new_columns
        return action_time_act_feats.fillna(0)

    # Apply feature creation to train and test logs
    train_action_time_activity = create_action_time_activity_features(train_logs)
    test_action_time_activity = create_action_time_activity_features(test_logs)

    return train_action_time_activity, test_action_time_activity

# EFFECTIVE_TIME BASED ON ADJUSTED DOWN_TIME UP_TIME
def create_adjusted_eff_time_df(comb_df):
    eff_time_df = comb_df[['id', 'down_time', 'up_time', 'action_time', 'event_id']].sort_values(['id', 'event_id']).copy()
    eff_time_df['adj_dt'] = eff_time_df['down_time'] - eff_time_df.groupby('id')['down_time'].transform('min')

    max_adj_dt = eff_time_df.groupby('id')['down_time'].transform('min') + (30 * 60 * 1000)
    eff_time_df['adj_dt'] = eff_time_df['adj_dt'].clip(upper=max_adj_dt)

    eff_time_df['adj_ut'] = eff_time_df['adj_dt'] + eff_time_df['action_time']
    eff_time_df['eff_time'] = eff_time_df.groupby(['id'])['action_time'].cumsum()
    eff_time_df['eff_time'] = eff_time_df.groupby(['id'])['eff_time'].shift(1)

    eff_time_df['eff_time_gap'] = eff_time_df['down_time'] - eff_time_df['eff_time']
    eff_time_df['adj_eff_time_gap'] = eff_time_df['adj_dt'] - eff_time_df['eff_time']
    return eff_time_df

def process_adjusted_eff_time(train_logs, test_logs):
    def create_aggregations(comb_df):
        # Your existing aggregation logic
        agg_df = comb_df.groupby("id")[['adj_dt', 'adj_ut']].agg(
            ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
        agg_df.columns = ['_'.join(x) for x in agg_df.columns]
        agg_df = agg_df.add_prefix("tmp_")
        agg_df.reset_index(inplace=True)
        return agg_df

    # Apply create_adjusted_eff_time_df and aggregations to train logs
    train_eff = create_adjusted_eff_time_df(train_logs)
    train_adj_agg_fe_df = create_aggregations(train_eff)

    # Apply create_adjusted_eff_time_df and aggregations to test logs
    test_eff = create_adjusted_eff_time_df(test_logs)
    test_adj_agg_fe_df = create_aggregations(test_eff)
    return train_adj_agg_fe_df, test_adj_agg_fe_df

def process_re_cut_essays(train_logs, test_logs):
    def filter_and_transform_re_cut(logs_df):
        re_cut_df = logs_df[logs_df['activity'] == 'Remove/Cut'].copy()
        re_cut_df['activity'] = 'Input'
        return re_cut_df
    
    train_re_cut = filter_and_transform_re_cut(train_logs)
    train_re_cut_df = getEssays(train_re_cut)
    train_cut_words = create_word_length_features(train_re_cut_df, 'essay', 'id', 're_cut')

    test_re_cut = filter_and_transform_re_cut(test_logs)

    if test_re_cut.empty:
        test_cut_words = pd.DataFrame(columns=train_cut_words.columns)
        test_cut_words['id'] = test_logs['id'].unique()
        test_cut_words.iloc[:, 1:] = np.nan
    else:
        test_re_cut_df = getEssays(test_re_cut)
        test_cut_words = create_word_length_features(test_re_cut_df, 'essay', 'id', 're_cut')

    return train_cut_words, test_cut_words

def process_feats_action_time_gap(train_logs, test_logs):
    def calc_action_time_gap(comb_df):
        action_time_gap_df = comb_df.copy()
        action_time_gap_df['up_time_shift1'] = action_time_gap_df.groupby('id')['up_time'].shift(1)
        action_time_gap_df['action_time_gap'] = action_time_gap_df['down_time'] - action_time_gap_df['up_time_shift1']

        grouped = action_time_gap_df.groupby('id')['action_time_gap']
        return grouped.apply(lambda x: diverse_stats(x, 'action_time_gap')).reset_index()

    train_action_time_gap_feats = calc_action_time_gap(train_logs).pivot(index='id', columns='level_1', values='action_time_gap').reset_index()
    test_action_time_gap_feats = calc_action_time_gap(test_logs).pivot(index='id', columns='level_1', values='action_time_gap').reset_index()

    return train_action_time_gap_feats, test_action_time_gap_feats

def process_feats_time_gap_activity(train_logs, test_logs):
    def calc_time_gap_activity(comb_df):
        action_time_gap_df = comb_df.copy()
        msk = comb_df['activity'].str.contains('Move From')
        action_time_gap_df.loc[msk, 'activity'] = 'Move'
        action_time_gap_df['up_time_shift1'] = action_time_gap_df.groupby('id')['up_time'].shift(1)
        action_time_gap_df['action_time_gap'] = action_time_gap_df['down_time'] - action_time_gap_df['up_time_shift1']
        
        grouped = action_time_gap_df.groupby(['id', 'activity'])['action_time_gap']
        action_time_gap_stats = grouped.apply(lambda x: diverse_stats(x, 'action_time_gap')).reset_index()
        action_time_gap_stats['feat_name'] = action_time_gap_stats['activity'] + "_" + action_time_gap_stats['level_2']
        return action_time_gap_stats[['id', 'feat_name', 'action_time_gap']].pivot(index=['id'], columns=['feat_name'], values=['action_time_gap']).reset_index()

    train_time_gap_activity_feats = calc_time_gap_activity(train_logs)
    test_time_gap_activity_feats = calc_time_gap_activity(test_logs)

    # Adjust column names
    for df in [train_time_gap_activity_feats, test_time_gap_activity_feats]:
        new_columns = [col[1] if col[0] == 'id' else '_'.join(col) for col in df.columns]
        new_columns[0] = 'id'
        df.columns = new_columns

    return train_time_gap_activity_feats, test_time_gap_activity_feats

def calculate_pause_features(essay_df, pause_threshold=2000):
    """Calculate various pause-related features for an essay.

    This function computes the Inter-Keystroke Intervals (IKIs) for each keystroke
    in the given dataframe, identifies pauses based on a specified threshold,
    calculates features such as the number of pauses, total pause time, mean pause length,
    and the proportion of pause time.

    Args:
        essay_df (pd.DataFrame): A dataframe representing the keystroke log of a single essay. 
                                 It should contain at least 'down_time' and 'up_time' columns.
        pause_threshold (int, optional): The threshold (in milliseconds) to define a pause. 
                                         A pause is considered when the IKI is greater than this value. 
                                         Defaults to 2000 milliseconds.

    Returns:
        pd.Series: A Series containing the calculated pause features.
    """
    # Calculate IKIs
    # This line shifts the down_time column up by one row, so that each down_time aligns with the up_time of the previous keystroke.
    # The subtraction then gives the IKI for each pair of consecutive keystrokes.
    #essay_df["IKI"] = essay_df["down_time"].shift(-1) - essay_df["up_time"]    
    # Calculate IKI with a groupby and transform, which will reset the diff() calculation for each group
    essay_df["IKI"] = essay_df["down_time"].diff().fillna(0)

    # Identify pauses (IKI > 2000 milliseconds)
    pauses = essay_df[essay_df["IKI"] > pause_threshold]

    # Calculate pause features
    num_pauses = len(pauses)

    total_pause_time = pauses["IKI"].sum()
    total_writing_time = essay_df["up_time"].max() - essay_df["down_time"].min()
    proportion_pause_time = (total_pause_time / total_writing_time) * 100 if total_writing_time != 0 else 0
    mean_pause_length = pauses["IKI"].mean() if num_pauses != 0 else 0

    return pd.Series({
        "IKI_num_pauses": num_pauses,
        "IKI_total_pause_time": total_pause_time,
        "IKI_proportion_pause_time": proportion_pause_time,
        "IKI_mean_pause_length": mean_pause_length
    })

def basic_stats(series, prefix):
    if isinstance(series, list):
        series = pd.Series([len(item) for item in series])

    stats = {
        f'{prefix}_count': series.count(),
        f'{prefix}_mean': series.mean(),
        f'{prefix}_std': series.std(),
        f'{prefix}_max': series.max(),
        f'{prefix}_sum': series.sum(),

    }
    return pd.Series(stats)

def create_feats_wc_change(comb_df):
    wc_change_df = comb_df.copy()
    wc_change_df['word_count_shift1'] = wc_change_df.groupby('id')['word_count'].shift(1)
    wc_change_df['word_count_change'] = np.abs(wc_change_df['word_count'] - wc_change_df['word_count_shift1']) #why abs

    msk = wc_change_df['activity'].str.contains('Move From')
    wc_change_df.loc[msk, 'activity'] = 'Move'

    grouped = wc_change_df.groupby(['id', 'activity'])['word_count_change']
    wc_change_stats = grouped.apply(lambda x: basic_stats(x, 'wc_change')).reset_index()
    wc_change_stats['feat_name'] = wc_change_stats['activity'] + "_" + wc_change_stats['level_2']
    wc_change_feats = wc_change_stats[['id', 'feat_name', 'word_count_change']].pivot(index=['id'], columns=['feat_name'], values=['word_count_change']).reset_index()
    new_columns = [col[1] if col[0] == 'id' else '_'.join(col) for col in wc_change_feats.columns]
    new_columns[0] = 'id'
    wc_change_feats.columns = new_columns
    return wc_change_feats.fillna(0, inplace=True)

def essay_replace_words(train_logs, test_logs):
    train_replace = train_logs[train_logs['activity']=='Replace'].copy()
    train_replace['activity'] = 'Input'
    train_replace_df = getEssays(train_replace)
    train_replace_words = create_word_length_features(train_replace_df, 'essay', 'id', 'replace')

    tr_cols = train_replace_words.columns
    test_replace = test_logs[test_logs['activity']=='Replace'].copy()
    test_replace['activity'] = 'Input'
    # Check if test_rereplace is empty
    if test_replace.empty:
        # Create a DataFrame with test_logs['id'] and columns of train_cut_words filled with NaN
        test_replace_words = pd.DataFrame(columns=tr_cols)
        test_replace_words['id'] = test_logs.id.unique()
        for col in tr_cols[1:]:
            test_replace_words[col] = np.nan
    else:
        test_replace_df = getEssays(test_replace)
        test_replace_words = create_word_length_features(test_replace_df, 'essay', 'id', 'replace')

    return train_replace_words.fillna(0, inplace=True), test_replace_words.fillna(0, inplace=True)


# WORDS PER MINUTE
# https://www.kaggle.com/code/eraikako/from-a-to-z-eda-feature-engineering-modeling - His IKI features were impressive but didn't work out in CV

def wpm_feats(train_logs, test_logs):
    data = []
    for logs in [train_logs, test_logs]:
        grouped_logs = logs.groupby("id").agg(start_time=("down_time", "min"), end_time=("up_time", "max"))
        grouped_logs["duration_ms"] = grouped_logs["end_time"] - grouped_logs["start_time"]
        grouped_logs["duration_min"] = grouped_logs["duration_ms"] / 60000

        final_word_count = logs.groupby("id")["word_count"].last()
        grouped_logs = pd.merge(grouped_logs, final_word_count, left_index=True, right_index=True)
        grouped_logs["WPM"] = grouped_logs["word_count"] / grouped_logs["duration_min"]


        grouped_logs.drop(['start_time', 'end_time', 'duration_ms'], axis=1, inplace=True)
        data.append(grouped_logs)

    train_wpm, test_wpm = data
    return train_wpm, test_wpm

def essay_paste_words(train_logs, test_logs):
    train_paste = train_logs[train_logs['activity'] == 'Paste'].copy()
    train_paste['activity'] = 'Input'
    train_paste_df = getEssays(train_paste)
    train_paste_words = create_word_length_features(train_paste_df, 'essay', 'id', 'paste')

    test_paste = test_logs[test_logs['activity'] == 'Paste'].copy()
    test_paste['activity'] = 'Input'

    tr_cols = train_paste_words.columns

    # Check if test_paste is empty
    if test_paste.empty:
        # Create a DataFrame with test_logs['id'] and columns of train_paste_words filled with NaN
        test_paste_words = pd.DataFrame(columns=tr_cols)
        test_paste_words['id'] = test_logs.id.unique()
        for col in train_paste_words.columns[1:]:
            test_paste_words[col] = np.nan
    else:
        test_paste_df = getEssays(test_paste)
        test_paste_words = create_word_length_features(test_paste_df, 'essay', 'id', 'paste')

    train_paste_words.fillna(0, inplace=True)
    test_paste_words.fillna(0, inplace=True)
    return train_logs, test_logs

def countvectorize_one_one(train_logs, test_logs, train_feats, test_feats):

    tr_ids = train_feats.id
    tst_ids = test_feats.id
    tr_ts_logs = pd.concat([train_logs, test_logs], axis=0)
    tr_ts_feats = pd.concat([train_feats['id'], test_feats['id']], axis=0).reset_index(drop=True)

    essays = getEssays(tr_ts_logs)
    c_vect = CountVectorizer(ngram_range=(1, 1))
    toks = c_vect.fit_transform(essays['essay']).todense()
    toks_df = pd.DataFrame(columns = [f'tok_{i}' for i in range(toks.shape[1])], data=toks)
    toks_df.reset_index(drop=True, inplace=True)
    print(toks_df.shape, tr_ts_feats.shape)

    tr_ts_feats = pd.concat([tr_ts_feats, toks_df], axis=1)

    train_feats = tr_ts_feats[tr_ts_feats['id'].isin(tr_ids)]
    test_feats = tr_ts_feats[tr_ts_feats['id'].isin(tst_ids)]

    return train_feats, test_feats

def get_keys_pressed_per_minute(train_logs, test_logs):
    inputs_remove_cut = train_logs[train_logs['activity'].isin(['Input', 'Remove/Cut'])]
    total_keys_pressed = inputs_remove_cut.groupby(['id']).agg(keys_pressed_per_minute=('event_id', 'count'))
    train_ = round(total_keys_pressed / 60, 2)

    inputs_remove_cut = test_logs[test_logs['activity'].isin(['Input', 'Remove/Cut'])]
    total_keys_pressed = inputs_remove_cut.groupby(['id']).agg(keys_pressed_per_minute=('event_id', 'count'))
    test_ = round(total_keys_pressed / 60, 2)

    return train_, test_