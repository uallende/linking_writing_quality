import pandas as pd
import warnings, re
from feats_functions import *
warnings.filterwarnings("ignore", category=FutureWarning)

def feature_creation_pipeline(data_path):

    data_path = 'kaggle/input/linking-writing-processes-to-writing-quality/'
    train_logs = pd.read_csv(f'{data_path}train_logs.csv') 
    train_scores = pd.read_csv(f'{data_path}train_scores.csv')
    test_logs = pd.read_csv(f'{data_path}test_logs.csv')
    sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')
    comb_df = pd.concat([train_logs, test_logs], axis=0)

    comb_df = pd.concat([train_logs, test_logs], axis=0)
    simple_feats = make_feats(comb_df)
    final = simple_feats.merge(train_scores, on=['id'], how='left')

    essayConstructor = EssayConstructor()
    essays_df = essayConstructor.getEssays(comb_df)
    sentence_delimiter = re.compile(r'[.?!]')
    sen_df = split_text_compute_stats(essays_df, 'text', sentence_delimiter, 'sent')
    paragraph_delimiter = '\n'
    par_df = split_text_compute_stats(essays_df, 'text', paragraph_delimiter, 'paragraph')

    msk = comb_df['activity'].str.contains('Move From')
    comb_df.loc[msk, 'activity'] = 'Move'

    word_feats = create_word_length_features(essays_df, 'text', 'id', 'essay_words')
    final = final.merge(word_feats, on=['id'], how='left')

    sent_feats = create_essay_features(sen_df, 'sent_len', 'sent_word_count')
    final = final.merge(sent_feats, on=['id'], how='left')

    par_feats = create_essay_features(par_df, 'paragraph_len', 'paragraph_word_count')
    final = final.merge(par_feats, on=['id'], how='left')

    activities_feats = create_feats_activities_stats(comb_df)
    final = final.merge(activities_feats, on=['id'], how='left')

    action_time_feats = create_features_action_time(comb_df)
    final = final.merge(action_time_feats, on=['id'], how='left')

    action_time_gap_feats = create_feats_action_time_gap(comb_df)
    final = final.merge(action_time_gap_feats, on=['id'], how='left')

    buckets_wc_feats = create_feats_buckets_wc(comb_df)
    final = final.merge(buckets_wc_feats, on=['id'], how='left')

    action_time_gap_activity_feats = create_feats_time_gap_activity(comb_df)
    final = final.merge(action_time_gap_activity_feats, on=['id'], how='left')


    final=final.fillna(-1)
    print(f'Final shape: {final.shape}')
    return final    

print(f'Data prep OK')
