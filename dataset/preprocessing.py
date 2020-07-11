import os, sys, gzip
import argparse, random, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from operator import itemgetter
from collections import defaultdict
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer


class CrossData:
    def __init__(self, path_s, path_t, ratio, thre_i, thre_u):
        self.path_s, self.path_t, self.ratio, self.thre_i, self.thre_u= path_s, path_t, ratio, thre_i, thre_u
        self.df_s, self.df_t = self.get_df()
        self.df_s, self.df_t = self.filterout(self.df_s, self.df_t, thre_i, thre_u)
        self.udict, self.idict_s, self.idict_t, self.overlap_user_set = self.convert_idx()
        self.coldstart_user_set, self.common_user_set, self.train_common_s, self.train_common_t, \
            self.train_s, self.train_t, self.train_cs_s, self.vali, self.test = self.split_train_test()
        if args.info: exit()
        self.vocab_dict, self.word_embedding = self.get_w2v()
        self.docu_udict_s, self.docu_idict_s, self.auxiliary_docu_udict_s = \
            self.get_documents(self.train_s, self.coldstart_user_set | self.common_user_set)
        self.docu_udict_t, self.docu_idict_t, self.auxiliary_docu_udict_t = \
            self.get_documents(self.train_t, self.common_user_set)


    def get_df(self):
        def parse(path):
            g = gzip.open(path, 'rb')
            for l in g:
                yield eval(l)

        def get_raw_df(path):
            df = {}
            for i, d in tqdm(enumerate(parse(path)), ascii=True):
                if "" not in d.values():
                    df[i] = d
            df = pd.DataFrame.from_dict(df, orient='index')
            df = df[["reviewerID", "asin", "reviewText", "overall"]]
            return df

        csv_path_s = self.path_s.replace('.json.gz', '.csv')
        csv_path_t = self.path_t.replace('.json.gz', '.csv')

        if os.path.exists(csv_path_s) and os.path.exists(csv_path_t):
            df_s = pd.read_csv(csv_path_s)
            df_t = pd.read_csv(csv_path_t)
            print('Load raw data from %s.' % csv_path_s)
            print('Load raw data from %s.' % csv_path_t)
        else:
            df_s = get_raw_df(self.path_s)
            df_t = get_raw_df(self.path_t)

            df_s.to_csv(csv_path_s, index=False)
            df_t.to_csv(csv_path_t, index=False)
            print('Build raw data to %s.' % csv_path_s)
            print('Build raw data to %s.' % csv_path_t)

        return df_s, df_t


    def filterout(self, df_s, df_t, thre_i, thre_u):
        index_s = df_s[["overall", "asin"]].groupby('asin').count() >= thre_i
        index_t = df_t[["overall", "asin"]].groupby('asin').count() >= 30
        item_s = set(index_s[index_s['overall'] == True].index)
        item_t = set(index_t[index_t['overall'] == True].index)
        df_s = df_s[df_s['asin'].isin(item_s)]
        df_t = df_t[df_t['asin'].isin(item_t)]

        index_s = df_s[["overall", "reviewerID"]].groupby('reviewerID').count() >= thre_u
        index_t = df_t[["overall", "reviewerID"]].groupby('reviewerID').count() >= thre_u
        user_s = set(index_s[index_s['overall'] == True].index)
        user_t = set(index_t[index_t['overall'] == True].index)
        df_s = df_s[df_s['reviewerID'].isin(user_s)]
        df_t = df_t[df_t['reviewerID'].isin(user_t)]

        return df_s, df_t


    def convert_idx(self):
        uiterator = count(0)
        udict = defaultdict(lambda: next(uiterator))
        [udict[user] for user in self.df_s["reviewerID"].tolist()+ self.df_t["reviewerID"].tolist()]
        iiterator_s = count(0)
        idict_s = defaultdict(lambda: next(iiterator_s))
        [idict_s[item] for item in self.df_s["asin"]]
        iiterator_t = count(0)
        idict_t = defaultdict(lambda: next(iiterator_t))
        [idict_t[item] for item in self.df_t["asin"]]

        self.df_s['uid'] = self.df_s['reviewerID'].map(lambda x: udict[x])
        self.df_t['uid'] = self.df_t['reviewerID'].map(lambda x: udict[x])
        self.df_s['iid'] = self.df_s['asin'].map(lambda x: idict_s[x])
        self.df_t['iid'] = self.df_t['asin'].map(lambda x: idict_t[x])

        user_set_s = set(self.df_s['uid'])
        item_set_s = set(self.df_s['iid'])
        user_set_t = set(self.df_t['uid'])
        item_set_t = set(self.df_t['iid'])
        overlap_user_set = user_set_s & user_set_t
        all_user_set = user_set_s | user_set_t

        assert len(item_set_s) == len(idict_s)
        assert len(item_set_t) == len(idict_t)

        self.user_num_s, self.item_num_s, self.user_num_t, self.item_num_t, self.overlap_num_user, self.user_num = \
            len(user_set_s), len(item_set_s), len(user_set_t), len(item_set_t), len(overlap_user_set), len(all_user_set)

        print('Source domain users %d, items %d, ratings %d.' % (self.user_num_s, self.item_num_s, len(self.df_s)))
        print('Target domain users %d, items %d, ratings %d.' % (self.user_num_t, self.item_num_t, len(self.df_t)))
        print('Overlapping users %d (%.3f%%, %.3f%%).' % (
            self.overlap_num_user, self.overlap_num_user/self.user_num_s*100, self.overlap_num_user/self.user_num_t*100))

        return dict(udict), dict(idict_s), dict(idict_t), overlap_user_set


    def split_train_test(self):
        random.seed(666)  # 这里的seed保证cold-start users的相同
        coldstart_user_set = set(random.sample(self.overlap_user_set, int(self.overlap_num_user*0.5)))
        common_user_all_set = self.overlap_user_set-coldstart_user_set
        random.seed(2019)
        common_user_set = set(random.sample(common_user_all_set, int(len(common_user_all_set)*self.ratio)))
        train_common_s = self.df_s[self.df_s['uid'].isin(common_user_set)]
        train_common_t = self.df_t[self.df_t['uid'].isin(common_user_set)]
        train_s = self.df_s[self.df_s['uid'].isin(common_user_all_set-common_user_set).apply(lambda x: not x)]
        train_t = self.df_t[self.df_t['uid'].isin((common_user_all_set-common_user_set)|coldstart_user_set).apply(lambda x: not x)]
        train_cs_s = self.df_s[self.df_s['uid'].isin(coldstart_user_set)]

        random.seed(2019)
        coldstart_user_vali_set = set(random.sample(coldstart_user_set, int(len(coldstart_user_set)*0.4)))
        coldstart_user_test_set = coldstart_user_set - coldstart_user_vali_set

        vali = self.df_t[self.df_t['uid'].isin(coldstart_user_vali_set)]
        test = self.df_t[self.df_t['uid'].isin(coldstart_user_test_set)]

        print('Ratio %.2f, common users %d, vali cold-start users %d, test cold-start users %d.' %
              (self.ratio, len(common_user_set), len(coldstart_user_vali_set), len(coldstart_user_test_set)))

        return coldstart_user_set, common_user_set, train_common_s, train_common_t, train_s, train_t, train_cs_s, \
               vali, test


    def get_w2v(self):
        all_text = self.df_s['reviewText'].tolist()
        all_text.extend(self.df_t['reviewText'].tolist())
        vectorizer = TfidfVectorizer(max_df=0.5, stop_words={'english'}, max_features=20000)
        tfidf = vectorizer.fit_transform(all_text)
        vocab_dict = vectorizer.vocabulary_

        word_embedding = np.zeros([len(vocab_dict) + 1, 300])

        if not args.info:
            print('Building word embedding matrix...')
            for word, idx in vocab_dict.items():
                if word in google_vocab:
                    word_embedding[idx] = google_model.vectors[google_vocab[word].index]

        return vocab_dict, word_embedding


    def get_documents(self, df, user_set):
        reviews = [list(map(lambda x: self.vocab_dict.get(x, -1), review.split(' '))) for review in df['reviewText']]
        reviews = [np.array(review)[np.array(review) != -1].tolist() for review in reviews]
        df = df.copy()
        df['review_idx'] = reviews
        docu_udict, cut_docu_udict = defaultdict(list), defaultdict(list)
        docu_idict, cut_docu_idict = defaultdict(list), defaultdict(list)
        auxiliary_docu_udict, cut_auxiliary_docu_udict = defaultdict(list), defaultdict(list)
        for user, item, review in zip(df['uid'], df['iid'], df['review_idx']):
            docu_udict[user].extend(review)
            docu_idict[item].extend(review)

        print('Constructing auxiliary documents...')
        df_aux = df[~df['uid'].isin(user_set)]
        for idx in tqdm(df_aux.index, ascii=True):
            row = df_aux[['uid', 'iid', 'overall', 'review_idx']].loc[idx]
            user, item, rating, review = row[0], row[1], row[2], row[3]

            exact_matches = df_aux[(df_aux['iid']==item) & (df_aux['overall']==rating)]
            if not exact_matches.empty:
                auxiliary_docu_udict[user].extend(exact_matches.sample(1)['review_idx'].to_list()[0])
                continue
            elif not df_aux[(df_aux['iid']==item) & (df_aux['overall']==rating+1)].empty:
                up_matches = df_aux[(df_aux['iid']==item) & (df_aux['overall']==rating+1)]
                auxiliary_docu_udict[user].extend(up_matches.sample(1)['review_idx'].to_list()[0])
                continue
            else:
                down_matches = df_aux[(df_aux['iid']==item) & (df_aux['overall']==rating-1)]
                auxiliary_docu_udict[user].extend(down_matches.sample(1)['review_idx'].to_list()[0])

        max_length = 500
        for u, docu in docu_udict.items():
            docu_cut = np.array(docu)[:max_length]
            cut_docu_udict[u] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                       'constant', constant_values=(0, -1))
        for i, docu in docu_idict.items():
            docu_cut = np.array(docu)[:max_length]
            cut_docu_idict[i] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                       'constant', constant_values=(0, -1))
        for u, docu in auxiliary_docu_udict.items():
            docu_cut = np.array(docu)[:max_length]
            cut_auxiliary_docu_udict[u] = np.pad(docu_cut, (0, max_length - docu_cut.shape[0]),
                                                 'constant', constant_values=(0, -1))

        return cut_docu_udict, cut_docu_idict, cut_auxiliary_docu_udict


    def dump_pkl(self):
        def extract_ratings(df):
            ratings = df.apply(lambda x:(x['uid'], x['iid'], x['overall']), axis=1).tolist()
            return ratings

        pkl_path = self.path_s.replace(self.path_s.split('/')[-1], 'crossdata_i%d_u%d_%.2f.pkl' %
                                       (self.thre_i, self.thre_u, self.ratio))
        with open(pkl_path, 'wb') as f:
            data = [self.udict, self.idict_s, self.idict_t, self.coldstart_user_set, self.common_user_set,
                    self.user_num, self.item_num_s, self.item_num_t,
                    extract_ratings(self.train_common_s), extract_ratings(self.train_common_t),
                    extract_ratings(self.train_cs_s), extract_ratings(self.vali), extract_ratings(self.test),
                    extract_ratings(self.train_s), extract_ratings(self.train_t),
                    self.vocab_dict, self.word_embedding,
                    self.docu_udict_s, self.docu_idict_s, self.docu_udict_t, self.docu_idict_t,
                    self.auxiliary_docu_udict_s, self.auxiliary_docu_udict_t
                    ]
            pickle.dump(data, f)
            print('Build data to %s.' % pkl_path)
            print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--info', type=bool, default=True)
    args = parser.parse_args()

    if not args.info:
        print('Loading GoogleNews w2v model...')
        google_model = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.vector', binary=False)
        google_vocab = google_model.vocab

    CrossData('book2movie/reviews_Books_5.json.gz', 'book2movie/reviews_Movies_and_TV_5.json.gz',
              ratio=args.ratio, thre_i=30, thre_u=10).dump_pkl()
    CrossData('movie2music/reviews_Movies_and_TV_5.json.gz', 'movie2music/reviews_CDs_and_Vinyl_5.json.gz',
              ratio=args.ratio, thre_i=30, thre_u=10).dump_pkl()
    CrossData('book2music/reviews_Books_5.json.gz', 'book2music/reviews_CDs_and_Vinyl_5.json.gz',
              ratio=args.ratio, thre_i=30, thre_u=10).dump_pkl()