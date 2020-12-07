import tensorflow as tf
import pickle, os, sys
import numpy as np
from time import time
from collections import defaultdict
import pandas as pd
import argparse
sys.path.append('..')
from utils.CATN import CATN
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class CATN_RUNNER:
    def __init__(self, data):
        _, _, self.user_num, self.item_num_s, self.item_num_t, _, _, _, _, _, self.train_s, self.train_t, \
            self.vocab_dict, self.word_embedding, \
            self.docu_udict_s, self.docu_idict_s, self.docu_udict_t, self.docu_idict_t,\
            self.auxiliary_docu_udict_s, self.auxiliary_docu_udict_t = data

        self.data = data
        self.item_ave_rating_s = self.building_ave_item_rating(self.train_s, self.item_num_s)
        self.item_ave_rating_t = self.building_ave_item_rating(self.train_t, self.item_num_t)
        self.reviews = self.get_reviews()


    def get_reviews(self):
        def reviews2arr(reviews, num):
            Reviews = np.zeros([num, args.docu_length], dtype=np.int32)
            Reviews_mask = np.zeros_like(Reviews, dtype=np.float32)

            for _id, review in reviews.items():
                Reviews_mask[_id] = (review != -1)
                review = [20000 if w == -1 else w for w in review]
                review = np.array(review)
                Reviews[_id] = review

            return Reviews, Reviews_mask

        self.user_reviews_s, self.user_reviews_mask_s = reviews2arr(self.docu_udict_s, self.user_num)
        self.item_reviews_s, self.item_reviews_mask_s = reviews2arr(self.docu_idict_s, self.item_num_s)
        self.user_reviews_t, self.user_reviews_mask_t = reviews2arr(self.docu_udict_t, self.user_num)
        self.item_reviews_t, self.item_reviews_mask_t = reviews2arr(self.docu_idict_t, self.item_num_t)
        self.aux_user_reviews_s, self.aux_user_reviews_mask_s = reviews2arr(self.auxiliary_docu_udict_s, self.user_num)
        self.aux_user_reviews_t, self.aux_user_reviews_mask_t = reviews2arr(self.auxiliary_docu_udict_t, self.user_num)

        return (self.user_reviews_s, self.user_reviews_mask_s, self.item_reviews_s, self.item_reviews_mask_s,
                self.user_reviews_t, self.user_reviews_mask_t, self.item_reviews_t, self.item_reviews_mask_t,
                self.aux_user_reviews_s, self.aux_user_reviews_mask_s,
                self.aux_user_reviews_t, self.aux_user_reviews_mask_t)


    @staticmethod
    def building_ave_item_rating(train, item_num):
        item_ave_rating_dict = defaultdict(list)
        train_users_list, train_items_list, train_ratings_list = zip(*train)
        for item, rating in zip(train_items_list, train_ratings_list):
            item_ave_rating_dict[item].append(rating)

        item_ave_rating = np.zeros([item_num], dtype=np.float32)
        for item , rating_list in item_ave_rating_dict.items():
            item_ave_rating[item] = np.array(rating_list).mean()

        return item_ave_rating


    def step_train(self, sess):
        self.catn = CATN(data, self.item_ave_rating_s, self.item_ave_rating_t, self.reviews, args, pkl_path)
        self.catn.train_step(sess)


parser = argparse.ArgumentParser()
parser.add_argument('--pkl_idx', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--docu_length', type=int, default=500)
parser.add_argument('--num_filters', type=int, default=50)
parser.add_argument('--num_aspects', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--window_size', type=int, default=3)
parser.add_argument('--dropout_rate', type=float, default=0.8)
parser.add_argument('--regulazation_rate', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_epoches', type=int, default=200)
parser.add_argument('--ratio', type=float, default=0.2)
parser.add_argument('--score_way', type=str, choices=['simple','share','doubleS', 'aux_doubleS'],
                    default='aux_doubleS')
args = parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    print(args)
    pkl_paths = ['../dataset/book2movie/crossdata_i30_u10_%.2f.pkl',
                '../dataset/movie2music/crossdata_i30_u10_%.2f.pkl',
                '../dataset/book2music/crossdata_i30_u10_%.2f.pkl']
    pkl_path = pkl_paths[args.pkl_idx]

    firtime = time()
    with open(pkl_path % args.ratio, 'rb') as f:
        all_data = pickle.load(f)
    data = all_data[3:]
    udict, idict_s, idict_t = all_data[:3]
    print("Load data from %s, time: %.2fs." % (pkl_path % args.ratio, time() - firtime))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        catn_runner = CATN_RUNNER(data)
        catn_runner.step_train(sess)