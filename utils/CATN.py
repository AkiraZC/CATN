import tensorflow as tf
import numpy as np
import random, os, math
from time import time
from tqdm import tqdm
from decimal import Decimal
from operator import itemgetter
from collections import Counter, defaultdict
from scipy.sparse.csr import csr_matrix


class CATN:
    def __init__(self, data, item_ave_rating_s, item_ave_rating_t, reviews, args, pkl_path):
        self.coldstart_user_set, self.common_user_set, self.num_users, self.num_items_s, self.num_items_t, \
            self.train_common_s, self.train_common_t, _, self.vali, self.test, self.train_s, self.train_t,\
            self.vocab_dict, self.word_embeddings, _, _, _, _, _, _ = data
        self.item_ave_rating_s, self.item_ave_rating_t = item_ave_rating_s, item_ave_rating_t
        self.user_reviews_s, self.user_reviews_mask_s, self.item_reviews_s, self.item_reviews_mask_s, \
            self.user_reviews_t, self.user_reviews_mask_t, self.item_reviews_t, self.item_reviews_mask_t, \
            self.aux_user_reviews_s, self.aux_user_reviews_mask_s, \
            self.aux_user_reviews_t, self.aux_user_reviews_mask_t = reviews
        self.args, self.pkl_path = args, pkl_path

        self.vocab = {v : k for k, v in self.vocab_dict.items()}
        self.tmp_likelihood = tf.constant(0, dtype=tf.float32, name='tmp_likelihood')
        self.word_embeddings = tf.Variable(self.word_embeddings, dtype=tf.float32, name='pre_word_embeddings')

        self.get_placeholder()
        self.inference()


    def get_placeholder(self):
        self.domain_ph = tf.placeholder(tf.bool)
        self.dropout_rate_ph = tf.placeholder(tf.float32)
        self.users_ph = tf.placeholder(tf.int32, shape=[None], name='users')
        self.items_ph = tf.placeholder(tf.int32, shape=[None], name='items')
        self.ratings_ph = tf.placeholder(tf.float32, shape=[None], name='ratings')


    def inference(self):
        user_bias = tf.Variable(tf.truncated_normal([self.num_users], mean=0, stddev=0.1), name='user_bias')
        item_bias_s = tf.Variable(self.item_ave_rating_s, trainable=True, name='item_bias_s')
        item_bias_t = tf.Variable(self.item_ave_rating_t, trainable=True, name='item_bias_t')
        if self.args.score_way == 'simple' or self.args.score_way == 'share':
            param_s = self.init_aspect_extraction('share')
            param_t = param_s
        else:
            param_s = self.init_aspect_extraction('source')
            param_t = self.init_aspect_extraction('target')

        asp_vs_s, asp_vs_t = [], []
        for num in range(self.args.num_aspects):
            asp_vs_s.append(tf.Variable(tf.truncated_normal([self.args.latent_dim, 1], stddev=0.1),
                                            name='asp_v_s_%d' % num))
            asp_vs_t.append(tf.Variable(tf.truncated_normal([self.args.latent_dim, 1], stddev=0.1),
                                            name='asp_v_t_%d' % num))

        self.convert_mtrx = tf.Variable(tf.truncated_normal([self.args.latent_dim, self.args.latent_dim], stddev=0.3),
                                     name='convert_mtrx')
        self.tmp_likelihood = self.tmp_likelihood + tf.nn.l2_loss(self.convert_mtrx)

        item_bias, asp_vs_u, asp_vs_i, param, user_reviews, user_reviews_mask, item_reviews, item_reviews_mask,\
            aux_user_reviews, aux_user_reviews_mask, convert_mtrx =  tf.cond(self.domain_ph,
                lambda: (item_bias_s, asp_vs_t, asp_vs_s, param_s,
                         self.user_reviews_t, self.user_reviews_mask_t,
                         self.item_reviews_s, self.item_reviews_mask_s,
                         self.aux_user_reviews_t, self.aux_user_reviews_mask_t, self.convert_mtrx),
                lambda: (item_bias_t, asp_vs_s, asp_vs_t, param_t,
                         self.user_reviews_s, self.user_reviews_mask_s,
                         self.item_reviews_t, self.item_reviews_mask_t,
                         self.aux_user_reviews_s, self.aux_user_reviews_mask_s, tf.transpose(self.convert_mtrx))
        )

        self.user_bs = tf.gather(user_bias, self.users_ph)
        self.item_bs = tf.gather(item_bias, self.items_ph)

        self.user_inputs = tf.nn.embedding_lookup(user_reviews, self.users_ph)
        self.user_inputs_mask = tf.expand_dims(tf.nn.embedding_lookup(user_reviews_mask, self.users_ph), -1)
        self.user_reviews_repr = tf.nn.embedding_lookup(self.word_embeddings, self.user_inputs)*self.user_inputs_mask

        self.aux_user_inputs = tf.nn.embedding_lookup(aux_user_reviews, self.users_ph)
        self.aux_user_inputs_mask = tf.expand_dims(tf.nn.embedding_lookup(aux_user_reviews_mask, self.users_ph), -1)
        self.aux_user_reviews_repr = tf.nn.embedding_lookup(self.word_embeddings, self.aux_user_inputs)*self.aux_user_inputs_mask

        self.item_inputs = tf.nn.embedding_lookup(item_reviews, self.items_ph)
        self.item_inputs_mask = tf.expand_dims(tf.nn.embedding_lookup(item_reviews_mask, self.items_ph), -1)
        self.item_reviews_repr = tf.nn.embedding_lookup(self.word_embeddings, self.item_inputs) * self.item_inputs_mask

        v_u = tf.concat(asp_vs_u, axis=-1)
        v_i = tf.concat(asp_vs_i, axis=-1)
        tmp_S = tf.matmul(v_u, convert_mtrx, transpose_a=True)
        self.S_attn = tf.nn.leaky_relu(tf.matmul(tmp_S, v_i))

        self.user_asp_reprs_concat, self.aux_user_asp_reprs_concat, self.item_asp_reprs_concat, \
            self.user_attn_concat, self.aux_user_attns_concat, self.item_attn_concat = \
            self.aspects_extraction(param, asp_vs_u, asp_vs_i)

        if self.args.score_way == 'simple':
            loss_ratings = self.simple_weighting(convert_mtrx)
        elif self.args.score_way == 'share' or self.args.score_way == 'doubleS':
            loss_ratings = self.doubleS_weighting(convert_mtrx)
        elif self.args.score_way == 'aux_doubleS':
            loss_ratings = self.aux_doubleS_weighting(convert_mtrx)
        else:
            print('Invalid option!')
            exit()

        self.loss = loss_ratings + self.args.regulazation_rate * self.tmp_likelihood
        self.train_op = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)


    def init_aspect_extraction(self, name):
        with tf.name_scope(name):
            W_conv = tf.Variable(tf.truncated_normal([self.args.window_size, 300, 1, self.args.num_filters],
                                                     stddev=0.1), name='pre_W_conv')
            b_conv = tf.Variable(tf.constant(0.1, shape=[self.args.num_filters]), name='pre_b_conv')

            self.tmp_likelihood = tf.reduce_sum([self.tmp_likelihood, tf.nn.l2_loss(W_conv), tf.nn.l2_loss(b_conv)])

            W_conv_aux = tf.Variable(tf.truncated_normal(
                [self.args.window_size, self.args.num_filters, 1, self.args.num_filters], stddev=0.1), name='W_conv_aux')
            b_conv_aux = tf.Variable(tf.constant(0.1, shape=[self.args.num_filters]), name='b_conv_aux')

            if self.args.score_way == 'aux_doubleS':
                self.tmp_likelihood = tf.reduce_sum(
                    [self.tmp_likelihood, tf.nn.l2_loss(W_conv_aux), tf.nn.l2_loss(b_conv_aux)])

            W_gate, b_gate, W_prj, b_prj, = {}, {}, {}, {}
            for num in range(self.args.num_aspects):
                W_gate[num] = tf.Variable(tf.truncated_normal([self.args.num_filters, self.args.latent_dim],stddev=0.1),
                                          name='pre_W_gate_%d' % num)
                b_gate[num] = tf.Variable(tf.constant(0.1, shape=[self.args.latent_dim]),
                                          name="pre_b_gate_%d" % num)
                W_prj[num] = tf.Variable(tf.truncated_normal([self.args.num_filters, self.args.latent_dim], stddev=0.1),
                                         name='pre_W_prj_%d' % num)
                b_prj[num] = tf.Variable(tf.truncated_normal([self.args.latent_dim], stddev=0.1),
                                         name='pre_b_prj_%d' % num)
                self.tmp_likelihood = tf.reduce_sum( [self.tmp_likelihood,
                                                      tf.nn.l2_loss(W_gate[num]), tf.nn.l2_loss(b_gate[num]),
                                                      tf.nn.l2_loss(W_prj[num]), tf.nn.l2_loss(b_prj[num])])

        return (W_conv, b_conv, W_conv_aux, b_conv_aux, W_gate, b_gate, W_prj, b_prj)


    def aspects_extraction(self, param, asp_vs_u, asp_vs_i):
        W_conv, b_conv, W_conv_aux, b_conv_aux, W_gate, b_gate, W_prj, b_prj = param

        item_asp_reprs, user_asp_reprs, aux_user_asp_reprs = [], [], []
        item_attns, user_attns, aux_user_attns = [], [], []
        for num in range(self.args.num_aspects):
            param_i = W_conv, b_conv, W_gate[num], b_gate[num], W_prj[num], b_prj[num], asp_vs_i[num]
            param_u = W_conv, b_conv, W_gate[num], b_gate[num], W_prj[num], b_prj[num], asp_vs_u[num]
            param_u_aux = W_conv, b_conv, W_conv_aux, b_conv_aux, W_gate[num], b_gate[num], W_prj[num], b_prj[num],\
                          asp_vs_u[num]

            item_attn, item_asp_repr = self.get_gate_asp(self.item_reviews_repr, self.item_inputs_mask, param_i)
            item_asp_reprs.append(item_asp_repr)
            item_attns.append(item_attn)

            user_attn, user_asp_repr = self.get_gate_asp(self.user_reviews_repr, self.user_inputs_mask, param_u)
            user_asp_reprs.append(user_asp_repr)
            user_attns.append(user_attn)

            aux_user_attn, aux_user_asp_repr = self.get_aux_gate_asp(
                                        self.aux_user_reviews_repr, self.aux_user_inputs_mask, param_u_aux)
            aux_user_asp_reprs.append(aux_user_asp_repr)
            aux_user_attns.append(aux_user_attn)

        user_asp_reprs_concat = tf.concat(user_asp_reprs, axis=1)
        aux_user_asp_reprs_concat = tf.concat(aux_user_asp_reprs, axis=1)
        item_asp_reprs_concat = tf.concat(item_asp_reprs, axis=1)
        user_attn_concat = tf.concat(user_attns, axis=-1)
        aux_user_attns_concat = tf.concat(aux_user_attns, axis=-1)
        item_attn_concat = tf.concat(item_attns, axis=-1)

        return user_asp_reprs_concat, aux_user_asp_reprs_concat, item_asp_reprs_concat,\
               user_attn_concat, aux_user_attns_concat, item_attn_concat


    def get_gate_asp(self, reviews_repr, inputs_mask, param):
        W_conv, b_conv, W_gate, b_gate, W_prj, b_prj, asp_v = param

        # conv layer
        reviews_repr_expand = tf.expand_dims(reviews_repr, -1)
        conv = tf.nn.conv2d(reviews_repr_expand, W_conv, strides=[1, 1, 300, 1], padding='SAME') + b_conv
        c = tf.squeeze(tf.nn.relu(conv), -2)
        c_reshape = tf.reshape(c, [-1, self.args.num_filters])

        # gate machenism
        gate_words_embed = tf.reshape(
            tf.nn.sigmoid(tf.matmul(c_reshape, W_gate) + b_gate), [-1, self.args.docu_length, self.args.latent_dim])
        prj_words_embed = tf.reshape(
            tf.matmul(c_reshape, W_prj) + b_prj, [-1, self.args.docu_length, self.args.latent_dim])
        gate_asp_embed = gate_words_embed * prj_words_embed

        # attention
        # v_uniform = tf.expand_dims(tf.reduce_mean(gate_words_embed, axis=1),axis=-1)
        # attn = tf.nn.softmax(tf.matmul(gate_asp_embed*inputs_mask, v_uniform), axis=1)

        if self.args.score_way == 'simple':
            attn = tf.constant(np.ones([self.args.docu_length, 1]) * (1/self.args.docu_length), dtype=tf.float32)
            asp_repr = tf.nn.dropout(tf.reduce_sum(gate_asp_embed*inputs_mask*attn, 1, True), self.dropout_rate_ph)
        else:
            attn = tf.expand_dims(tf.nn.softmax(tf.reshape(tf.matmul(
                tf.reshape(gate_asp_embed*inputs_mask, [-1, self.args.latent_dim]), asp_v), [-1, self.args.docu_length]),
                axis=-1), axis=-1)
            asp_repr = tf.nn.dropout(tf.reduce_sum(gate_asp_embed*inputs_mask*attn, 1, True), self.dropout_rate_ph)

        return attn, asp_repr


    def get_aux_gate_asp(self, reviews_repr, inputs_mask, param):
        W_conv, b_conv, W_conv_aux, b_conv_aux, W_gate, b_gate, W_prj, b_prj, asp_v = param

        # conv layer
        reviews_repr_expand = tf.expand_dims(reviews_repr, -1)
        conv = tf.nn.conv2d(reviews_repr_expand, W_conv, strides=[1, 1, 300, 1], padding='SAME') + b_conv
        c = tf.squeeze(tf.nn.relu(conv), -2)

        conv_abs = tf.nn.conv2d(tf.expand_dims(c,-1), W_conv_aux, strides=[1,1,self.args.num_filters,1], padding='SAME') \
                   + b_conv_aux
        c_abs = tf.squeeze(tf.nn.relu(conv_abs), -2)

        c_reshape = tf.reshape(c_abs, [-1, self.args.num_filters])

        # gate machenism
        gate_words_embed = tf.reshape(
            tf.nn.sigmoid(tf.matmul(c_reshape, W_gate) + b_gate), [-1, self.args.docu_length, self.args.latent_dim])
        prj_words_embed = tf.reshape(
            tf.matmul(c_reshape, W_prj) + b_prj, [-1, self.args.docu_length, self.args.latent_dim])
        gate_asp_embed = gate_words_embed * prj_words_embed

        # attention
        # v_uniform = tf.expand_dims(tf.reduce_mean(gate_words_embed, axis=1),axis=-1)
        # attn = tf.nn.softmax(tf.matmul(gate_asp_embed*inputs_mask, v_uniform), axis=1)

        attn = tf.expand_dims(tf.nn.softmax(tf.reshape(tf.matmul(
            tf.reshape(gate_asp_embed*inputs_mask, [-1, self.args.latent_dim]), asp_v), [-1, self.args.docu_length]),
            axis=-1), axis=-1)
        asp_repr = tf.nn.dropout(tf.reduce_sum(gate_asp_embed*inputs_mask*attn, 1, True), self.dropout_rate_ph)

        return attn, asp_repr


    def simple_weighting(self, convert_mtrx):
        tmp_S = tf.reshape(tf.matmul(tf.reshape(self.user_asp_reprs_concat, [-1, self.args.latent_dim]), convert_mtrx),
                           [-1, self.args.num_aspects, self.args.latent_dim])
        logits = tf.reduce_mean(tf.matmul(tmp_S, tf.transpose(self.item_asp_reprs_concat, perm=[0, 2, 1])))
        # logits = tf.reduce_mean(tf.matmul(
        #     self.user_asp_reprs_concat, tf.transpose(self.item_asp_reprs_concat, [0, 2, 1])), axis=[1, 2])
        self.predict_ratings = logits + self.user_bs + self.item_bs
        loss_ratings = tf.reduce_mean(tf.squared_difference(self.ratings_ph, self.predict_ratings))

        return loss_ratings


    def doubleS_weighting(self, convert_mtrx):
        tmp_S = tf.reshape(tf.matmul(tf.reshape(self.user_asp_reprs_concat, [-1, self.args.latent_dim]), convert_mtrx),
                           [-1, self.args.num_aspects, self.args.latent_dim])
        self.S = tf.matmul(tmp_S, tf.transpose(self.item_asp_reprs_concat, perm=[0, 2, 1]))
        # self.S = tf.matmul(self.user_asp_reprs_concat, tf.transpose(self.item_asp_reprs_concat, [0, 2, 1]))
        logits = tf.reduce_mean(self.S * self.S_attn, axis=[1,2])
        self.predict_ratings = logits + self.user_bs + self.item_bs
        loss_ratings = tf.reduce_mean(tf.squared_difference(self.ratings_ph, self.predict_ratings))

        return loss_ratings


    def aux_doubleS_weighting(self, convert_mtrx):
        aux_merge = tf.concat([
            self.aux_user_asp_reprs_concat - self.user_asp_reprs_concat,
            self.aux_user_asp_reprs_concat * self.user_asp_reprs_concat], axis=-1)
        W_gate = tf.Variable(tf.truncated_normal([2*self.args.latent_dim, self.args.latent_dim], stddev=0.1))
        b_gate = tf.Variable(tf.constant(0.1, shape=[self.args.latent_dim]))
        aux_gate = tf.reshape(tf.nn.sigmoid(tf.matmul(
            tf.reshape(aux_merge, [-1, 2*self.args.latent_dim]), W_gate) + b_gate),
            [-1, self.args.num_aspects, self.args.latent_dim])
        aux_reprs = aux_gate * self.aux_user_asp_reprs_concat

        W_merge = tf.Variable(tf.truncated_normal([2 * self.args.latent_dim, self.args.latent_dim], stddev=0.1))
        b_merge = tf.Variable(tf.constant(0.1, shape=[self.args.latent_dim]))
        user_asp_reprs = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(
            tf.concat([aux_reprs, self.user_asp_reprs_concat], axis=-1),
            [-1, 2 * self.args.latent_dim]), W_merge) + b_merge), [-1, self.args.num_aspects, self.args.latent_dim])

        self.tmp_likelihood = tf.reduce_sum([self.tmp_likelihood, tf.nn.l2_loss(W_gate), tf.nn.l2_loss(b_gate),
                                             tf.nn.l2_loss(W_merge), tf.nn.l2_loss(b_merge)])

        tmp_S = tf.reshape(tf.matmul(tf.reshape(user_asp_reprs, [-1, self.args.latent_dim]), convert_mtrx),
                           [-1, self.args.num_aspects, self.args.latent_dim])
        self.S = tf.matmul(tmp_S, tf.transpose(self.item_asp_reprs_concat, perm=[0, 2, 1]))
        # self.S = tf.matmul(user_asp_reprs, tf.transpose(self.item_asp_reprs_concat, [0, 2, 1]))
        logits = tf.reduce_mean(self.S * self.S_attn, axis=[1, 2])
        self.predict_ratings = logits + self.user_bs + self.item_bs
        loss_ratings = tf.reduce_mean(tf.squared_difference(self.ratings_ph, self.predict_ratings))

        return loss_ratings


    def train_step(self, sess):
        sess.run(tf.global_variables_initializer())
        print('Start training...')
        vali_mse_list, test_mse_list = [], []
        st_ratio = len(self.train_common_s) / (len(self.train_common_s) + len(self.train_common_t))
        for e in range(self.args.max_epoches):
            # user_aspects_words, item_aspects_words, rating, S = self.case_word_attn(sess, True, 0)
            # user_aspects_words_s, user_aspects_words_t = self.user_word_attn(sess, 0)
            t = time()

            random.shuffle(self.train_common_s)
            random.shuffle(self.train_common_t)
            train_users_list_s, train_items_list_s, train_ratings_list_s = zip(*self.train_common_s)
            train_users_list_t, train_items_list_t, train_ratings_list_t = zip(*self.train_common_t)

            batch_size = self.args.batch_size
            batch_size_s = math.ceil(batch_size * st_ratio)
            batch_size_t = batch_size - batch_size_s
            num_train_batches = math.ceil((len(self.train_common_s) + len(self.train_common_t)) / batch_size)
            loss_total = 0.0

            for batch_idx in tqdm(range(num_train_batches), ascii=True):
                users_batch_s = train_users_list_s[batch_idx * batch_size_s: (batch_idx + 1) * batch_size_s]
                items_batch_s = train_items_list_s[batch_idx * batch_size_s: (batch_idx + 1) * batch_size_s]
                ratings_batch_s = train_ratings_list_s[batch_idx * batch_size_s: (batch_idx + 1) * batch_size_s]

                if not users_batch_s:
                    continue

                _, loss_val_s = sess.run(
                    [self.train_op, self.loss], feed_dict={
                    self.domain_ph: True,
                    self.dropout_rate_ph: self.args.dropout_rate,
                    self.users_ph: users_batch_s,
                    self.items_ph: items_batch_s,
                    self.ratings_ph: ratings_batch_s,
                })

                users_batch_t = train_users_list_t[batch_idx * batch_size_t: (batch_idx + 1) * batch_size_t]
                items_batch_t = train_items_list_t[batch_idx * batch_size_t: (batch_idx + 1) * batch_size_t]
                ratings_batch_t = train_ratings_list_t[batch_idx * batch_size_t: (batch_idx + 1) * batch_size_t]

                if not users_batch_t:
                    continue

                _, loss_val_t = sess.run(
                    [self.train_op, self.loss], feed_dict={
                    self.domain_ph: False,
                    self.dropout_rate_ph: self.args.dropout_rate,
                    self.users_ph: users_batch_t,
                    self.items_ph: items_batch_t,
                    self.ratings_ph: ratings_batch_t,
                })

                if not math.isnan(loss_val_s):
                    loss_total += loss_val_s
                if not math.isnan(loss_val_t):
                    loss_total += loss_val_t

            vali_mse, vali_mae = self.eval(sess, self.vali)
            test_mse, test_mae = self.eval(sess, self.test)
            print("Epoch %d, time: %.2fs, loss: %.3f, vali MSE: %.4f; test MSE: %.3f." % (
                e, time() - t, loss_total / num_train_batches, vali_mse, test_mse))

            vali_mse_list.append(round(Decimal(vali_mse), 4))
            test_mse_list.append(round(Decimal(test_mse), 4))

            if e >= 4 and vali_mse_list.index(min(vali_mse_list)) <= e-4:
                break

        best_vali_index = vali_mse_list.index(min(vali_mse_list))
        print('Finish training, epoch %d, test MSE: %.3f; best test MSE: %.3f.'
              % (best_vali_index, test_mse_list[best_vali_index], min(test_mse_list)))


    def eval(self, sess, data):
        test_users_list, test_items_list, test_ratings_list = zip(*data)
        mse, mae = [], []
        num_test_batches = math.ceil(len(data)/1024)
        for batch_idx in range(num_test_batches):
            users_batch = test_users_list[batch_idx * 1024: (batch_idx + 1) * 1024]
            items_batch = test_items_list[batch_idx * 1024: (batch_idx + 1) * 1024]
            ratings_batch = test_ratings_list[batch_idx * 1024: (batch_idx + 1) * 1024]

            predict_ratings = sess.run(self.predict_ratings, feed_dict={
                self.domain_ph: False,
                self.dropout_rate_ph: 1.0,
                self.users_ph: users_batch,
                self.items_ph: items_batch
            })
            mse.append(np.square(np.array(predict_ratings) - np.array(ratings_batch)).mean())
            mae.append(np.abs(np.array(predict_ratings) - np.array(ratings_batch)).mean())

        return np.array(mse).mean(), np.array(mae).mean()


    def case_word_attn(self, sess, domain_bool, idx):
        if domain_bool == True:
            data = self.train_common_s
        else:
            data = self.train_common_t

        train_users_list, train_items_list, train_ratings_list = zip(*data)
        user_input, item_input, user_attn_concat, item_attn_concat, S = sess.run(
            [self.user_inputs, self.item_inputs, self.user_attn_concat, self.item_attn_concat, self.S],
            feed_dict={
                self.domain_ph: domain_bool,
                self.dropout_rate_ph: 1.0,
                self.users_ph: train_users_list[idx:idx+1],
                self.items_ph: train_items_list[idx:idx+1]
            }
        )
        user_input, item_input, user_attn_concat, item_attn_concat = \
            np.squeeze(user_input), np.squeeze(item_input), np.squeeze(user_attn_concat), np.squeeze(item_attn_concat)

        user_aspects_words = self.calcu_word_attn(user_input, user_attn_concat)
        item_aspects_words = self.calcu_word_attn(item_input, item_attn_concat)

        return user_aspects_words, item_aspects_words, train_ratings_list[idx:idx+1], S


    def user_word_attn(self, sess, idx):
        user = list(self.common_user_set)[idx]
        user_input_s, user_attn_concat_s, aux_input_s, aux_user_attn_concat_s = sess.run(
            [self.user_inputs, self.user_attn_concat, self.aux_user_inputs, self.aux_user_attns_concat], feed_dict={
                self.domain_ph: False,
                self.dropout_rate_ph: 1.0,
                self.users_ph: [user],
            }
        )
        user_input_t, user_attn_concat_t = sess.run(
            [self.user_inputs, self.user_attn_concat], feed_dict={
                self.domain_ph: True,
                self.dropout_rate_ph: 1.0,
                self.users_ph: [user],
            }
        )
        user_input_s, user_attn_concat_s, user_input_t, user_attn_concat_t, aux_input_s, aux_user_attn_concat_s = \
            np.squeeze(user_input_s), np.squeeze(user_attn_concat_s), \
            np.squeeze(user_input_t), np.squeeze(user_attn_concat_t), \
            np.squeeze(aux_input_s), np.squeeze(aux_user_attn_concat_s)

        user_aspects_words_s = self.calcu_word_attn(user_input_s, user_attn_concat_s)
        user_aspects_words_t = self.calcu_word_attn(user_input_t, user_attn_concat_t)
        aux_aspects_words_s = self.calcu_word_attn(aux_input_s, aux_user_attn_concat_s)

        return user_aspects_words_s, user_aspects_words_t, aux_aspects_words_s


    def calcu_word_attn(self, input, attn_concat, topn=10):
        words_attn_list = []
        vocab_list = list(set(input))
        if -1 in vocab_list: vocab_list.remove(-1)
        for a in range(self.args.num_aspects):
            words_attn_list.extend(list(zip(input.tolist(), attn_concat[:, a].tolist(), [a] * self.args.docu_length)))
        aspect_vocab_value = np.zeros([len(vocab_list), self.args.num_aspects], dtype=np.float32)
        aspect_vocab_count = np.ones([len(vocab_list), self.args.num_aspects], dtype=np.float32) * 1e-5
        for w, attn, a in words_attn_list:
            if w != -1:
                aspect_vocab_value[vocab_list.index(w), a] += attn
                aspect_vocab_count[vocab_list.index(w), a] += 1

        aspect_vocab = (aspect_vocab_value - np.mean(aspect_vocab_value, axis=1,
                                                     keepdims=True)) / aspect_vocab_count
        aspect_vocab_rank = np.argsort(aspect_vocab, axis=0)

        top_vocabs = []
        for a in range(self.args.num_aspects):
            top_vocabs_index = aspect_vocab_rank[::-1, a][:topn]
            top_vocabs.append(list(map(lambda x: self.vocab[vocab_list[x]], top_vocabs_index)))

        return top_vocabs