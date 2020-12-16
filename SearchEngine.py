import json
import sqlite3
import random
import numpy as np
import math
import copy

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from my_suggestor import WordSuggestor
from my_suggestor import cosine_sim
import sys
sys.path.append('/Users/jiangpuhua/Documents/class/549/project/SteamSearchEngine/steam_search')
from config import config


def load_cache(cache_file_name):
    cache_file = open(cache_file_name)
    cache_file_content = cache_file.read()
    cache = json.loads(cache_file_content)
    cache_file.close()

    return cache


class SearchEngine(object):
    def __init__(self, config, tokenizer, stemmer, isStemming):
        self.config = config
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.isStemming = isStemming

        self.ann = load_cache(config.ANN_FILE)
        self.cn2eng_dic = load_cache(config.CN2ENG)

        self.corpus_idf = load_cache(config.CORPUS_IDF)
        self.corpus = [c.strip() for c in self.corpus_idf]
        self.tokenized_corpus = self.tokenize_corpus(self.corpus, self.tokenizer)

        self.queries = self.load_txt(config.QUERIES)
        self.tokenized_queries = self.tokenize_corpus(self.queries, self.tokenizer)

        self.mis_queries = self.load_txt(config.MIS_QUERIES)
        self.tokenized_mis_queries = self.tokenize_corpus(self.mis_queries, self.tokenizer)
        self.tokenized_suggested_queries = copy.deepcopy(self.tokenized_mis_queries)

        self.WORD_FREQUENCY = self.generate_word_freq(config.SPELLING_SUGGESTOR.WORD_FREQUENCY)
        self.STOP_WORD = self.load_txt(config.STOP_WORD)

        self.db_file = config.DATABASE_FILE
        self.BASIC_TABLE = config.BASIC_TABLE
        self.db = self.load_db()

        self.ranker = BM25Okapi(self.tokenized_corpus, k1=1.2, b=0.75)

    def load_db(self):
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()

        command = '''
          SELECT *
          FROM {}
        '''.format(self.BASIC_TABLE)
        cur.execute(command)
        results = list(cur.fetchall())

        db = {}
        for r in results:
            if r[0] not in db:
                db[r[0]] = list(r[1:])

            # print(r[0], r[1])
        return db

    def load_txt(self, file_path):
        content = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                content.append(line.strip())
        return content

    def generate_word_freq(self, file_path):
        word_freq = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                w, count = line.strip().split()
                word_freq[w] = int(count)
        return word_freq

    def tokenize_corpus(self, corpus, tokenizer):
        tokenized_corpus = [[] for _ in range(len(corpus))]
        for i, c in enumerate(corpus):
            tc = tokenizer(corpus[i])
            for w in tc:
                w = ''.join(list(filter(str.isalnum, w)))
                w = w.lower()
                if self.isStemming:
                    w = PS.stem(w)
                tokenized_corpus[i].append(w)
        return tokenized_corpus

    def search(self, query):
        query = query.lower()
        tokenized_query = self.tokenizer(query)

        num_results = config.RANKING.NUM_RESULTS

        doc_scores = np.array(self.ranker.get_scores(tokenized_query), dtype=np.float64)

        idx = doc_scores.argsort()[::-1][:num_results]

        results = []
        for i in idx:
            doc_id = int(self.corpus_idf[self.corpus[i]])
            results.append([doc_id] + self.db[doc_id])
        # for game in results:
        #     print(game[:3])
        return results

    def search_game(self, query, lamb=3):
        query = query.lower()
        tokenized_query = self.tokenizer(query)

        num_results = config.RANKING.NUM_RESULTS

        doc_scores = np.array(self.ranker.get_scores(tokenized_query), dtype=np.float64)
        cos_sim = []
        for i in range(len(doc_scores)):
            doc_id = int(self.corpus_idf[self.corpus[i]])
            game_name = self.db[doc_id][0]
            game_name = game_name.lower()
            cos_sim.append(cosine_sim(query, game_name))
        cos_sim = np.array(cos_sim, dtype=np.float64)
        scores = doc_scores + lamb * cos_sim

        idx = scores.argsort()[::-1][:num_results]
        results = []
        for i in idx:
            doc_id = int(self.corpus_idf[self.corpus[i]])
            results.append([doc_id] + self.db[doc_id])
        #print(results)
        # for game in results:
        #     print(game[:3])
        return results

    def cross_lang_search(self, query):
        results = []
        for cn_name in self.cn2eng_dic.keys():
            if query in cn_name:
                doc_id = int(self.cn2eng_dic[cn_name][0])
                results.append([doc_id] + self.db[doc_id])
        #print(results)
        return results

    def testBM25(self, mode):
        if mode == 1:
            queries = self.queries
            tokenized_queries = self.tokenized_mis_queries
        elif mode == 0:
            queries = self.queries
            tokenized_queries = self.tokenized_queries
        elif mode == 2:
            queries = self.queries
            tokenized_queries = self.tokenized_suggested_queries
        else:
            raise ValueError('Test mode mush be 0, 1 or 2!')

        print(tokenized_queries)
        query_scores, gt_scores = self.ranking(queries, tokenized_queries,
                                               self.corpus, self.corpus_idf, self.ann)
        for i, q in enumerate(queries):
            print(q, ':', self.getNDCG(query_scores[i], gt_scores[i]))

    def test_spell_suggestor(self):
        max_dist = config.SPELLING_SUGGESTOR.MAX_DIST
        word_scr = config.SPELLING_SUGGESTOR.WORD_SRC

        # t = create_triedict(word_scr)
        t = WordSuggestor(word_scr)

        for i in range(len(self.tokenized_suggested_queries) - 2):
            for j, w in enumerate(self.tokenized_suggested_queries[i]):
                if w not in self.WORD_FREQUENCY:
                    self.tokenized_suggested_queries[i][j] = self.get_suggested_word(t, max_dist, w)[0]
        # print(self.tokenized_suggested_queries)
        self.testBM25(mode=2)

    def get_suggested_word(self, t, max_dist, word, n=1):
        suggestions = t.suggest_words(word, max_dist)
        sim = []
        for w in suggestions:
            if w in self.WORD_FREQUENCY and w not in self.STOP_WORD:
                sim.append(cosine_sim(w, word) * math.log(1 + self.WORD_FREQUENCY[w]))
            else:
                sim.append(0)
        sim = np.array(sim)
        index = sim.argsort()[::-1]

        return [suggestions[index[i]] for i in range(min(len(index), n))]

    def ranking(self, queries, tokenized_queries, corpus, corpus_idf, ann):
        num_results = config.RANKING.NUM_RESULTS
        doc_scores = []
        docs = []
        idxes = []

        for tq in tokenized_queries:
            doc_scores.append(np.array(self.ranker.get_scores(tq), dtype=np.float64))
            docs.append(self.ranker.get_top_n(tq, corpus, n=num_results))

        # get index of docs with top k highest scores
        for s in doc_scores:  # doc_socres: num_queries * num_results
            top_k_idx = s.argsort()[::-1][:num_results]
            idxes.append(top_k_idx)

        # get results and its annotations
        query_scores = []
        for i in range(len(idxes)):
            scores = []
            for idx in idxes[i]:
                q = queries[i]
                doc_id = corpus_idf[corpus[idx]]
                scores.append(ann[q][doc_id])
            query_scores.append(np.array(scores))
        print(query_scores)

        # generate gt
        gt_scores = []
        for q in queries:
            t = ann[q]
            t = sorted(t.items(), key=lambda item: item[1], reverse=True)[:num_results]
            gt_scores.append(np.array([i[1] for i in t]))
        print(gt_scores)

        return query_scores, gt_scores

    def getDCG(self, scores):
        return np.sum(
            np.divide(scores, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
            dtype=np.float32)

    def getNDCG(self, rank_scores, gt_rank_scores):
        rs_len = rank_scores.shape[0]
        gt_rs_len = gt_rank_scores.shape[0]

        if rs_len <= gt_rs_len:
            gt_rank_scores = gt_rank_scores[:rs_len]
        else:
            t = np.zeros(rs_len, dtype=np.float32)
            t[:gt_rs_len] = gt_rank_scores
            gt_rank_scores = t

        idcg = self.getDCG(gt_rank_scores)

        dcg = self.getDCG(rank_scores)

        if dcg == 0.0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg
    def test_auc(self):
        db = self.db
        #print(db)
        id_list = list(db.keys())
        test_num = 200
        id_list_sample = random.sample(id_list,test_num)
        
        test_list = {}
        for i in range(test_num):
            test_list[id_list_sample[i]] = db[id_list_sample[i]][0]
        auc_num = 0
        for ids in test_list.keys():
            game_name = test_list[ids]
            results = self.search_game(game_name)
            for game in results:
                if game[0] == ids:
                    auc_num += 1
                    break
        print('test_num:',test_num,'accuracy',auc_num/test_num)
    def test_auc_miss(self):
        db = self.db
        #print(db)
        id_list = list(db.keys())
        test_num = 200
        id_list_sample = random.sample(id_list,test_num)
        alph = random.sample('qwertyuiopasdfghjklzxcvbnm',1)
        test_list = {}
        for i in range(test_num):
            test_list[id_list_sample[i]] = db[id_list_sample[i]][0]
        auc_num = 0
        for ids in test_list.keys():
            game_name = test_list[ids]

            num = random.randint(0,len(game_name)-1)
            change = game_name[num]
            a = game_name.replace(change,''.join(alph),1)
            results = self.search_game(a)
            for game in results:
                if game[0] == ids:
                    auc_num += 1
                    break
        print('test_num:',test_num,'accuracy',auc_num/test_num)

    def test_auc_zh(self):
        db = self.db
        #print(db)
        id_list = list(db.keys())
        test_num = 200
        id_list_sample = random.sample(id_list,test_num)
        
        test_list = {}
        for i in range(test_num):
            test_list[id_list_sample[i]] = db[id_list_sample[i]][-1]
        auc_num = 0
        for ids in test_list.keys():
            game_name = test_list[ids]
            results = self.cross_lang_search(game_name)
            for game in results:
                if game[0] == ids:
                    auc_num += 1
                    break
        print('test_num:',test_num,'accuracy',auc_num/test_num)

    def test_auc_zh_steam(self):
        db = self.db
        #print(db)
        id_list = list(db.keys())
        test_num = 200
        id_list_sample = random.sample(id_list,test_num)
        
        test_list = {}
        for i in range(test_num):
            test_list[id_list_sample[i]] = db[id_list_sample[i]][-1]
        return test_list
        # auc_num = 0
        # for ids in test_list.keys():
        #     game_name = test_list[ids]
        #     results = self.cross_lang_search(game_name)
        #     for game in results:
        #         if game[0] == ids:
        #             auc_num += 1
        #             break
        # print('test_num:',test_num,'accuracy',auc_num/test_num)


if __name__ == '__main__':
    PS = PorterStemmer()
    search_engine = SearchEngine(config, word_tokenize, PS, isStemming=False)
    search_engine.test_auc_miss()
    # search_engine.test_auc_zh()
# #search_engine.test_spell_suggestor()

# # search_engine.testBM25(mode=1)
# # search_engine.testBM25(mode=0)
# search_engine.search_game('Call of dust')
# print('---------')
# search_engine.search('explore the space')
# # search_engine.cross_lang_search('巫师3')