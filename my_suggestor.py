import math
import re

class Trie(object):
    def __init__(self):
        self.characters = {}
        self.end_word = None

    def create_word_branch(self, word):
        cur_trie = self
        for c in word:
            if c not in cur_trie.characters:
                cur_trie.characters[c] = Trie()
            cur_trie = cur_trie.characters[c]       # dfs
        cur_trie.end_word = word


class WordSuggestor(object):
    def __init__(self, word_src):
        self.word_src = word_src
        self.trie = self.create_Trie()

    def create_Trie(self):
        t = Trie()
        with open(self.word_src, 'r', encoding='utf-8') as f:
            words = f.readlines()
            for word in words:
                word = word.strip().lower()
                t.create_word_branch(word)
        return t

    def find_suggested_words(self, trie, target_word, max_dist, suggestions, cur_row_num, last_row_dist):
        character_tree = trie.characters

        if min(last_row_dist) > max_dist:
            return

        for c, sub_trie in character_tree.items():
            if cur_row_num == 1:
                last_row_dist = [i for i in range(len(target_word)+1)]

            cur_row_dist = [cur_row_num]
            for i, t_c in enumerate(target_word, start=1):
                count = 0 if t_c == c else 1
                cur_row_dist.append(min(last_row_dist[i-1]+count, last_row_dist[i]+1, cur_row_dist[i-1]+1))

            edit_distance = cur_row_dist[-1]

            if sub_trie.end_word and edit_distance <= max_dist:
                suggestions.append(sub_trie.end_word)

            self.find_suggested_words(sub_trie, target_word, max_dist, suggestions, cur_row_num+1, cur_row_dist)

    def suggest_words(self, target_word, max_distance):
        suggestions = []
        self.find_suggested_words(self.trie, target_word, max_distance, suggestions, 1, [0])

        return suggestions


def cosine_sim(word1, word2):
    cop = re.compile("[^a-z^A-Z]")
    word1 = cop.sub('', word1)
    word2 = cop.sub('', word2)

    dics = [[0] * 26 for _ in range(2)]

    for i, w in enumerate([word1, word2]):
        for c in w:
            dics[i][ord(c)-ord('a')] += 1

    val1 = 0
    val2 = 0
    val3 = 0
    for i in range(26):
        val1 += dics[0][i] ** 2
        val2 += dics[1][i] ** 2
        val3 += dics[0][i] * dics[1][i]

    return val3 * 1.0 / math.sqrt(val1 * val2 + 1e-8)