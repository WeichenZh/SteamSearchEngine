from easydict import EasyDict as edict

config = edict()

config.DATABASE_FILE = 'games3.sqlite'
config.ANN_FILE = 'ann_from_tags_new.json'
config.CN2ENG = 'cn2eng.json'
config.CORPUS_IDF = 'corpus_idf_full.json'
config.QUERIES = 'test_queries.txt'
config.MIS_QUERIES = 'mis_test_queries.txt'
config.STOP_WORD = 'lemur-stopwords.txt'
config.BASIC_TABLE = '''
  (SELECT Id, TITLE, DESCRIPTION, DEVELOPER, RELEASE_DATE, REVIEW, SCORE_RATE, ZH_NAME
  FROM
    (SELECT TITLE AS game_name, RELEASE_DATE, REVIEW, SCORE_RATE
    FROM GAMES) as t1
  JOIN Details
    ON Details.TITLE = t1.game_name)
    '''

config.RANKING = edict()
config.RANKING.NUM_RESULTS = 10

config.SPELLING_SUGGESTOR = edict()
config.SPELLING_SUGGESTOR.MAX_DIST = 2
config.SPELLING_SUGGESTOR.WORD_SRC = 'unix-words.txt'
config.SPELLING_SUGGESTOR.WORD_FREQUENCY = 'word_freq.txt'