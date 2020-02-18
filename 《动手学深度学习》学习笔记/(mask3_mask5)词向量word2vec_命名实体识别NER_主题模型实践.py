# -*- coding: utf-8 -*-
# -*- created_time: 2019/7/12 -*-

from collections import Counter
import ftfy
import re
from Chinese_sources_scraping import Sina_scraping,Xinhua_scraping

# Tokenization for Chinese
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
# import jieba
import thulac
import jiagu
from nltk.tag import StanfordNERTagger
from gensim.models import word2vec
from gensim.models import Word2Vec,TfidfModel,LdaMulticore,HdpModel
from gensim import corpora
from gensim.test.utils import datapath
from sklearn.decomposition import FastICA

import numpy as np
import pandas as pd
import math

import warnings
warnings.filterwarnings("ignore") # ignore the warning information

import os
from os import listdir
from os.path import isfile,join
# # 在环境变量中添加JAVAHOME变量,指向java.exe所在的目录， D:/Java/jdk1.8.0_211/binjava.exe 也可以
java_path = "C:/Program Files/Java/jdk1.8.0_131/bin/java.exe"
os.environ['JAVAHOME'] = java_path
os.environ["PYTHONHASHSEED"] = "0"

# Construct NER Tagger based on stanfordnlp class in NLTK package
nltk_ner = StanfordNERTagger(
        "D:\Python\\7.Python_NLP\stanford-ner-2018-10-16\classifiers\edu\stanford\\nlp\models\\ner\chinese.misc.distsim.crf.ser.gz",
        "D:\Python\\7.Python_NLP\stanford-ner-2018-10-16\stanford-ner.jar")


class Document():
    def __init__(self,meta_data = {},language = "zh",filter=False):
        self.meta_data = meta_data # the meta-data of the text (or news articles)
        self.language = language  # indicate the language of the text in corpus

        # Whether some stopwords should be filtered
        self.filter = filter

        # storing each token and corresponding pos-tag into lists
        self.tokens_list = []
        # contain tokens and corresponding pos-tags(also contain person and organization tags, made by thulac package)
        self.thu_term_tag_results = []
        # contain pos-tags made bt NLTK packages
        self.pos_tags_list = []

        # storing NER tuples(contain token and correspondig NER tag in each tuple) into list; also storing NER PERSON tokens
        # and NER ORGANIZATION tokens into separate lists(self.ner_person_list and self.ner_organization_list)
        self.ner_results_list = []
        self.ner_persons_list = []
        self.ner_organizations_list = []

        # count the number of tokens and assign the inputed text to the variable self.text
        self.num_tokens = 0
        self.text = ""

        # storing the frequencies of tokens into a statistical Counter class
        self.token_frequencies_dict = Counter()
        # storing the frequencies of pos-tags(made by NLTK) into a statistical Counter class
        self.pos_tag_frequencies_dict = Counter()

    def extract_features_from_text(self,text):
        preprocessed_text = self.preprocess(text)
        self.text = preprocessed_text

        # perform tokenization for the text (sentence)
        initial_tokens,thu_results = self.tokenizer(preprocessed_text)
        self.tokens_list.extend(initial_tokens)
        self.thu_term_tag_results.extend(thu_results)


        # Get tokens and corresponding NER tags. Get selected tokens whose corresponding NER tags are "PERSON" or "ORGANIZATION".
        ner_results, ner_persons, ner_organizations,renew_tokens = self.NER_tagger(initial_tokens = self.tokens_list)
        self.ner_results_list.extend(ner_results)
        self.ner_persons_list.extend(self._set(ner_persons)) # filter duplicated persons in list "ner_persons"
        self.ner_organizations_list.extend(self._set(ner_organizations)) # filter duplicated organizations in list "ner_organizations"
        # 在进行完NER之后,有些词被拆开的organization被重新合并,这时候在一个新的列表renew_tokens里存储其他的tokens以及被合并的organization的tokens,
        # 之后更新self.tokens_list
        self.tokens_list = renew_tokens

        self.num_tokens += len(self.tokens_list)
        self.token_frequencies_dict.update(self.tokens_list)


    def preprocess(self,text):
        # preprocess the multi-spaces, newlines and urls in strings

        multispace_re = re.compile(r"\s{2,}")
        newline_re = re.compile(r"\n+")
        url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")
        punctuation_re = re.compile(r"(\*|-|·)")
        # emoji_re = re.compile("([\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF])")

        preprocessed_text = multispace_re.sub(" ", text)
        preprocessed_text = newline_re.sub("", preprocessed_text)
        preprocessed_text = url_re.sub("url", preprocessed_text)
        preprocessed_text = punctuation_re.sub("",preprocessed_text)
        preprocessed_text = ftfy.fix_text(preprocessed_text)
        return preprocessed_text

    # Filter the repetitive elements in the list
    def _set(self, original_list):
        filtered_list = []

        for token in original_list:
            if token not in filtered_list:
                filtered_list.append(token)
            else:
                continue
        return filtered_list


    def tokenizer(self,text):
        # seg = StanfordSegmenter(path_to_jar="D:/Python/7.Python_NLP/stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar")
        # seg.default_config(lang=self.language)

        # tokenization for text (sentence)
        thu = thulac.thulac(seg_only=False,filt=self.filter)
        # return tokens and corresponding pos-tags(also contain person and organization tags, made by thulac package)
        cut_results = thu.cut(text, text=False)

        initial_tokens = [token_tag[0] for token_tag in cut_results]
        # thu_pos_tags = [token_tag[1] for token_tag in cut_results]
        return initial_tokens, cut_results

    def NER_tagger(self,initial_tokens):
        # return tokens and corresponding NER tags(contain person and organization tags, made by NLTK package)
        ner_results = nltk_ner.tag(initial_tokens)
        # In tuple list "ner_results", it should convert all tuple elements into list elements for the convenience of followed process
        ner_results = [list(result) for result in ner_results]

        # select tokens whose corresponding NER tags are "ORGANIZATION" and "PERSON"
        # If some tokens marked as ORGANIZATION, are successive,then thses "ORGANIZATION" tokens should be connected.
        # Since, these successive tokens may represent a same organization.
        # If some tokens marked as PERSON are successive,then thses "PERSON" tokens should be connected.
        # Since, these successive tokens may represent a same person.

        len_ner_results = len(ner_results) - 1
        iter = 0
        # 因为在循环中每次扫描ner_results列表中当前索引以及下一索引的词与词性标记,因此这循环条件这里len_ner_results还要减1以防止列表指针超出范围
        while iter <= len_ner_results - 1:
            if ner_results[iter][1] == "ORGANIZATION" and ner_results[iter + 1][1] == "ORGANIZATION":
                # Connect two succrssive words with smae NER tag of "ORGANIZATION"
                ner_results[iter][0] = ner_results[iter][0] + ner_results[iter + 1][0]
                del ner_results[iter + 1]
                len_ner_results = len(ner_results) - 1
                continue

            elif ner_results[iter][1] == "PERSON" and ner_results[iter + 1][1] == "PERSON":
                # Connect two succrssive words with smae NER tag of "PERSON"
                ner_results[iter][0] = ner_results[iter][0] + ner_results[iter + 1][0]
                del ner_results[iter + 1]
                len_ner_results = len(ner_results) - 1
                continue

            else:
                iter += 1

        # The list "renew_tokens" is used to renew tokens_list
        renew_tokens = [result[0] for result in ner_results]
        ner_organizations = [result[0] for result in ner_results if result[1] == "ORGANIZATION"]
        ner_persons = [result[0] for result in ner_results if result[1] == "PERSON"]

        return ner_results, ner_persons, ner_organizations, renew_tokens



class Proximity_models():
    def __init__(self,tokens_list, ner_persons_list, ner_organizations_list, ner_results_list, trained_models_path=None,continue_training=False):
        # 若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,则所有文本texts被分成的tokens的列表
        # 都应存入列表tokens_list中; 即tokens_list为一个列表嵌套列表的结构,其中每个列表代表一个文本text或者一篇news article的所有tokens
        self.tokens_list = tokens_list

        # 保存已训练好的模型文件(.model结尾的文件)的相对路径
        self.trained_models_path = trained_models_path
        # 是否继续训练word2vec模型,若是,则读取模型的同时对新文本(或新news article)进行增量训练(incremental training);若否,则只读取模型运用其训练的词向量
        # 当self.trained_models_path为None时,self.continue_training必为False;当self.trained_models_path不为None时,self.continue_training可为True或False
        self.continue_training = continue_training
        # Store all trained word2vec models
        self.w2c_models = []

        # Store all tokens and corresponding NER tags of all texts or news articles (contain person and organization tags)
        # 若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,则所有文本texts的NER的结果都应存入同一个列表ner_results_list中
        # 此列表结构为列表嵌套列表,如:[['Get', 'O'], ['a', 'O'], ['fresh', 'O'],  ["'s", 'O'], ['National Bureau of Statistics', 'ORGANIZATION']...]
        self.ner_results_list = ner_results_list
        # Store all Pesrons extrated from texts，若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,
        # 则所有文本中的Persons都应存在列表ner_persons_list中,并进行去重操作
        self.ner_persons_list = self._set(ner_persons_list)
        # Store all Organizations extrated from texts，若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,
        # 则所有文本中的Organizations都应存在列表ner_organizations_list中,并进行去重操作
        self.ner_organizations_list = self._set(ner_organizations_list)

        # Store all models' computed total distances from given Person to all Organizations, into the dictionary "all_persons_proximity_rank"
        # self.all_persons_proximity_rank字典的每个键keys为一个Person,对应的值为所有models计算的此Person与所有Organizations的Proximities,
        # 即值values为一个列表嵌套着列表再嵌套着字典的结构(每个字典只有一个Organization键,与一个proximity值),
        # 如{"Xi Jingping": [ [{'嘉吉公司': 0.13809946977723012}, {'美国农业部': 0.38254013327187913}, {'新华社': 0.47936039695089083}],
        # [{'嘉吉公司': 0.13809946977723012}, {'美国农业部': 0.38254013327187913}, {'新华社': 0.47936039695089083}], ......], "Liu He": [...] }一共包含9个models的结果
        self.all_persons_proximity_rank = {}  # 所有的Persons与Organizations的proximities均存在字典self.all_persons_proximity_rank中(最重要！！！)

        # 此处,针对所有models计算的proximities (某个Person与所有Organizations之间的proximities)计算平均值,
        # 即将所有models的proximities结果合为一个总的proximities结果，结果存在字典self.final_persons_organizations_proximity_results中,
        # 每个键为一个Person,值为此Person与所有Organizations的proximities总的结果(也为一个字典)(最重要！！！)
        self.final_persons_organizations_proximity_results = {}


        # Store all tokens' distance differences of Person tokens to Organization tokens
        # For instance, the distance differences are
        # self.all_persons_token_distances为一个字典嵌套字典的结构,其中所有的Persons键下均包含当前Person与所有Organizations在文本中的分词距离差的字典(token distances)
        self.all_persons_token_distances = {}
        # self.all_organizations_token_distances = {}
        self.all_persons_tokenDistance_calculation()

        # self.all_persons_locationVector_distances = {}
        # self.locationVector_distances_calculation()


        if self.trained_models_path is None:
            self._train_models()
        else:
            self._load_trained_models()


    def _train_models(self):
        # 应将一个完整text的所有tokens的列表再放入另一个列表中传给sentences参数,否则Word2Vec模型将会将单个token分成一个个字训练词向量
        word2vec_model_1 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=3, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_1.save("Chinese_articles_Skip-gram_models\word2vec_model_1.model")

        word2vec_model_2 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=5, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_2.save("Chinese_articles_Skip-gram_models\word2vec_model_2.model")

        word2vec_model_3 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=7, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_3.save("Chinese_articles_Skip-gram_models\word2vec_model_3.model")

        word2vec_model_4 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=9, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_4.save("Chinese_articles_Skip-gram_models\word2vec_model_4.model")

        word2vec_model_5 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=14, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_5.save("Chinese_articles_Skip-gram_models\word2vec_model_5.model")

        word2vec_model_6 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=19, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_6.save("Chinese_articles_Skip-gram_models\word2vec_model_6.model")

        word2vec_model_7 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=21, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_7.save("Chinese_articles_Skip-gram_models\word2vec_model_7.model")

        word2vec_model_8 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=23, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_8.save("Chinese_articles_Skip-gram_models\word2vec_model_8.model")

        word2vec_model_9 = word2vec.Word2Vec(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                             size=200, window=25, min_count=0, sg=1, hs=0, negative=5, workers=4, iter=20)
        word2vec_model_9.save("Chinese_articles_Skip-gram_models\word2vec_model_9.model")

        # 将所有训练的模型存入列表self.w2c_models中,以方便之后使用(Save all trained models into the list self.w2c_models for later use)
        self.w2c_models.extend([word2vec_model_1, word2vec_model_2, word2vec_model_3, word2vec_model_4, word2vec_model_5, word2vec_model_6,
             word2vec_model_7, word2vec_model_8, word2vec_model_9])

    # can do the incremental training
    def _load_trained_models(self):
        # 利用列表生成器,读取存储着已训练模型的路径中,所有已训练好的模型的路径名(为路径加模型名的连接)
        trained_models_name = [join(self.trained_models_path, model_name) for model_name in listdir(self.trained_models_path)
                               if isfile(join(self.trained_models_path, model_name)) and model_name.endswith(".model")]

        for trained_model_path_name in trained_models_name:
            word2vec_model = Word2Vec.load(trained_model_path_name)

            # 若此次没有传入新的文本语料库(或新的news articles),则只读取模型运用其训练的词向量 (即self.continue_training为False)
            # 若此次传入了新的文本语料库(或新的news articles),则要对读取的所有word2vec模型继续训练(即增量训练)
            if self.continue_training is True:
                # 新添加的语料库(news articles)需要更新到词表中(即update=True)
                word2vec_model.build_vocab(self.tokens_list,update=True)
                # 计算此时词表中,总共的文本语料库数量 (word2vec_model.corpus_count + len(self.tokens_list))
                total_examples = word2vec_model.corpus_count

                # 下方继续训练模型时需指定最新的文本语料库数量total_examples
                word2vec_model.train(sentences=[each_article_tokens for each_article_tokens in self.tokens_list ],
                                     total_examples=total_examples,epochs=5)

                # 将增量训练之后的word vector的Skip-gram重新存储起来
                word2vec_model.save(trained_model_path_name)

            # 将所有读取的模型存入列表self.w2c_models中,以方便之后使用
            self.w2c_models.append(word2vec_model)

    # Filter the repetitive elements in the list
    def _set(self, original_list):
        filtered_list = []

        for token in original_list:
            if token not in filtered_list:
                filtered_list.append(token)
            else:
                continue
        return filtered_list


    def all_persons_tokenDistance_calculation(self):
        # Store all Person tokens and corresponding indexes in the list "persons_indexes",
        # also store all Organization tokens and corresponding indexes in the list "organizations_indexes"
        # 下方两个列表的结构为列表嵌套列表,其中每个列表元素均存储着Person或者Organization与其分词后token的索引号,如:[['Mitchell', 7], ['National Bureau of Statistics', 17]...]
        persons_indexes = []
        organizations_indexes = []

        for index,token in enumerate(self.ner_results_list):
            if token[1] == "PERSON":
                persons_indexes.append([token[0], index])
            elif token[1] == "ORGANIZATION":
                organizations_indexes.append([token[0], index])
            else:
                continue

        # Calculate the token distances of each Person token to all Organization tokens
        # persons_indexes列表如：[["Mitchell", 7], ["Xi Jingping", 17]...]
        for person in persons_indexes:
            # 字典each_person_token_distances下包含当前循环的Person与所有Organizations在文本中的分词距离差(token distances),字典的键为Organizations
            # 字典each_person_token_distances如: {"G2O":2 ,"China's bank of people":14 ,...} (以单个Person的each_person_token_distances字典为例)
            each_person_token_distances = {}

            # organizations_indexes列表如：[["G2O", 9], ["China's bank of people", 21]...]
            for organization in organizations_indexes:
                # Token distance difference即Orgaization token的索引减去Person token索引的绝对值
                token_distance = abs(organization[1] - person[1])

                # 若当前each_person_token_distances字典中已存在一个相同的Organization的token distance difference,则判断当前Organization
                # 的token distance difference与已存在的Organization的token distance difference谁更小,选择更小的token distance difference保留
                if organization[0] in each_person_token_distances.keys() and token_distance < each_person_token_distances[organization[0]]:
                    # 字典each_person_token_distances下包含当前循环的Person与所有Organizations在文本中的分词距离差(token distances),字典的键为Organizations
                    each_person_token_distances[organization[0]] = token_distance
                elif organization[0] not in each_person_token_distances.keys():
                    each_person_token_distances[organization[0]] = token_distance
                else:
                    continue

            # 将字典each_person_token_distances下包含的所有Organizations的键的顺序排列为与self.ner_organizations_list列表中的Organizations的顺序一样,
            # 以方便之后将此字典的值转化为列表再转化为numpy数组后,其Organizations的顺序与每个w2c_model计算的proximities数组(numpy)的Organizations的顺序一样,
            # 以方便这两个计算得出的numpy数组的对应元素两两相加
            each_person_token_distances_copy = {}
            for organ in self.ner_organizations_list:
                each_person_token_distances_copy[organ] = each_person_token_distances[organ]

            # 字典each_person_token_distances的所有organizations的token distances经过重新排序后以及拷贝入字典each_person_token_distances_copy中,
            # 因此字典each_person_token_distances可以被删除释放内存
            del each_person_token_distances

            # 若当前Person的所有模型的proximities都已经计算完成,且此Person不在self.all_persons_token_distances字典中时,
            # 向此字典中加入此Person与所有Organizations的token distances
            if person[0] not in list(self.all_persons_token_distances.keys()):
                # self.all_persons_token_distances为一个字典嵌套字典的结构,其中所有的Persons键下均包含当前Person与所有Organizations
                # 在文本中的分词距离差的字典(token distances)
                self.all_persons_token_distances[person[0]] = each_person_token_distances_copy

            # 若当前Person的所有模型的proximities都已经计算完成,但此Person已存在self.all_persons_token_distances字典中时,
            # 新的此Person与所有Organizations的token distances应与上一次此Person计算的与所有的Organizations的token distances进行比较,
            # 若此次计算的某个organization的token distance比上一次计算的organization的token distance要小,则替换为更小的token distance值;
            # 若此次计算的某个organization的token distance比上一次计算的organization的token distance大于或等于,则不改变此token distance值.
            elif person[0] in list(self.all_persons_token_distances.keys()):
                for organ in list(self.all_persons_token_distances[person[0]].keys()):
                    # 若此次计算的某个organization的token distance比上一次计算的organization的token distance要小,则替换为更小的token distance值
                    if each_person_token_distances_copy[organ] < self.all_persons_token_distances[person[0]][organ]:
                        # self.all_persons_token_distances[person[0]]取到得为此Person与所有organizations的token distance的字典
                        self.all_persons_token_distances[person[0]][organ] = each_person_token_distances_copy[organ]

                    # 若此次计算的某个organization的token distance比上一次计算的organization的token distance大于或等于,则不改变此token distance值
                    elif each_person_token_distances_copy[organ] >= self.all_persons_token_distances[person[0]][organ]:
                        continue


        # for organization in organizations_indexes:
        #     # 字典each_organization_token_distances下包含当前循环的Organization与所有Persons在文本中的分词距离差(token distances),字典的键为Persons
        #     # 字典each_organization_token_distances如: {"Zhou":7 ,"Xi Jingping":3,...} (以单个Person的each_person_token_distances字典为例)
        #     each_organization_token_distances = {}
        #
        #     # persons_indexes列表如：[["Zhou", 9], ["Xi Jingping", 21]...]
        #     for person in persons_indexes:
        #         # Token distance difference即Orgaization token的索引减去Person token索引的绝对值
        #         token_distance = abs(person[1] - organization[1])
        #
        #         # 若当前each_person_token_distances字典中已存在一个相同的Organization的token distance difference,则判断当前Organization
        #         # 的token distance difference与已存在的Organization的token distance difference谁更小,选择更小的token distance difference保留
        #         if person[0] in each_organization_token_distances.keys() and token_distance < each_organization_token_distances[person[0]]:
        #             # 字典each_person_token_distances下包含当前循环的Person与所有Organizations在文本中的分词距离差(token distances),字典的键为Organizations
        #             each_organization_token_distances[person[0]] = token_distance
        #         elif person[0] not in each_organization_token_distances.keys():
        #             each_organization_token_distances[person[0]] = token_distance
        #         else:
        #             continue
        #
        #     # 将字典each_person_token_distances下包含的所有Organizations的键的顺序排列为与self.ner_organizations_list列表中的Organizations的顺序一样,
        #     # 以方便之后将此字典的值转化为列表再转化为numpy数组后,其Organizations的顺序与每个w2c_model计算的proximities数组(numpy)的Organizations的顺序一样,
        #     # 以方便这两个计算得出的numpy数组的对应元素两两相加
        #     each_organization_token_distances_copy = {}
        #     for person in self.ner_persons_list:
        #         each_organization_token_distances_copy[person] = each_organization_token_distances[person]
        #
        #     # 字典each_person_token_distances的所有organizations的token distances经过重新排序后以及拷贝入字典each_person_token_distances_copy中,
        #     # 因此字典each_person_token_distances可以被删除释放内存
        #     del each_organization_token_distances
        #
        #     # 若当前Person的所有模型的proximities都已经计算完成,且此Person不在self.all_persons_token_distances字典中时,
        #     # 向此字典中加入此Person与所有Organizations的token distances
        #     if organization[0] not in list(self.all_organizations_token_distances.keys()):
        #         # self.all_persons_token_distances为一个字典嵌套字典的结构,其中所有的Persons键下均包含当前Person与所有Organizations
        #         # 在文本中的分词距离差的字典(token distances)
        #         self.all_organizations_token_distances[organization[0]] = each_organization_token_distances_copy
        #
        #     # 若当前Person的所有模型的proximities都已经计算完成,但此Person已存在self.all_persons_token_distances字典中时,
        #     # 新的此Person与所有Organizations的token distances应与上一次此Person计算的与所有的Organizations的token distances进行比较,
        #     # 若此次计算的某个organization的token distance比上一次计算的organization的token distance要小,则替换为更小的token distance值;
        #     # 若此次计算的某个organization的token distance比上一次计算的organization的token distance大于或等于,则不改变此token distance值.
        #     elif organization[0] in list(self.all_organizations_token_distances.keys()):
        #         for person in list(self.all_organizations_token_distances[organization[0]].keys()):
        #             # 若此次计算的某个organization的token distance比上一次计算的organization的token distance要小,则替换为更小的token distance值
        #             if each_organization_token_distances_copy[person] < self.all_organizations_token_distances[organization[0]][person]:
        #                 # self.all_persons_token_distances[person[0]]取到得为此Person与所有organizations的token distance的字典
        #                 self.all_organizations_token_distances[organization[0]][person] = each_organization_token_distances_copy[person]
        #
        #             # 若此次计算的某个organization的token distance比上一次计算的organization的token distance大于或等于,则不改变此token distance值
        #             elif each_organization_token_distances_copy[person] >= self.all_organizations_token_distances[organization[0]][person]:
        #                 continue


    # def locationVector_distances_calculation(self):
    #     # 为保证persons与organizations的顺序一致,此处列表生成式中需用self.ner_persons_list与
    #     persons_location_vectors = np.vstack(( np.array(list(self.all_persons_token_distances[person].values())) for person
    #                                            in self.ner_persons_list ))
    #     organs_location_vectors = np.vstack(( np.array(list(self.all_organizations_token_distances[organ].values())) for organ
    #                                           in self.ner_organizations_list ))
    #
    #     n_component = math.floor(persons_location_vectors.shape[1] + organs_location_vectors.shape[1])
    #     ica = FastICA(n_components=n_component,random_state=3)
    #
    #     persons_location_vectors = ica.fit_transform(persons_location_vectors)
    #     organs_location_vectors = ica.fit_transform(organs_location_vectors)
    #
    #     # self.all_persons_locationVector_distances
    #     for person_index,person in enumerate(self.ner_persons_list):
    #         each_person_locationVector_distance = np.hstack(( np.sqrt(np.sum(np.square(persons_location_vectors[person_index,:] - organs_location_vectors[i,:])))
    #                                                           for i in range(organs_location_vectors.shape[0]) ))
    #         self.all_persons_locationVector_distances[person] = each_person_locationVector_distance


    def all_persons_proximity_calculation(self):
        for person in self.ner_persons_list:
            # Given a specific Person name, storing each model's computed cosine distances from given Person to all Organizations into the list "all_models_proximity_rank"
            all_models_proximity_rank = []

            for w2c_model in self.w2c_models:
                # Compute cosine distances from given Person to all Organizations in "doc.ner_organizations_list" list.
                w2c_Distances = w2c_model.wv.distances(person, self.ner_organizations_list)  # return a numpy array
                w2c_Distances = np.abs(w2c_Distances)
                w2c_Distances = w2c_Distances / np.sum(w2c_Distances)  # 进行归一化(normalization)

                # # self.all_persons_token_distances为一个字典嵌套字典的结构,其中所有的Persons键下均包含当前Person与
                # 所有Organizations在文本中的分词距离差的字典(token distances)
                # 字典self.all_persons_token_distances[person]下包含的所有Organizations的键的顺序与self.ner_organizations_list列表中的Organizations的顺序一样,
                # 方便了现在将此字典的值转化为列表再转化为numpy数组后,其Organizations的顺序与每个w2c_model计算的w2c_Distances数组(numpy)的Organizations的顺序一样,
                # 以方便这两个计算得出的numpy数组的对应元素两两相加或相乘
                token_Distances = np.array(list(self.all_persons_token_distances[person].values()))
                # token_Distances = self.all_persons_locationVector_distances[person]
                token_Distances = token_Distances / np.sum(token_Distances)  # 进行归一化(normalization)

                # total_Distance的计算公式为 (3/5)*token_Distances + (2/5)*w2c_Distances, 即token_Distances占比更大
                total_Distances = (2 / 3) * token_Distances + (1 / 3) * w2c_Distances
                total_Distances = total_Distances / np.sum(total_Distances)  # 进行归一化(normalization)

                organizations_and_distances = [{organization: organization_dis} for organization, organization_dis in
                                               zip(self.ner_organizations_list, total_Distances)]

                # 当organizations_and_distances为列表嵌套字典的结构(每个字典只有一个Organization键,与一个proximity值),对此字典列表进行排序的代码
                organizations_and_distances = sorted(organizations_and_distances, key=lambda x: list(x.values())[0] )
                # 当organizations_and_distances为列表嵌套列表的结构,对此列表进行排序的代码
                # organizations_and_distances = sorted(organizations_and_distances, key=lambda x: x[1])
                # 当organizations_and_distances为字典,对此字典进行排序的代码
                # organizations_and_distances = sorted(organizations_and_distances.items(), key=lambda x: x[1])

                # To find the potential relations between Person entity and Organization entites except the most related organization with Person entity
                the_most_related_organization = list(organizations_and_distances[0].keys())[0]
                the_most_related_proximity = list(organizations_and_distances[0].values())[0]
                # Except the most related organization with Person entity
                other_ner_organizations_copy = self.ner_organizations_list[:]
                other_ner_organizations_copy.remove(the_most_related_organization)
                potential_relations_proximities = w2c_model.wv.distances(the_most_related_organization, other_ner_organizations_copy) # return 1D numpy array
                potential_relations_proximities = potential_relations_proximities + the_most_related_proximity

                # 将最相关的Organization的名称与proximity分别加入other_ner_organizations_copy与potential_relations_proximities中,以方便之后排序与归一化
                # Add the most relevant Organization name and proximity to other_ner_organizations_copy and potential_relations_proximities to facilitate subsequent sorting and normalization.
                other_ner_organizations_copy.insert(0,the_most_related_organization)
                potential_relations_proximities = np.hstack( (np.array([the_most_related_proximity]), potential_relations_proximities) )
                # potential_relations_proximities = np.exp(potential_relations_proximities) / np.sum( np.exp(potential_relations_proximities) )
                potential_relations_proximities = potential_relations_proximities / np.sum(potential_relations_proximities)

                organizations_and_distances = [{organization: poten_proximity} for organization, poten_proximity in
                                        zip(other_ner_organizations_copy,potential_relations_proximities)]
                # rank organization-proximity pairs in ascending order
                organizations_and_distances = sorted(organizations_and_distances, key=lambda x: list(x.values())[0])


                # Given a specific Person name, storing each model's computed cosine distances from given Person to all Organizations into the list
                all_models_proximity_rank.append(organizations_and_distances)

            # Store all models' computed cosine distances from given Person to all Organizations, into the dictionary "all_persons_proximity_rank"
            self.all_persons_proximity_rank[person] = all_models_proximity_rank

        for person,proximities_results in self.all_persons_proximity_rank.items():
            print("\nPerson: ",person)
            for proximities_result in proximities_results:
                print(proximities_result)

    def final_persons_organizations_proximities(self):
        # self.all_persons_proximity_rank字典的每个键为一个Person,对应的值为所有models计算的此Person与所有Organizations的Proximities,
        # 在此处,针对所有models计算的proximities (某个Person与所有Organizations之间的proximities)计算平均值,即将所有models的proximities结果合为一个总的proximities结果
        # 结果存在字典self.final_persons_organizations_proximity_results中,每个键为一个Person,值为此Person与所有Organizations的proximities总的结果(也为一个字典)
        for person in self.all_persons_proximity_rank.keys():

            # 先将所有models计算出的Person与每个Organization的proximity的结果都一起放入一个列表中(即将之前列表嵌套列表再嵌套Person
            # 与Organization的proximity的字典的结构,变为一个总的列表嵌套所有模型计算的Person与Organization的proximity的字典的结构)
            # 如:[{'嘉吉公司': 0.13809946977723012},{'美国农业部': 0.38254013327187913},{'新华社': 0.47936039695089083},{'嘉吉公司': 0.13809946977723012},......]
            all_models_proximity_results = []
            for each_model_proximity_results in self.all_persons_proximity_rank[person]:
                all_models_proximity_results.extend(each_model_proximity_results)

            # 字典each_person_proximity_results为此Person与所有Organizations的proximities总的结果,即字典self.final_persons_organizations_proximity_results中的值
            each_person_proximity_results = {}

            # 循环列表self.ner_organizations_list中的每个Organization
            # 针对所有models计算的同一个Organization的proximities(某个Person与此Organization的proximities)计算平均值,
            # 即将所有models的同一个Organization的proximities结果合为一个总的平均proximity结果
            for organization in self.ner_organizations_list:
                each_person_proximity_results[organization] = 0

                for organ_proximity in all_models_proximity_results:
                    # 若Organization与其proximity的字典结构organ_proximity的键的Organization与当前循环的Organization一样,
                    # 则将当前organ_proximity字典中Organization键的proximity值与字典each_person_proximity_results中已有Organization键的proximity值进行累加
                    if list(organ_proximity.keys())[0] == organization:
                        each_person_proximity_results[organization] += organ_proximity[organization]

                # 字典each_person_proximity_results中同一个Organization键的proximities累加值除以模型列表self.w2c_models中的模型总数,
                # 即求出了所有models对同一个Organization的所有proximities结果的一个总的平均proximity结果
                each_person_proximity_results[organization] = each_person_proximity_results[organization] / len(self.w2c_models)

            # 对字典each_person_proximity_results中所有Organizations与其总的平均proximities结果进行归一化(Normalization)
            sum_each_person_proximity_results = sum(list(each_person_proximity_results.values()))
            for key in each_person_proximity_results.keys():
                each_person_proximity_results[key] = each_person_proximity_results[key] / sum_each_person_proximity_results
            del sum_each_person_proximity_results

            # 最后将字典each_person_proximity_results中所有Organizations与其总的平均proximities结果按照proximities的大小,进行从小到大的排列
            # 排序后的each_person_proximity_results字典变为了一个列表嵌套元组的结构
            # 如:[('嘉吉公司', 0.11765984560526263), ('美国农业部', 0.3905237147102674), ('新华社', 0.49181643968447)]
            each_person_proximity_results = sorted(each_person_proximity_results.items(), key=lambda x: x[1])

            # 最后将当前循环中此Person与所有Organizations的总的平均proximities结果存入字典self.final_persons_organizations_proximity_results中
            self.final_persons_organizations_proximity_results[person] = each_person_proximity_results

        print("\nFinal_persons_organizations_proximity_results: ")
        for person,proximities_results in self.final_persons_organizations_proximity_results.items():
            print("\nPerson: ", person)
            for proximities_result in proximities_results:
                print(proximities_result)


# Utilize Latent Dirichlet Allocation model or Hierarchical Dirichlet Process to build the topic model
class Topic_model():
    def __init__(self,tokens_list,ner_persons_list,ner_organizations_list, w2c_models = None,
                 trained_topic_model_path = "Topic_model\Chinese_topic_model",continue_training = False):

        # 若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,则所有文本texts被分成的tokens的列表
        # 都应存入列表tokens_list中; 即tokens_list为一个列表嵌套列表的结构,其中每个列表代表一个文本text或者一篇news article的所有tokens
        self.tokens_list = tokens_list
        # remove all Chinese stopwords from self.tokens_list
        self._remove_stopwords()

        # Store all Pesrons extrated from texts，若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,
        # 则所有文本中的Persons都应存在列表ner_persons_list中,并进行去重操作
        self.ner_persons_list = self._set(ner_persons_list)
        # Store all Organizations extrated from texts，若一次输入多个news articles或多个文本texts进行Persons与Organizations的proximities的计算,
        # 则所有文本中的Organizations都应存在列表ner_organizations_list中,并进行去重操作
        self.ner_organizations_list = self._set(ner_organizations_list)

        # Get all trained word2vec models
        self.w2c_models = w2c_models

        # The path that contains the trained topic model (Latent Dirichlet Allocation model), needs to be loaded
        self.trained_topic_model_path = trained_topic_model_path
        # Whether to perform the incremental training
        self.continue_training = continue_training

        # 生成词袋字典,所谓词袋字典就是一个字典(dictionary),里面存放了所有文本texts(或news articles)的tokens和他们的索引(key)。
        # gensim的dictionary还有个最大的特色是它不仅存放了语料库中的所有文档中的所有tokens,还存放了这些单词出现的次数等信息。
        self.dictionary = None
        # 之前已经生成了词袋字典dictionary,接下来要使用词袋字典dictionary的方法doc2bow来统计每篇文档中的所有tokens的索引号
        # 以及所有tokens在每篇文档中出现的频数,将结果存储在词袋语料库bow_corpus中
        self.bow_corpus = None

        # 训练的文档主题分布模型:分层狄利克雷过程HDP(gensim.models.HdpModel方法训练HDP模型)
        # 或者隐狄利克雷分配LDA模型(使用gensim.models.LdaMulticore方法训练LDA模型),在分层狄利克雷过程中,无需特别指定需要提取的topic的数量,模型会自动计算
        self.topic_model = None

        self.model_saved_name = "Chinese_hdp_model"
        if self.trained_topic_model_path is None:
            self._initial_topic_model()
        else:
            self._continue_training_topic_model()


    # Filter the repetitive elements in the list
    def _set(self, original_list):
        filtered_list = []

        for token in original_list:
            if token not in filtered_list:
                filtered_list.append(token)
            else:
                continue
        return filtered_list

    def _remove_stopwords(self):
        # load all Chinese stopwords in the list "stopwords"
        stopwords = [line.strip("\n").strip("\ufeff") for line in
                     open("stopwords\Chinese_stopwords\Chinese_stopwords.txt", encoding="utf-8")]

        # remove all Chinese stopwords from self.tokens_list
        new_tokens_list = []
        for each_tokens_list in self.tokens_list:
            each_new_tokens_list = [token for token in each_tokens_list if token not in stopwords]
            new_tokens_list.append(each_new_tokens_list)

        # Copy the new_tokens_list(remove all Chinese stopwords) to the self.tokens_list, then delete new_tokens_list
        # and stopwords to save memory
        self.tokens_list = new_tokens_list[:]
        del stopwords,new_tokens_list


    # First time to train topic model based on the documents
    def _initial_topic_model(self):
        # 生成词袋字典,所谓词袋字典就是一个字典(dictionary),里面存放了所有文本texts(或news articles)的tokens和他们的索引(key)。
        # gensim的dictionary还有个最大的特色是它不仅存放了语料库中的所有文档中的所有tokens,还存放了这些单词出现的次数等信息。
        self.dictionary = corpora.Dictionary(self.tokens_list)

        # 之前已经生成了词袋字典dictionary,接下来要使用词袋字典dictionary的方法doc2bow来统计每篇文档中的所有tokens的索引号
        # 以及所有tokens在每篇文档中出现的频数,将结果存储在词袋语料库bow_corpus中
        self.bow_corpus = [self.dictionary.doc2bow(each_tokens_list) for each_tokens_list in self.tokens_list]

        # 接着使用gensim的models.TfidfModel方法根据bow_corpus创建tf-idf的corpus对象
        # fit TF-IDF model
        tfidf_model = TfidfModel(self.bow_corpus)
        # apply model to all tokens of documents in the bow_corpus
        tfidf_corpus = tfidf_model[self.bow_corpus]

        # 在TF-IDF语料库tfidf_corpus上训练隐狄利克雷分配LDA模型,使用gensim.models.LdaMulticore方法训练LDA模型,模型中需要设置主题参数num_topics
        # 在词袋语料库bow_corpus上训练分层狄利克雷过程HDP(gensim.models.HdpModel方法训练HDP模型),在分层狄利克雷过程中,无需特别指定需要提取的topic的数量,模型会自动计算
        self.topic_model = HdpModel(corpus=self.bow_corpus,id2word=self.dictionary)

        # # svae the trained HDP model to the target path
        # fname_path = self.trained_topic_model_path + "\\" + self.model_saved_name
        # self.topic_model.save(fname_or_handle=fname_path)
        #
        # print("The trained topic model is saved")


    # load the trained topic model (if needed, the topic model will continue to be trained)
    def _continue_training_topic_model(self):
        # 生成词袋字典,所谓词袋字典就是一个字典(dictionary),里面存放了所有文本texts(或news articles)的tokens和他们的索引(key)。
        # gensim的dictionary还有个最大的特色是它不仅存放了语料库中的所有文档中的所有tokens,还存放了这些单词出现的次数等信息。
        self.dictionary = corpora.Dictionary(self.tokens_list)

        # 之前已经生成了词袋字典dictionary,接下来要使用词袋字典dictionary的方法doc2bow来统计每篇文档中的所有tokens的索引号
        # 以及所有tokens在每篇文档中出现的频数,将结果存储在词袋语料库bow_corpus中
        self.bow_corpus = [self.dictionary.doc2bow(each_tokens_list) for each_tokens_list in self.tokens_list]

        # 找到已训练的topic model的路径path,并载入topic model
        fname_path = self.trained_topic_model_path + "\\" + self.model_saved_name
        fname = datapath(fname_path)
        # load the trained topic model (注意,即使未继续训练更新的topic model,也可在新文档上计算并推断其相关的主题topics)
        self.topic_model = HdpModel.load(fname=fname)

        # If needed, the topic model will continue to be trained.
        # 此时的词袋字典self.dictionary与词袋语料库self.bow_corpus中存储的为新的未被topic model训练过的文档,
        # 可以利用这些新文档继续训练topic model。但注意,即使未继续训练更新的topic model,也可在新文档上计算并推断其相关的主题topics
        if self.continue_training is True:
            self.topic_model.update(self.bow_corpus)
        print("The topic model is continued to train")

    def show_all_topics(self,num_topics = None,num_words = 15):
        # If topic_num is None, then all topics and corresponding tokens will be outputted
        # If num_words is setted, then the number of tokens to be included per topics (ordered by significance) would be selected and outputted
        if num_topics is None:
            for topic_idx,topic in self.topic_model.print_topics(num_topics=-1,num_words=num_words):
                print("Topic: {} \n Tokens: {}".format(topic_idx,topic))

        # If topic_num is not None, then the top num_topics(ordered by significance) will be outputted.
        elif num_topics is not None:
            for topic_idx,topic in self.topic_model.print_topics(num_topics=num_topics,num_words=num_words):
                print("Topic: {} \n Tokens: {}".format(topic_idx,topic))


    # Based on one news article or one text, show it's most related topics and topic tokens
    def show_significant_topics(self,doc_index=0):
        print("Orignal article: \n",self.tokens_list[doc_index],"\n")

        for index, score in sorted(self.topic_model[self.bow_corpus[doc_index]],key = lambda tup: -1 * tup[1]):
            print("\nScore: {}\t \nTopic{}: {}".format(score, index, self.topic_model.print_topic(index, 25)))

            # 将每一个抽取的topic中前topn个最重要的tokens(tokens按对于此topic的重要性程度排列)存入列表topic_tokens中
            topic_tokens = [token_weight[0] for token_weight in self.topic_model.show_topic(topic_id=index,topn=20,formatted=False) if len(token_weight[0]) > 1]
            print("topic_{}_tokens:\n".format(index),topic_tokens)
            print("topic_{}_tokens pos tags:\n".format(index),jiagu.pos(topic_tokens))

    # Based on a person in a specific news article or text, show his/her most related topics and topic tokens
    def person_associated_topics(self,topic_tokens_num = 25):

        for person in self.ner_persons_list:
            for article_index, each_article_tokens in enumerate(self.tokens_list):
                # Check whether the present person is in the present article's tokens list.
                # If not, then turn to the next article's tokens list.
                # If the present person is in the present article's tokens list, then use topic model to show the topics
                # and topic tokens that most related to this article.
                if person in each_article_tokens:
                    print("\nPerson: ", person,";  Article index: ",article_index)

                    # score 为此topic与此篇article的相关程度概率评分
                    for topic_index,topic_score in sorted(self.topic_model[self.bow_corpus[article_index]],key = lambda tup: -1 * tup[1]):
                        # 将每一个抽取的topic中前topn个最重要的tokens的scores(tokens按对于此topic的重要性程度排列)存入列表topic_tokens_scores中
                        # formatted参数为False代表返回一个list of (token, weight) pairs
                        topic_tokens_scores = [token_weight[1] for token_weight in self.topic_model.show_topic(topic_id=topic_index, topn=topic_tokens_num, formatted=False)
                                               if len(token_weight[0]) > 1 and token_weight[0]!=person]
                        topic_tokens_scores = np.array(topic_tokens_scores) # convert list into numpy array
                        topic_tokens_scores = topic_tokens_scores / np.sum(topic_tokens_scores) # Normalization

                        # 将每一个抽取的topic中前topn个最重要的tokens(tokens按对于此topic的重要性程度排列)存入列表topic_tokens中
                        # formatted参数为False代表返回一个list of (token, weight) pairs
                        topic_tokens = [token_weight[0] for token_weight in self.topic_model.show_topic(topic_id=topic_index, topn=topic_tokens_num, formatted=False)
                                        if len(token_weight[0]) > 1 and token_weight[0]!=person]

                        # Compute cosine distances from a given Person to all topic tokens of a specific topic (in topic_tokens list)
                        person_topic_tokens_distances = np.vstack((w2c_model.wv.distances(person,topic_tokens)
                                                                   for w2c_model in self.w2c_models)) # return a numpy array
                        # Each w2c model will compute a result of cosine distances from a given Person to all topic tokens of a specific topic,
                        # then all w2c models' results should be used to compute the mean result.
                        person_topic_tokens_distances = np.mean(person_topic_tokens_distances,axis=0)
                        # Normalization
                        person_topic_tokens_distances = person_topic_tokens_distances / np.sum(person_topic_tokens_distances)


                        # person_topic_tokens_distances = person_topic_tokens_distances * topic_tokens_scores
                        # person_topic_tokens_distances = person_topic_tokens_distances / np.sum(person_topic_tokens_distances)

                        person_topic_tokens_results = [[token,score] for token,score in zip(topic_tokens,person_topic_tokens_distances)]
                        # Rank topic tokens based on the scores of topic tokens
                        person_topic_tokens_results = sorted(person_topic_tokens_results, key=lambda x: x[1])


                        print("Topic index: ",topic_index,"/ ","Topic score: ",topic_score)
                        print("Topic tokens and proximities: ",person_topic_tokens_results)



# -*- Creating cospus -*-

def corpus_creation(data_df):
    # for srti in range(data_df.shape[0]):
    for article in data_df:

        # input each news article into the class Document, and iteratly create the Document class
        doc = Document()
        doc.extract_features_from_text(article)

        yield doc


if __name__ == "__main__":
    xinhua = pd.read_csv("Chinese_news_articles_corpus.csv")
    # print(xinhua.iloc[1,:]["article content"])
    # print(xinhua.shape)

    # xinhua = Xinhua_scraping(term=["中美", "贸易战"], pages=7)
    # xinhua.MainPage_scraping()
    # # xinhua.save_as_csv()
    # contents = xinhua.contents
    # for content in contents:
    #     print("\n------------------------")
    #     print(content)


    sentence1 = "原标题：美国农产品巨头：贸易流向或正被永久改变……参考消息网7月15日报道 台媒称，美国农业产品巨头嘉吉公司7月11日发布财报显示，该公司出现4年来最大的67%单季利润下滑，公司认为贸易战是其中很重要的原因。据台湾钜亨网7月12日报道，嘉吉公司首席财务官戴维丹斯警告，贸易战持续的时间愈长，农产品贸易流动的转变就会变得更持久，或将难以恢复。报道称，由于中美贸易摩擦，嘉吉公司设法将黄豆等农产品转销至欧洲、中东和非洲买家。丹斯说，在贸易摩擦下，公司并未改变任何长期的投资计划。他们的目标是希望能快速解决当前状况。嘉吉公司指出，该公司第四财季不仅营收下跌，利润也下滑67%。报道介绍，美国农业部数据显示，多年来，中国一直是美国农产品的重要买家，但由于贸易战，2018年出口到中国的农产品大幅下降。资料图片。新华社责任编辑：张宁"
    # sentence2 = "台湾“中时电子报”15日发表评论说，台湾大选前，美国务院同意卖108辆M1A2T战车，而蔡当局仍“凯子”般接受这桩生意，4项共27亿美金，合计超过800亿元台币。我们不得不问为什么还要买？更可怕的是随之而来66架F16V战机，也将售台，这款三代半战机，当台湾面对大陆歼20及五代战机时，无论是火力、侦搜力都差一大截，是毫无招架之力废铁，为什么要买？这桩极不对称交易，特朗普与蔡英文，其实各有各的小算盘。特朗普、蔡英文两人都同一年面对领导人大选，而且都是拼连任，因此，什么事都得干。对特朗普而言，不外是赚钱，任何生意没有比军火生意好赚，尤其针对一个不能杀价的台湾，特朗普可以任意宰割，而蔡英文则是任人割宰的一方。蔡英文笨吗？特朗普聪明吗？都不是，最聪明的该是菲律宾总统杜特尔特。在中国大陆与美菲之间角力，本月8日他向美国强烈发出一个经典讯号，直言不会做美国在南海与大陆之间的马前卒，要打，美国先上，不要把“重责大任”放在区域伙伴身上。他并明确指出：“我们绝对赢不了和中国的战争，但美国始终在催促我们、煽动我们，把我当诱饵，你们把菲律宾当什么？”直白一点说，特朗普的生意经，是赤裸裸地想要不花钱还附带赚钱下，让台湾做中国大陆对抗的马前卒，在中美贸易战作为美对付大陆谈判的筹码。而蔡英文则为了胜选，用这批军火交易，拉近与美国的距离，换得访问加勒比海4个“邦交小国”，“过境”美国4天。后续的66架F16V战机更是钱坑，名为买军火防卫，实为自己连任，用“外交”政策为自己连任埋单，让人民情何以堪？这场军火交易，就是一场特朗普和蔡英文之间搞“水帮鱼、鱼帮水”的世纪大骗局。香港“大公网”15日也发表文章说，明眼人都看得出来，美国之所以愿意给蔡英文甜头尝，一来为了把台湾当提款机，不断收取巨额“保护费”；二来为了让台湾甘做中美博弈的“马前卒”，使唤起来更加顺手。美国所谓“挺台”从来不是要支持台湾“独立”，所谓“友台”也从未曾真正视台湾为盟友，而是把台湾当作筹码，“以台制华”才是根本目的。蔡英文当局一厢情愿地卖力迎合美方，拿台湾民生福祉当自己“拼选举”的赌注，用台湾百姓的血汗钱给自己交“保护费”。如何看待民进党的谋算和操弄，如何选择台湾的前途和命运，相信台湾人民自有定论。特别声明：以上文章内容仅代表作者本人观点，不代表新浪网观点或立场。如有关于作品内容、版权或其它问题请于作品发表后的30日内与新浪网联系。新浪简介|广告服务|About Sina联系我们|招聘信息|通行证注册产品答疑|网站律师|SINA EnglishCopyright © 1996-2019 SINA CorporationAll Rights Reserved 新浪公司 版权所有".split("特别声明：")[0]
    sentence3 = xinhua.iloc[2,:]["article content"]

    # sentence = "周柱君爱商汤科技有限责任公司,同时他的女朋友陈航在南开大学上研究生,但他的女儿周淼邈却在华为公司上班"
    #sentence = "周柱君 在 兰卡斯特大学 学习".split()
    print( "\n",sentence3,"\n\n",sentence1)

    articles = [sentence1]
    Documents = []
    Documents.extend(corpus_creation(articles))

    tokens_list = [doc.tokens_list for doc in  Documents]

    ner_persons_list = []
    for doc in Documents:
        ner_persons_list.extend(doc.ner_persons_list)
    print("\nTotal persons", ner_persons_list)

    ner_organizations_list = []
    for doc in Documents:
        ner_organizations_list.extend(doc.ner_organizations_list)
    print("Total organizations",ner_organizations_list)

    ner_results_list = []
    for doc in Documents:
        ner_results_list.extend(doc.ner_results_list)


    Proximity_model = Proximity_models(tokens_list=tokens_list, ner_persons_list=ner_persons_list,
                                       ner_organizations_list=ner_organizations_list,ner_results_list=ner_results_list,
                                       trained_models_path="Chinese_articles_Skip-gram_models",continue_training=False)
    Proximity_model.all_persons_proximity_calculation()
    Proximity_model.final_persons_organizations_proximities()
    print("\nall_persons_token_distances\n", Proximity_model.all_persons_token_distances)
    print("\nner_results_list\n", Proximity_model.ner_results_list)

    # topic_model = Topic_model(tokens_list=tokens_list,ner_persons_list=ner_persons_list,ner_organizations_list=ner_organizations_list,
    #                           w2c_models=Proximity_model.w2c_models,trained_topic_model_path=None)
    # topic_model.person_associated_topics()




    # a = {"G20":0.003}
    # a = list(a.values())[0]
    # print(a)

    # ner_results = nltk_ner.tag(sentence.split())
    # print(ner_results)


    # for term,tag in nltk_ner_results:
    #     print(term,tag)
