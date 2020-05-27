import pprint
import pandas as pd
import numpy as np
import string
import random
import collections
import heapq
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline



class TextUtils(object):
    # Text pre-processing

    @staticmethod
    def select_by_tag(x, y):
        return bool(set(x) & set(y))

    @staticmethod
    def remove_punctuation(text):    # remove punctuation
        soup = BeautifulSoup(text)
        htmlremoved = soup.get_text()
        return htmlremoved.encode('utf-8').translate(None, string.punctuation.translate(None, ''))

    @staticmethod
    def lower_case(text):            # lowercase all text in title
        return text.lower()

    @staticmethod
    def split_tag(tags):
        return list(tags.split('|'))

    @staticmethod
    def combine_title_body(title, body):
        return " ".join((title, body))



class Posts(object):
    # TODO: Prune documents that are too short.
    def __init__(self, db_cursor, top_tags, sampling_rate):
        column_names = ['id','title','body','score', 'tags', 'creation_date']
        self.rows = pd.DataFrame(db_cursor, columns=column_names)

        self.rows['tags_list_all'] = self.rows['tags'].map(TextUtils.split_tag)
        self.rows['mask'] = self.rows['tags_list_all'].map(lambda tags_lst: TextUtils.select_by_tag(tags_lst, top_tags))
        self.rows = self.rows[self.rows['mask']]

        self.rows['tags_list'] = self.rows['tags_list_all'].map(
            lambda tag_list: [tag for tag in tag_list if tag in top_tags])

        self.rows = self.rows.sample(frac=sampling_rate).reset_index(drop=True)
        self.rows.fillna(value="", inplace=True)

        self.rows['title_lower'] = self.rows['title'].apply(TextUtils.lower_case)
        self.rows['title_clean'] = self.rows['title_lower'].apply(TextUtils.remove_punctuation)
        self.rows['body_lower'] = self.rows['body'].apply(TextUtils.lower_case)
        self.rows['body_clean'] = self.rows['body_lower'].apply(TextUtils.remove_punctuation)
        self.rows['combined_clean'] = np.vectorize(TextUtils.combine_title_body)(self.rows['title_clean'], self.rows['body_clean'])

        # Drop columns not needed.
        columns_to_remove = ['title', 'body', 'score', 'tags', 'creation_date', 'mask', 'title_lower', 'body_lower', 'tags_list_all']
        for col in columns_to_remove:
            self.rows = self.rows.drop(col, axis=1)



class Featurizer(object):

    @staticmethod
    def getBasicFeatures(post_title, post_body, tag):
        '''
        Get basic features for new question entered on web append.
        '''
        new_post_in_title = pd.DataFrame([Featurizer.is_in_title(post_title, tag)], columns=['is_in_title'])
        new_post_in_body = pd.DataFrame([Featurizer.frac_in_body(post_body, tag)], columns=['frac_in_body'])
        return (pd.concat([new_post_in_title, new_post_in_body], axis=1))

    @staticmethod
    def is_in_title(title_text, tag):
        return float(tag in title_text.split())

    @staticmethod
    def frac_in_body(body_text, tag):
        denominator = float(len(body_text.split()))
        if denominator == 0:
            return 0
        else:
            return float(body_text.count(tag))/denominator

    @staticmethod
    def assign_label(text, tag):
        if tag in text.split('|'):
            return 1
        else:
            return 0

    @staticmethod
    def get_negative_tags(positive_tags, all_tags, k, seed):
        assert(isinstance(all_tags, list))
        random.seed(seed)
        negative_tags = set()
        for idx in range(len(all_tags) * 10):
            tag_to_add = all_tags[random.randint(0, len(all_tags)-1)]
            if tag_to_add not in positive_tags:
                if len(negative_tags) >= k:
                    return negative_tags
                else:
                    negative_tags.add(tag_to_add)
        return negative_tags

    def __init__(self, posts, negative_sampling_rate, top_tags):
        assert(negative_sampling_rate > 0)

        data_array = []
        for idx, row in posts.rows.iterrows():
            if idx % 200 == 0:
                print(idx)
            question_id = row['id']
            for tag in row['tags_list']:
                if tag not in top_tags:
                    raise 'Illegal argument exception: ' + tag
                in_title = Featurizer.is_in_title(row['title_clean'], tag)
                in_body = Featurizer.frac_in_body(row['body_clean'], tag)
                label = 1
                data_array.append([in_title, in_body, label, question_id, tag])
            negative_tags = Featurizer.get_negative_tags(row['tags_list'], top_tags, negative_sampling_rate, row['id'])
            for tag in negative_tags:
                in_title = Featurizer.is_in_title(row['title_clean'], tag)
                in_body = Featurizer.frac_in_body(row['body_clean'], tag)
                label = 0
                data_array.append([in_title, in_body, label, question_id, tag])

        self.features_for_training = ['is_in_title', 'frac_in_body']
        self.data = pd.DataFrame(data_array, columns=['is_in_title', 'frac_in_body', 'label', 'id', 'tag'])



class TfidfFeaturizer(object):

    @staticmethod
    def getTfidfFeatures(new_post, tag, top_words_for_tag):
        words_in_txt = new_post.split()
        top_words = top_words_for_tag[tag]
        text_length = len(words_in_txt)
        return [TfidfFeaturizer.safeDiv(words_in_txt.count(word), text_length) for word in top_words]

    def tokens(self, x):
        StopWords = set(['able', 'run', 'want', 'using', 'need', 'does', 'just', 'dont', 'make', 'user', 'trying',
                        'like', 'new', 'able', 'like', 'new', 'able', 'use', 'error', 'code', 'pwhat', 'pwhen',
                         'pthanks', 'relnofollow', 'pthe', 'thisp', 'pwhen', 'pbut', 'pis', 'phow', 'pmy',
                        'p', 'n', 'l', 'b', 'pwe', 'im', 'x', 'y', '0', '1', '2', '3', '4', '5', '6', '7',
                         '8', '9', '10'])
        return [token for token in x.split() if token not in StopWords and len(token) >= self.MIN_WORD_LENGTH]

    def get_top_k_words_for_tag(self, tags_dict):
        top_k_tag_words = dict()
        for tag in tags_dict:
            top_k_tag_words[tag] = heapq.nlargest(
                self.TOP_K_FOR_TAG,
                tags_dict[tag].keys(),
                key=lambda x: tags_dict[tag][x]
            )
        return top_k_tag_words

    @staticmethod
    def safeDiv(x, y):
        return 0 if y == 0 else float(x)/y

    def words_in_text(self, x, tag, words_dict):
        assert(len(words_dict[tag]) == self.TOP_K_FOR_TAG)
        return TfidfFeaturizer.getTfidfFeatures(x, tag, words_dict)

    def __init__(self, posts, basic_features, top_tags):
        self.TOP_K_FOR_DOCUMENT = 8
        self.TOP_K_FOR_TAG = 20
        self.MIN_WORD_LENGTH = 0

        self.top_tags = top_tags

        vectorizer = TfidfVectorizer(
            tokenizer=self.tokens,
            encoding='utf-8',
            max_df = 0.80,
            min_df = 0.01,
            stop_words='english'
        )

        tfidf = vectorizer.fit_transform(
            posts.rows['combined_clean']
        )
        tfidf_array = tfidf.toarray()

        # Get feature names
        tfidf_feature_names = vectorizer.get_feature_names()

        terms_list = []
        for l in tfidf_array:
            terms_list.append([tfidf_feature_names[x] for x in (-1 * l).argsort()][:self.TOP_K_FOR_DOCUMENT])

        tag_word_count = dict()
        for idx, row in posts.rows.iterrows():
            for current_tag in row['tags_list']:
                if current_tag in self.top_tags:
                    if current_tag not in tag_word_count:
                        tag_word_count[current_tag] = dict()

                    current_count_of_top_words_for_tag = tag_word_count[current_tag]

                    top_words_in_document = terms_list[idx]

                    for word in top_words_in_document:
                        if word not in current_count_of_top_words_for_tag:
                            current_count_of_top_words_for_tag[word] = 0
                        current_count_of_top_words_for_tag[word] += 1

                    tag_word_count[current_tag] = current_count_of_top_words_for_tag

        self.top_k_words_for_tag = self.get_top_k_words_for_tag(tag_word_count)


        # Iterate on the basic features and compute the words_in_text only for
        # the (document, tag) pairs in the basic_features.data dataframe.
        top_words_in_text = []
        for idx, row in basic_features.data.iterrows():
            if idx % 200 == 0:
                print(idx)
            text = posts.rows[posts.rows['id'] == row['id']].iloc[0]['combined_clean']
            top_words_in_text.append(self.words_in_text(text, row['tag'], self.top_k_words_for_tag))

        self.features_for_training = [str(x) for x in range(self.TOP_K_FOR_TAG)]
        self.data = pd.DataFrame(top_words_in_text, columns=self.features_for_training)



class BuildModel(object):

    def get_test_metrics(self):
        pred = self.final_clf.predict(self.test_X)
        print "Random Forest Classifier"
        print "Accuracy", accuracy_score(self.test_Y, pred)
        print "ROC-AUC", roc_auc_score(self.test_Y, pred)
        print "F1", f1_score(self.test_Y, pred)
        print "Recall", recall_score(self.test_Y, pred)
        print "Precision", precision_score(self.test_Y, pred)
        print "Confusion Matrix\n", confusion_matrix(self.test_Y, pred)
        print()
        pred = self.final_clf.predict(self.test_hard_X)
        print "Random Forest Classifier"
        print "Accuracy", accuracy_score(self.test_hard_Y, pred)
        print "ROC-AUC", roc_auc_score(self.test_hard_Y, pred)
        print "F1", f1_score(self.test_hard_Y, pred)
        print "Recall", recall_score(self.test_hard_Y, pred)
        print "Precision", precision_score(self.test_hard_Y, pred)
        print "Confusion Matrix\n", confusion_matrix(self.test_hard_Y, pred)
    
    def get_confusion_matrix(self):
        pred = self.final_clf.predict(self.test_hard_X)
        return confusion_matrix(self.test_hard_Y, pred)
        
    

    def __init__(self, basic_f, tfidf_f):
        joined_features = pd.concat(
            [basic_f.data[basic_f.features_for_training] , tfidf_f.data[tfidf_f.features_for_training], basic_f.data['label']],
            axis=1)

        train_len = int(0.8 * len(joined_features))
        test_len = len(joined_features) - train_len

        joined_features = shuffle(joined_features)

        train_X = joined_features.iloc[:train_len, :-1]
        train_Y = joined_features['label'].iloc[:train_len]

        self.test_X = joined_features.iloc[test_len:, :-1]
        self.test_Y = joined_features['label'].iloc[test_len:]

        hard_cases_mask = (self.test_X['is_in_title'] == 0.0) & (self.test_X['frac_in_body'] == 0.0)

        self.test_hard_X = self.test_X[hard_cases_mask]
        self.test_hard_Y = self.test_Y[hard_cases_mask]

        print(train_X.head(3))
        print(train_Y.head(3))

        print(len(self.test_hard_X))
        print(len(self.test_hard_Y))

        self.rfc = RandomForestClassifier()
        

        '''
        pipe = Pipeline([ ('rfc', self.rfc)])
        param = {
            'rfc__n_estimators': [100, 200, 300, 400 ],
            'rfc__criterion': ['gini', 'entropy'],
            'rfc__min_samples_split': [2, 3]
        }
        gs = GridSearchCV(pipe, param_grid=param, cv=3,n_jobs=-1, scoring='f1')
        gs.fit(train_X, train_Y)

        print 'Best score: %0.3f' % gs.best_score_
        print 'Best parameters set:'
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(param.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name])

        self.final_clf=gs.best_estimator_
        '''
        self.final_clf=self.rfc.fit(train_X, train_Y)



class Prediction(object):

    def __init__(self, model, top_words_for_tag, tags_for_prediction):
        self.model = model
        self.top_words_for_tag = top_words_for_tag
        self.tags = tags_for_prediction
        assert(len(self.tags) > 0)
        for tag in self.tags:
            assert(tag in self.top_words_for_tag)
        for tag in self.top_words_for_tag:
            assert(len(self.top_words_for_tag[tag]) == len(self.top_words_for_tag[self.tags[0]]))

    def predict(self, new_post_title, new_post_body):

        new_post_title_lower = TextUtils.lower_case(new_post_title)
        new_post_title_clean = TextUtils.remove_punctuation(new_post_title_lower)

        new_post_body_lower = TextUtils.lower_case(new_post_body)
        new_post_body_clean = TextUtils.remove_punctuation(new_post_body_lower)

        new_post_combined = TextUtils.combine_title_body(new_post_title_clean, new_post_body_clean)

        new_post_basic_features = pd.DataFrame(columns=[])
        new_post_tfidf_features = pd.DataFrame(columns=[])
        new_post_tfidf_features_array = []

        for tag in self.tags:
            new_post_basic_features = new_post_basic_features.append(
                Featurizer.getBasicFeatures(
                    new_post_title_clean,
                    new_post_body_clean,
                    tag
                ),
                ignore_index=True
            )
            new_post_tfidf_features_array.append(
                TfidfFeaturizer.getTfidfFeatures(new_post_combined, tag, self.top_words_for_tag)
            )

        tfidf_columns_names = [str(x) for x in range(len(new_post_tfidf_features_array[0]))]
        new_post_tfidf_features = pd.DataFrame(new_post_tfidf_features_array, columns=tfidf_columns_names)

        # print(new_post_basic_features.head(100))
        # print(new_post_tfidf_features.head(100))

        joined_new_post_features = pd.concat(
            [new_post_basic_features, new_post_tfidf_features],
            axis=1
        ).as_matrix()

        prob_list = []
        for tag, q in zip(self.tags, joined_new_post_features):
            predictions = self.model.predict_proba(q)[0]
            prob_list.append((predictions[0], predictions[1], tag))
        return sorted(prob_list)

