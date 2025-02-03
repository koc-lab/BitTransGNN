import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
import sys
from transformers import AutoTokenizer

#if len(sys.argv) != 2:
#	sys.exit("Use: python build_graph.py <dataset>")

#datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
#datasets = ["cola", "mrpc", "rte", "stsb", "wnli", '20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
# inductive

def reconstruct_sentences(tokens, tokenizer_pre_model="bert-base-uncased"):
    reconstructed_sentence = ""
    if tokenizer_pre_model == "bert-base-uncased":
        for token in tokens:
            if token.startswith("##"):
                reconstructed_sentence += token[2:]  # Remove "##" and attach it to the last word
            else:
                reconstructed_sentence += " " + token  # Add space before a new word
    elif tokenizer_pre_model == "roberta-base":
        reconstructed_sentence = "".join(tokens).replace("Ġ", " ")
    return reconstructed_sentence.strip()

if __name__ == "__main__":

    datasets = ["cola", "mrpc", "rte", "stsb", "wnli", '20ng', 'R8', 'R52', 'ohsumed', 'mr']
    glue_datasets = ["cola", "mrpc", "rte", "stsb", "wnli"]

    for dataset in datasets:
        if dataset in glue_datasets:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            #tokenizer.tokenize()

        random.seed(0)
        np.random.seed(0)

        doc_name_list = []
        doc_train_list = []
        doc_test_list = []
        doc_content_list = []
        tokenized_doc_content_list = []

        f = open('dataset/' + dataset + '.txt', 'r')
        f_clean = open('dataset/corpus/' + dataset + '.clean.txt', 'r')
        lines = f.readlines()
        clean_lines = f_clean.readlines()
        for line, clean_line in zip(lines, clean_lines):
            if clean_line == "\n":
                continue
            temp = line.split("\t")

            if temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
                doc_name_list.append(line.strip())
            doc_content_list.append(clean_line.strip())
            if dataset in glue_datasets:
                tokenized_clean_line = tokenizer.tokenize(clean_line.strip())
                tokenized_clean_line = reconstruct_sentences(tokenized_clean_line, tokenizer_pre_model="bert-base-uncased")
                tokenized_doc_content_list.append(tokenized_clean_line)
            else:
                tokenized_doc_content_list.append(clean_line.strip())

        f.close()
        f_clean.close()
        print(doc_name_list)
        print(doc_train_list)
        print(doc_test_list)
        print(doc_content_list)

        dataset += "_inductive"

        train_ids = []
        for train_name in doc_train_list:
            train_id = doc_name_list.index(train_name)
            train_ids.append(train_id)
        print(train_ids)
        random.shuffle(train_ids)
        print(train_ids)
        train_ids_str = '\n'.join(str(index) for index in train_ids)
        f = open('dataset/' + dataset + '.train.index', 'w')
        f.write(train_ids_str)
        f.close()

        test_ids = []
        test_size = len(test_ids)

        ids = train_ids + test_ids
        print(ids)
        print(len(ids))

        shuffle_doc_name_list = []
        shuffle_doc_words_list = []
        shuffle_tokenized_doc_words_list = []
        for id in ids:
            shuffle_doc_name_list.append(doc_name_list[int(id)])
            shuffle_doc_words_list.append(doc_content_list[int(id)])
            if dataset in glue_datasets:
                shuffle_tokenized_doc_words_list.append(tokenized_doc_content_list[int(id)])
            else:
                shuffle_tokenized_doc_words_list.append(doc_content_list[int(id)])
        shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
        shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
        print(shuffle_doc_name_list)
        print(shuffle_doc_words_list)
        print(shuffle_tokenized_doc_words_list)
        f = open('dataset/' + dataset + '_shuffle.txt', 'w')
        f.write(shuffle_doc_name_str)
        f.close()

        f = open('dataset/corpus/' + dataset + '_shuffle.txt', 'w')
        f.write(shuffle_doc_words_str)
        f.close()

        word_freq = {}
        word_set = set()
        for doc_words in shuffle_tokenized_doc_words_list:
            words = doc_words.split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        vocab = list(word_set)
        vocab_size = len(vocab)
        print(vocab)
        print(vocab_size)

        word_doc_list = {}

        for i in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        print(word_doc_list)

        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)

        print(word_doc_freq)

        word_id_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

        print(word_id_map)
        print(max(word_id_map.values()))

        vocab_str = '\n'.join(vocab)
        f = open('dataset/corpus/' + dataset + '_vocab.txt', 'w')
        f.write(vocab_str)
        f.close()

        # label list
        label_set = set()
        for doc_meta in shuffle_doc_name_list:
            temp = doc_meta.split('\t')
            label_set.add(temp[2])
        label_list = list(label_set)
        label_list_str = '\n'.join(label_list)
        print(label_list)
        print(label_set)
        f = open('dataset/corpus/' + dataset + '_labels.txt', 'w')
        f.write(label_list_str)
        f.close()

        # x: feature vectors of training docs, no initial features
        # select 90% training set
        train_size = len(train_ids)
        val_size = int(0.1 * train_size)
        real_train_size = train_size - val_size  # - int(0.5 * train_size)
        # different training rates

        real_train_doc_names = shuffle_doc_name_list[:real_train_size]
        real_train_doc_names_str = '\n'.join(real_train_doc_names)
        f = open('dataset/' + dataset + '.real_train.name', 'w')
        f.write(real_train_doc_names_str)
        f.close()

        row_x = []
        col_x = []
        data_x = []
        word_embeddings_dim = 300
        word_vector_map = {}
        for i in range(real_train_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_x.append(doc_vec[j] / doc_len)

        # x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            real_train_size, word_embeddings_dim))

        print(shuffle_doc_name_list)
        if dataset == "stsb":
            y = []
            for i in range(real_train_size):
                doc_meta = shuffle_doc_name_list[i]
                temp = doc_meta.split('\t')
                label = temp[2]
                y.append(label)
            y = np.array(y)
            y = y.astype(float)
            if y.ndim == 1:
                y = np.expand_dims(y, axis=-1)
            print(y)
            print(y.shape)
        else:
            y = []
            for i in range(real_train_size):
                doc_meta = shuffle_doc_name_list[i]
                temp = doc_meta.split('\t')
                label = temp[2]
                one_hot = [0 for l in range(len(label_list))]
                label_index = label_list.index(label)
                one_hot[label_index] = 1
                y.append(one_hot)
            y = np.array(y)
            print(y)
            print(y.shape)

        # allx: the the feature vectors of both labeled and unlabeled training instances
        # (a superset of x)
        # unlabeled training instances -> words

        word_vectors = np.random.uniform(-0.01, 0.01,
                                        (vocab_size, word_embeddings_dim))

        for i in range(len(vocab)):
            word = vocab[i]
            if word in word_vector_map:
                vector = word_vector_map[word]
                word_vectors[i] = vector

        row_allx = []
        col_allx = []
        data_allx = []

        for i in range(train_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in word_vector_map: #no effect since we don't use external word embeddings
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        for i in range(vocab_size):
            for j in range(word_embeddings_dim):
                row_allx.append(int(i + train_size))
                col_allx.append(j)
                data_allx.append(word_vectors.item((i, j)))


        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix(
            (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))
        print(train_size)
        print(vocab_size)
        print(allx[train_size:,])

        if dataset == "stsb":
            ally = []
            for i in range(train_size):
                doc_meta = shuffle_doc_name_list[i]
                temp = doc_meta.split('\t')
                label = temp[2]
                ally.append(label)
            
            for i in range(vocab_size):
                ally.append("0")
            
            ally = np.array(ally)
            ally = ally.astype(float)
            if ally.ndim == 1:
                ally = np.expand_dims(ally, axis=-1)
            print(ally)
            print(ally.shape)

        else:
            ally = []
            for i in range(train_size):
                doc_meta = shuffle_doc_name_list[i]
                temp = doc_meta.split('\t')
                label = temp[2]
                one_hot = [0 for l in range(len(label_list))]
                label_index = label_list.index(label)
                one_hot[label_index] = 1
                ally.append(one_hot)

            for i in range(vocab_size):
                one_hot = [0 for l in range(len(label_list))]
                ally.append(one_hot)
            
            ally = np.array(ally)

        print(x.shape, y.shape, allx.shape, ally.shape)
        print("real_train_size: ", real_train_size)
        print("val_size: ", val_size)
        print("test_size: ", test_size)
        print("word_embeddings_dim: ", word_embeddings_dim)
        print("vocab_size: ", vocab_size)

        # word co-occurence with context windows
        window_size = 20
        windows = []
        print(shuffle_tokenized_doc_words_list)

        for doc_words in shuffle_tokenized_doc_words_list:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
                    # print(window)
        print(windows)

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        print(word_window_freq)
        print(word_id_map)

        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    if word_i_id == word_j_id:
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        print(word_pair_count)

        #print("Doc-word complete.")
        print("Word co-occurence within documents complete.")

        # pmi as weights

        num_window = len(windows)
        print(num_window)
        print(word_pair_count)

        row = []
        col = []
        weight = []

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = word_pair_count[key]
            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]
            pmi = log((1.0 * count / num_window) /
                    (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

        print("pmi complete.")
        print(row)
        print(col)
        print(weight)

        # doc word frequency
        doc_word_freq = {}

        print(shuffle_tokenized_doc_words_list)
        print(word_id_map)

        for doc_id in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < train_size: #will always enter here for inductive
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                col.append(train_size + j)
                idf = log(1.0 * len(shuffle_tokenized_doc_words_list) /
                        word_doc_freq[vocab[j]])
                weight.append(freq * idf) #freq defined as raw count
                doc_word_set.add(word)

        print(doc_word_freq)
        print("doc-word freq complete.")
        node_size = train_size + vocab_size + test_size
        adj = sp.csr_matrix(
            (weight, (row, col)), shape=(node_size, node_size))
        print("adj complete.")

        f = open("dataset/ind.{}.x".format(dataset), 'wb')
        pkl.dump(x, f)
        f.close()

        f = open("dataset/ind.{}.y".format(dataset), 'wb')
        pkl.dump(y, f)
        f.close()

        f = open("dataset/ind.{}.allx".format(dataset), 'wb')
        pkl.dump(allx, f)
        f.close()

        f = open("dataset/ind.{}.ally".format(dataset), 'wb')
        pkl.dump(ally, f)
        f.close()

        f = open("dataset/ind.{}.adj".format(dataset), 'wb')
        pkl.dump(adj, f)
        f.close()

        print(f"build_graph.py complete for {dataset} dataset.")
