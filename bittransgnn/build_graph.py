import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph as knn_graph
from math import log
import sys
from transformers import AutoTokenizer

#datasets = ["cola", "mrpc", "rte", "stsb", "wnli", '20ng', 'R8', 'R52', 'ohsumed', 'mr']
#datasets = ["cola"]
# transductive
# build corpus

PAIR_DATASETS = ["mrpc", "rte", "stsb"]
SEP_WORD = "<<SEP>>"

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


    #datasets = ["cola", "mrpc", "rte", "stsb", '20ng', 'R8', 'R52', 'ohsumed', 'mr']
    #datasets = ["cola"]
    #datasets = ["R8"]
    #datasets = ["mrpc", "rte", "stsb"]
    datasets = ["mrpc"]
    glue_datasets = ["cola", "mrpc", "rte", "stsb", "wnli"]
    
    for dataset in datasets:
        print("dataset", dataset)
        if dataset in glue_datasets:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            #tokenizer.tokenize()

        if dataset in PAIR_DATASETS:
            BASE = "dataset_paired"
        else:
            BASE = "dataset"

        print("BASE", BASE)

        random.seed(0)
        np.random.seed(0)

        doc_name_list = []
        doc_train_list = []
        doc_test_list = []
        doc_content_list = []
        tokenized_doc_content_list = []

        f = open(f'{BASE}/' + dataset + '.txt', 'r')
        f_clean = open(f'{BASE}/corpus/' + dataset + '.clean.txt', 'r')
        lines = f.readlines()
        clean_lines = f_clean.readlines()
        for line, clean_line in zip(lines, clean_lines):
            if clean_line == "\n":
                continue
            doc_name_list.append(line.strip())
            temp = line.split("\t")

            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())

            raw = clean_line.strip()

            #doc_content_list.append(clean_line.strip())
            # 1) keep original (with <<SEP>>) for BERT side
            doc_content_list.append(raw)

            # 2) GRAPH VIEW: tokenize + reconstruct; for pairs, concat A+B (no SEP)
            if dataset in glue_datasets:
                if (dataset in PAIR_DATASETS) and (SEP_WORD in raw):
                    left, right = [s.strip() for s in raw.split(SEP_WORD, 1)]
                    t_left  = tokenizer.tokenize(left)
                    t_right = tokenizer.tokenize(right)
                    rec_left  = reconstruct_sentences(t_left,  tokenizer_pre_model="bert-base-uncased")
                    rec_right = reconstruct_sentences(t_right, tokenizer_pre_model="bert-base-uncased")
                    graph_text = f"{rec_left} {rec_right}"          # concat without sentinel
                else:
                    t = tokenizer.tokenize(raw)
                    tokenized_clean_line = reconstruct_sentences(t, tokenizer_pre_model="bert-base-uncased")
                    graph_text = tokenized_clean_line               # <-- define graph_text here
                tokenized_doc_content_list.append(" ".join(graph_text.split()))
            else:
                # non-GLUE or other corpora: just strip sentinel if present
                tokenized_doc_content_list.append(" ".join(raw.replace(SEP_WORD, " ").split()))

        f.close()
        f_clean.close()
        #print(doc_name_list)
        #print(doc_train_list)
        #print(doc_test_list)
        #print(doc_content_list)

        if dataset == "mrpc":
            # train_ids in FILE ORDER (so appended val is at the tail)
            train_ids = []
            for train_name in doc_train_list:
                train_id = doc_name_list.index(train_name)
                train_ids.append(train_id)

            # Keep appended validation at the end, but shuffle within each partition.
            # val_size is the 10% you later use in build_graph.py
            val_size = int(0.1 * len(train_ids))
            if val_size > 0:
                core = train_ids[:-val_size]   # original train block
                tail = train_ids[-val_size:]   # appended validation block
                random.shuffle(core)
                random.shuffle(tail)
                train_ids = core + tail
            else:
                random.shuffle(train_ids)  # fallback for tiny datasets
        
        else:
            train_ids = []
            for train_name in doc_train_list:
                train_id = doc_name_list.index(train_name)
                train_ids.append(train_id)
            #print(train_ids)
            random.shuffle(train_ids)
            #print(train_ids)

        train_ids_str = '\n'.join(str(index) for index in train_ids)
        f = open(f"{BASE}/" + dataset + '.train.index', 'w')
        f.write(train_ids_str)
        f.close()

        test_ids = []
        for test_name in doc_test_list:
            test_id = doc_name_list.index(test_name)
            test_ids.append(test_id)
        #print(test_ids)
        random.shuffle(test_ids)
        #print(test_ids)
        test_ids_str = '\n'.join(str(index) for index in test_ids)
        f = open(f"{BASE}/" + dataset + '.test.index', 'w')
        f.write(test_ids_str)
        f.close()

        ids = train_ids + test_ids
        #print(ids)
        #print(len(ids))

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
        #print(shuffle_doc_name_list)
        #print(shuffle_doc_words_list)
        #print(shuffle_tokenized_doc_words_list)
        f = open(f"{BASE}/" + dataset + '_shuffle.txt', 'w')
        f.write(shuffle_doc_name_str)
        f.close()

        f = open(f"{BASE}/corpus/" + dataset + '_shuffle.txt', 'w')
        f.write(shuffle_doc_words_str)
        f.close()

        # AFTER
        word_freq = {}
        word_set = set()
        for doc_words in shuffle_tokenized_doc_words_list:
            words = doc_words.split()
            for word in words:
                if word == SEP_WORD:
                    continue
                word_set.add(word)
                word_freq[word] = word_freq.get(word, 0) + 1

        vocab = list(word_set)
        vocab_size = len(vocab)
        #print(vocab)
        #print(vocab_size)

        word_doc_list = {}

        # AFTER
        for i in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word == SEP_WORD or word in appeared:
                    continue
                word_doc_list.setdefault(word, []).append(i)
                appeared.add(word)

        #print(word_doc_list)

        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)

        #print(word_doc_freq)

        word_id_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i

        #print(word_id_map)
        #print(max(word_id_map.values()))

        vocab_str = '\n'.join(vocab)
        f = open(f"{BASE}/corpus/" + dataset + '_vocab.txt', 'w')
        f.write(vocab_str)
        f.close()

        # label list
        label_set = set()
        for doc_meta in shuffle_doc_name_list:
            temp = doc_meta.split('\t')
            label_set.add(temp[2])
        label_list = list(label_set)
        #label_list = sorted(list(label_set))
        label_list_str = '\n'.join(label_list)
        #print(label_list)
        #print(label_set)
        f = open(f"{BASE}/corpus/" + dataset + '_labels.txt', 'w')
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
        f = open(f"{BASE}/" + dataset + '.real_train.name', 'w')
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

        #print(shuffle_doc_name_list)
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
            #print(y)
            #print(y.shape)
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
            #print(y)
            #print(y.shape)

        # tx: feature vectors of test docs, no initial features
        test_size = len(test_ids)
        #print(test_size)

        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(test_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_tokenized_doc_words_list[i + train_size]
            words = doc_words.split()
            doc_len = len(words)
            for word in words:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                        shape=(test_size, word_embeddings_dim))

        if dataset == "stsb":
            ty = []
            for i in range(test_size):
                doc_meta = shuffle_doc_name_list[i + train_size]
                temp = doc_meta.split('\t')
                label = temp[2]
                ty.append(label)
            ty = np.array(ty)
            ty = ty.astype(float)
            if ty.ndim == 1:
                ty = np.expand_dims(ty, axis=-1)
            #print(ty)
            #print(ty.shape)
        else:
            ty = []
            for i in range(test_size):
                doc_meta = shuffle_doc_name_list[i + train_size]
                temp = doc_meta.split('\t')
                label = temp[2]
                one_hot = [0 for l in range(len(label_list))]
                label_index = label_list.index(label)
                one_hot[label_index] = 1
                ty.append(one_hot)
            ty = np.array(ty)
            #print(ty)
            #print(ty.shape)

        # allx: the the feature vectors of both labeled and unlabeled training instances
        # (a superset of x)
        # unlabeled training instances -> words

        word_vectors = np.random.uniform(-0.01, 0.01,
                                        (vocab_size, word_embeddings_dim))

        for i in range(len(vocab)):
            word = vocab[i]
            if word in word_vector_map: #yine bizim için bir etkisi yok
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
                if word in word_vector_map: #no effect
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
        #print(allx[train_size:,])

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

        print("x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape")
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
        print("real_train_size: ", real_train_size)
        print("val_size: ", val_size)
        print("test_size: ", test_size)
        print("word_embeddings_dim: ", word_embeddings_dim)
        print("vocab_size: ", vocab_size)

        # word co-occurence with context windows
        # AFTER
        window_size = 20
        windows = []

        def segments(words):
            segs, start = [], 0
            for i, w in enumerate(words):
                if w == SEP_WORD:            # hard boundary
                    if i > start:
                        segs.append(words[start:i])
                    start = i + 1
            if start < len(words):
                segs.append(words[start:])
            return [s for s in segs if s]     # drop empties

        for doc_words in shuffle_tokenized_doc_words_list:
            words = [w for w in doc_words.split() if w != SEP_WORD]  # keep SEP out of windows
            # build windows within each segment only
            for seg in segments([w for w in doc_words.split()] if SEP_WORD in doc_words else [w for w in doc_words.split()]):
                if not seg:
                    continue
                L = len(seg)
                if L <= window_size:
                    windows.append([w for w in seg if w != SEP_WORD])
                else:
                    for j in range(L - window_size + 1):
                        win = seg[j:j+window_size]
                        windows.append([w for w in win if w != SEP_WORD])

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

        #print(word_window_freq)
        #print(word_id_map)

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

        #print(word_pair_count)

        #print("Doc-word complete.")
        print("Word co-occurence within documents complete.")

        # pmi as weights

        num_window = len(windows)
        #print(num_window)
        #print(word_pair_count)

        row = []
        col = []
        weight = []

        row_word_word = []
        col_word_word = []
        weight_word_word = []

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
            row_word_word.append(train_size + i)
            col_word_word.append(train_size + j)
            weight_word_word.append(pmi)
        
        row.extend(row_word_word)
        col.extend(col_word_word)
        weight.extend(weight_word_word)

        print("pmi complete.")

        # doc word frequency
        doc_word_freq = {}

        #print(shuffle_tokenized_doc_words_list)
        #print(word_id_map)

        # AFTER
        for doc_id in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                if word == SEP_WORD:
                    continue
                j = word_id_map[word]
                key = f"{doc_id},{j}"
                doc_word_freq[key] = doc_word_freq.get(key, 0) + 1

        row_word_doc = []
        col_word_doc = []
        weight_word_doc = []

        for i in range(len(shuffle_tokenized_doc_words_list)):
            doc_words = shuffle_tokenized_doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word == SEP_WORD or word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j) #doc_id , word_id
                freq = doc_word_freq[key]
                if i < train_size:
                    row_word_doc.append(i)
                else:
                    row_word_doc.append(i + vocab_size)
                col_word_doc.append(train_size + j)
                idf = log(1.0 * len(shuffle_tokenized_doc_words_list) /
                        word_doc_freq[vocab[j]])
                weight_word_doc.append(freq * idf) #freq defined as raw count
                doc_word_set.add(word)

        row.extend(row_word_doc)
        col.extend(col_word_doc)
        weight.extend(weight_word_doc)

        #print(doc_word_freq)
        print("doc-word freq complete.")

        tfidf_vectorizer = TfidfVectorizer(
            norm=None,  # No normalization
            sublinear_tf=False,  # Use raw count
            smooth_idf=False,  # No smoothing
            use_idf=True
        )
        #tfidf_matrix = tfidf_vectorizer.fit_transform(shuffle_tokenized_doc_words_list)
        cleaned_docs = [s.replace(SEP_WORD, " ").strip() for s in shuffle_tokenized_doc_words_list]
        tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_docs)

        # Compute nearest neighbors graph (top 25 neighbors, cosine similarity)
        knn = knn_graph(tfidf_matrix, n_neighbors=25, metric='cosine', mode='distance', include_self=False)
        knn = knn.tocoo()  # Convert to COO format for easy iteration

        doc_size = knn.shape[0]

        row_doc_doc = []
        col_doc_doc = []
        weight_doc_doc = []

        # Add doc-doc edges to row, col, weight
        for i, j, sim in zip(knn.row, knn.col, knn.data):
            # Convert cosine distance to similarity (1 - distance)
            cosine_sim = 1.0 - sim
            if cosine_sim > 0:  # Only add positive similarities
                row_doc_doc.append(i)
                col_doc_doc.append(j)
                weight_doc_doc.append(cosine_sim)

        row.extend(row_doc_doc)
        col.extend(col_doc_doc)
        weight.extend(weight_doc_doc)

        node_size = train_size + vocab_size + test_size
        adj = sp.csr_matrix(
            (weight, (row, col)), shape=(node_size, node_size))
        print("adj complete.")

        weight_no_doc = weight_word_word + weight_word_doc
        row_no_doc = row_word_word + row_word_doc
        col_no_doc = col_word_word + col_word_doc
        adj_no_doc = sp.csr_matrix(
            (weight_no_doc, (row_no_doc, col_no_doc)), shape=(node_size, node_size))

        weight_no_ww = weight_word_doc + weight_doc_doc
        row_no_ww = row_word_doc + row_doc_doc
        col_no_ww = col_word_doc + col_doc_doc
        adj_no_ww = sp.csr_matrix(
            (weight_no_ww, (row_no_ww, col_no_ww)), shape=(node_size, node_size))

        node_size_doc_doc = train_size + test_size
        adj_doc_doc = sp.csr_matrix(
            (weight_doc_doc, (row_doc_doc, col_doc_doc)), shape=(node_size_doc_doc, node_size_doc_doc))
        
        print("x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape")
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
        print("dataset")
        print(dataset)
        print("adj.shape")
        print(adj.shape)
        print(adj.count_nonzero())
        print("adj_no_doc.shape")
        print(adj_no_doc.shape)
        print(adj_no_doc.count_nonzero())
        print("adj_no_ww.shape")
        print(adj_no_ww.shape)
        print(adj_no_ww.count_nonzero())
        print("adj_doc_doc.shape")
        print(adj_doc_doc.shape)
        print(adj_doc_doc.count_nonzero())

        output_file_path = f"dataset_stats/{dataset}.txt"
        with open(output_file_path, "w") as output_file:
            output_file.write("dataset\n")
            output_file.write(f"{dataset}\n")
            output_file.write("adj.shape\n")
            output_file.write(f"{adj.shape}\n")
            output_file.write("{adj.count_nonzero()}\n")
            output_file.write(f"{adj.count_nonzero()}\n")
            output_file.write("adj_no_doc.shape\n")
            output_file.write(f"{adj_no_doc.shape}\n")
            output_file.write("{adj_no_doc.count_nonzero()}\n")
            output_file.write(f"{adj_no_doc.count_nonzero()}\n")
            output_file.write("adj_no_ww.shape\n")
            output_file.write(f"{adj_no_ww.shape}\n")
            output_file.write("{adj_no_ww.count_nonzero()}\n")
            output_file.write(f"{adj_no_ww.count_nonzero()}\n")
            output_file.write("adj_doc_doc.shape\n")
            output_file.write(f"{adj_doc_doc.shape}\n")
            output_file.write("{adj_doc_doc.count_nonzero()}\n")
            output_file.write(f"{adj_doc_doc.count_nonzero()}\n")

        f = open(f"{BASE}/ind.{dataset}.x", 'wb')
        pkl.dump(x, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.y", 'wb')
        pkl.dump(y, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.tx", 'wb')
        pkl.dump(tx, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.ty", 'wb')
        pkl.dump(ty, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.allx", 'wb')
        pkl.dump(allx, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.ally", 'wb')
        pkl.dump(ally, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.adj", 'wb')
        pkl.dump(adj, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.adj_no_doc", 'wb')
        pkl.dump(adj_no_doc, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.adj_no_ww", 'wb')
        pkl.dump(adj_no_ww, f)
        f.close()

        f = open(f"{BASE}/ind.{dataset}.adj_doc_doc", 'wb')
        pkl.dump(adj_doc_doc, f)
        f.close()

        print(f"build_graph.py complete for {dataset} dataset.")
        