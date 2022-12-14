import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE, Isomap
import torch
from torchtext import data
from torchtext import datasets
import torch.optim as optim
from torchtext.data import TabularDataset
import random
import torch.nn as nn
from torchnlp import nn as nlpnn
import numpy as np
import time
import torch.nn as nn
from torchnlp import nn as nlpnn
import numpy as np
import plotly
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pickle
import json
from collections import namedtuple
from allennlp.modules.elmo import Elmo, batch_to_ids


current_batch = namedtuple("batched_data", "text label sentences")


def split_data(filename, fraction):
    df = pd.read_csv(filename)
    df = shuffle(df)
    train_len = int(fraction * len(df))
    train = df.iloc[0:train_len]
    valid = df.iloc[train_len:]
    test = df.iloc[train_len:]
    train.to_csv(DATA_FOLDER + "/Train.csv", index=False, header=False)
    valid.to_csv(DATA_FOLDER + "/Valid.csv", index=False, header=False)
    test.to_csv(DATA_FOLDER + "/Test.csv", index=False, header=False)


def custom_threshold(l, t):
    if l > t:
        return 1.0
    else:
        return 0.0


class VisClass:
    def __init__(self):
        pass

    def export_vis(self, X_in, filename, title):
        save_dict = {
            "X": X_in,
            "title": title,
            "text": self.df["text"],
            "label": self.df["label"],
            "tweet_id": self.df["user_id"],
            "score": self.df["confidence"],
            "score_tag": self.df["score_tag"],
            "attention_weights": self.df["normalized_attention_weights"],
        }

        f = open(filename, "wb")
        # json.dump(save_dict, f)
        pickle.dump(save_dict, f)
        f.close()

    def plotly_2d_scattergl_embeddings_single(self, X_reduced, filename, title):

        # balance = self.cmocean_to_plotly(cmocean.cm.balance, len(X_reduced))

        trace1 = go.Scattergl(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            mode="markers",
            text=self.df["text"],
            marker=dict(
                color=np.array(self.df["label"]),
                opacity=0.5,
                size=15,
                colorscale="Jet",
                line=dict(color="#EBFBEC", width=2),
            ),
            name="embeddings ",
        )

        data = [trace1]
        layout = go.Layout(
            title=title,
            xaxis=dict(
                title="x component",
            ),
            yaxis=dict(
                title="y component",
            ),
        )
        fig = dict(data=data, layout=layout)
        offline.plot(fig, filename=filename)

    def vis(self, output, vis_filename, title):
        def get_projection(input_matrix, vis_type="pca"):
            if vis_type == "pca":
                proj = PCA(n_components=2)
            elif vis_type == "mds":
                input_matrix = input_matrix.astype(np.float64)
                proj = MDS(n_components=2, n_jobs=4)
            elif vis_type == "tsne":
                proj = TSNE(n_components=2)
            elif vis_type == "isomap":
                proj = proj = Isomap(n_components=2, n_jobs=4)

            X_r = proj.fit_transform(input_matrix)
            return X_r

        print("Reducing dimensionality for visualization")
        X_reduced = get_projection(output, vis_type="pca")
        X_reduced2 = get_projection(output, vis_type="mds")
        X_reduced_3 = get_projection(output, vis_type="tsne")
        X_reduced_4 = get_projection(output, vis_type="isomap")

        self.plotly_2d_scattergl_embeddings_single(
            X_reduced, vis_filename + "_pca.html", title + " PCA"
        )
        self.plotly_2d_scattergl_embeddings_single(
            X_reduced2, vis_filename + "_mds.html", title + " MDS"
        )
        self.plotly_2d_scattergl_embeddings_single(
            X_reduced_3, vis_filename + "_tsne_out.html", title + " Tsne"
        )
        self.plotly_2d_scattergl_embeddings_single(
            X_reduced_4, vis_filename + "_isomap_out.html", title + "Isomap"
        )

        # self.export_vis(X_reduced, 'Vis_data_pca.txt', 'PCA')
        # self.export_vis(X_reduced_3, 'Vis_data_tsne.txt', 'TSNE')


class IteratorClass:
    def __init__(self):
        self.get_data()
        self.get_character_ids()
        self.get_text_vocab()

        self.batch_parameters = {
            "batch_size": 128,
            "WRAP_AROUND": False,
            "RUN_TILL_END_RESTART": False,
            "RUN_TILL_END_AND_STOP": True,
        }

        print("Length of training data ", len(self.train_text_list))

        self.get_iterators()

    def get_iterators(self):

        self.train_iterator = self.get_train_iterator(self.batch_parameters)
        self.valid_iterator = self.get_valid_iterator(self.batch_parameters)
        self.test_iterator = self.get_test_iterator(self.batch_parameters)

    def get_data(self):
        self.df_train = pd.read_csv(DATA_FOLDER + "/Train.csv")
        self.df_valid = pd.read_csv(DATA_FOLDER + "/Valid.csv")
        self.df_test = pd.read_csv(DATA_FOLDER + "/Test.csv")

        self.df_train.columns = ["record_id", "text", "id", "label"]
        self.df_test.columns = ["record_id", "text", "id", "label"]
        self.df_valid.columns = ["record_id", "text", "id", "label"]

        print("Shuffling data")
        self.df_train = shuffle(self.df_train)
        self.df_valid = shuffle(self.df_valid)

        # self.df_train['tokenized'] = self.df_train['text'].apply(str.split)

    def get_character_ids(self):

        print("Creating character ids")
        self.train_text_list = list(self.df_train.text.values)
        self.train_labels = list(self.df_train.label.values)
        self.train_ids = batch_to_ids(self.train_text_list)

        self.valid_text_list = list(self.df_valid.text.values)
        self.valid_labels = list(self.df_valid.label.values)
        self.valid_ids = batch_to_ids(self.valid_text_list)

        self.test_text_list = list(self.df_test.text.values)
        self.test_labels = list(self.df_test.label.values)
        self.test_ids = batch_to_ids(self.test_text_list)

        # self.train_data = pd.DataFrame({ 'text': self.train_ids, 'label': self.train_labels})
        # self.valid_data = pd.DataFrame({ 'text': self.valid_ids, 'label': self.valid_labels})

    def get_text_vocab(self):
        self.device = device
        self.DEBUG = False
        self.TEXT = data.Field(tokenize="spacy", batch_first=True)
        self.LABEL = data.LabelField(dtype=torch.float)
        self.SEQ_LEN = data.Field(dtype=torch.int64, use_vocab=False, sequential=False)

        # self.tv_datafields = [("id",self.SEQ_LEN), ("text", self.TEXT), ("user_id", self.SEQ_LEN), ("label", self.LABEL)]
        self.tv_datafields = [
            ("id", self.SEQ_LEN),
            ("text", self.TEXT),
            ("user_id", self.SEQ_LEN),
            ("label", self.LABEL),
        ]

        self.trn, self.vld = TabularDataset.splits(
            path=DATA_FOLDER,  # the root directory where the data lies
            train="Train.csv",
            validation="Valid.csv",
            format="csv",
            skip_header=False,  # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=self.tv_datafields,
        )

        self.tst = TabularDataset(
            path=DATA_FOLDER + "/Test.csv",  # the root directory where the data lies
            format="csv",
            skip_header=False,  # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=self.tv_datafields,
        )

        self.TEXT.build_vocab(self.trn, max_size=75000, vectors="charngram.100d")
        self.LABEL.build_vocab(self.trn)

    def get_train_iterator(self, batch_parameters):
        return self.get_batch(
            self.train_ids, self.train_text_list, self.train_labels, batch_parameters
        )

    def get_valid_iterator(self, batch_parameters):
        return self.get_batch(
            self.valid_ids, self.valid_text_list, self.valid_labels, batch_parameters
        )

    def get_test_iterator(self, batch_parameters):
        return self.get_batch(
            self.test_ids, self.test_text_list, self.test_labels, batch_parameters
        )

    def get_batch(self, ids, text, labels, batch_parameters):

        curr = 0
        total_size = len(ids)
        DONE_FLAG = True

        while curr < total_size:
            start = curr
            end = (curr + batch_parameters["batch_size"]) % total_size

            # print("Current ",start,end, total_size)
            if (end < start) and batch_parameters["WRAP_AROUND"] == True:

                size_to_end = total_size - curr
                size_leftover = batch_parameters["batch_size"] - size_to_end

                data_chunk_to_end = ids[curr:]
                data_chunk_leftover = ids[0:end]

                label_chunk_to_end = labels[curr:]
                label_chunk_leftover = labels[0:end]

                text_chunk_to_end = text[curr:]
                text_chunk_leftover = text[0:end]

                data_return = data_chunk_to_end + data_chunk_leftover
                label_return = label_chunk_to_end + label_chunk_leftover
                text_return = text_chunk_to_end + text_chunk_leftover
                elem = current_batch(
                    data_return, torch.tensor(label_return), text_return
                )

                yield (elem)

            elif (end < start) and (
                (batch_parameters["RUN_TILL_END_RESTART"] == True)
                or (batch_parameters["RUN_TILL_END_AND_STOP"] == True)
            ):

                print("Getting to the end")
                DONE_FLAG = True
                elem = current_batch(
                    ids[curr:], torch.tensor(labels[curr:]), text[curr:]
                )
                yield (elem)

            else:

                elem = current_batch(
                    ids[start:end], torch.tensor(labels[start:end]), text[start:end]
                )
                yield (elem)

            if batch_parameters["RUN_TILL_END_RESTART"] == True:
                curr = curr + batch_parameters["batch_size"]
                if curr > total_size:
                    curr = 0
            if batch_parameters["RUN_TILL_END_AND_STOP"] == True:
                curr = curr + batch_parameters["batch_size"]
            if batch_parameters["WRAP_AROUND"] == True:
                curr = (curr + batch_parameters["batch_size"]) % total_size


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
    ):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)
        self.attention = nlpnn.Attention(hidden_dim * 2)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.params = {"embedding_dim": embedding_dim, "rnn_hidden_dim": hidden_dim}

    def forward(self, text, batch_lengths=[]):

        # text = [sent len, batch size]
        global output, hidden_orig1, hidden_orig2
        embedded = self.embedding(text)["elmo_representations"][0]
        # embedded = [sent len, batch size, emb dim]

        if PACKED_FLAG == False:
            output, (hidden, cell) = self.rnn(embedded)
            hf = torch.flip(output[:, :, self.params["rnn_hidden_dim"] :], dims=[1])
            res = torch.cat(
                (output[:, :, : self.params["rnn_hidden_dim"]], hf[:, :, :]), dim=2
            )
            attention_out, attention_weights = self.attention(
                res.contiguous(), res.contiguous()
            )
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, batch_first=True, lengths=batch_lengths
            )
            output, (hidden, cell) = self.rnn(packed)
            padded = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            attention_out, attention_weights = self.attention(
                padded[0].contiguous(), padded[0].contiguous()
            )

        attention_weights = torch.sum(attention_weights, dim=1)

        if DEBUG:
            print("Embedding layer", embedded.shape)
            print("Hidden layer ", hidden.shape)

        # output = [ batch size,  sent_len, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        hidden = self.dropout(torch.sum(attention_out, dim=1))

        if DEBUG == True:
            print(
                "Output of forward shape before squeeze",
                hidden.shape,
                hidden.squeeze(0).shape,
            )
            print("Output of attention layer", attention_out.shape)

        linear1 = self.fc1(hidden.squeeze(0))
        linear2 = self.fc2(linear1)
        sigmoid_res = self.sigmoid(linear2)
        return linear2, attention_weights, linear1, sigmoid_res
        # return(embedded)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    global DEBUG
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    data_total_length = 0

    for ct, batch in enumerate(iterator):

        optimizer.zero_grad()

        data = batch.text
        label = batch.label
        sentences = batch.sentences

        data = data.to(device)
        label = label.to(device)

        data_total_length += len(data)
        # print("From train ", ct, len(data), len(sentences), data_total_length)
        if PACKED_FLAG:
            data, label, batch_lengths = get_batch_lengths(data, label, sentences)
            predictions, attention_weights, _, _ = model(data.to(device), batch_lengths)
        else:
            predictions, attention_weights, _, _ = model(data.to(device))

        predictions = predictions.squeeze(1)

        label = label.cuda()

        label = label.type(torch.cuda.FloatTensor)

        loss = criterion(predictions, label)

        acc = binary_accuracy(predictions, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (ct % 100) == 0:
            print(ct)

        if DEBUG:
            print("Debug is ON, returning...")
            return epoch_loss / len(iterator), epoch_acc / len(iterator)

    ct += 1
    return epoch_loss / ct, epoch_acc / ct


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    data_total_length = 0

    with torch.no_grad():

        for ct, batch in enumerate(iterator):

            data = batch.text
            label = batch.label
            sentences = batch.sentences

            data = data.to(device)
            label = label.to(device)

            data_total_length += len(data)

            # print("From evaluate ", ct, len(data), len(sentences), data_total_length)

            if PACKED_FLAG:
                data, label, batch_lengths = get_batch_lengths(data, label, sentences)
                predictions, attention_weights, _, _ = model(data, batch_lengths)
            else:
                predictions, attention_weights, _, _ = model(data)

            predictions = predictions.squeeze(1)

            label = label.cuda()
            label = label.type(torch.cuda.FloatTensor)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    ct += 1
    return epoch_loss / ct, epoch_acc / ct


def get_batch_lengths(data, label, sentences, record_id=None):

    global unknown_size
    data_cpu = data.cpu()
    label_cpu = label.cpu()
    if record_id is not None:
        record_id_cpu = record_id.cpu()

    seq_len = [len(elem) for elem in sentences]
    seq_len_sorted_index = np.argsort(seq_len)[::-1]

    return_seq_len = [seq_len[sorted_index] for sorted_index in seq_len_sorted_index]
    return_data = torch.index_select(
        data.cpu(), 0, torch.tensor(list(seq_len_sorted_index))
    )
    return_label = torch.index_select(
        label_cpu, 0, torch.tensor(list(seq_len_sorted_index))
    )
    if record_id is not None:
        return_record_id = torch.index_select(
            record_id_cpu, 0, torch.tensor(list(seq_len_sorted_index))
        )

    if record_id is not None:
        return (
            return_data.to(device),
            return_label.to(device),
            return_seq_len,
            return_record_id,
        )
    else:
        return (return_data.to(device), return_label.to(device), return_seq_len)


def get_unknown_length(iterator):
    unknown_size = 0
    for batch in iterator:
        data = batch.text
        unknown_size += np.count_nonzero(data.cpu() == 0)

    print("Number of unknown tokens ", unknown_size)


class ModelClass(VisClass):

    model_metrics = {}
    model_metrics["total"] = 0
    model_metrics["total_incorrect"] = 0
    model_metrics["class_0_correct"] = 0
    model_metrics["class_0_classified_as_1"] = 0
    model_metrics["class_1_correct"] = 0
    model_metrics["class_1_classified_as_0"] = 0
    model_metrics["validation_accuracy"] = []
    model_metrics["training_accuracy"] = []

    roc = {}

    threshold_v = np.vectorize(custom_threshold)

    def __init__(self, SPLIT_DATA):

        super().__init__()
        self.init_roc_dict()
        self.SPLIT_DATA = SPLIT_DATA

    def init_roc_dict(self):

        xrange = np.arange(0, 1, 0.1)
        b = np.array([0.95, 0.99, 0.999, 0.9999, 1.0])
        self.xrange = np.concatenate((xrange, b), axis=None)
        for i in self.xrange:
            self.roc[i] = {}
            self.roc[i]["class_0_classified_as_1"] = 0
            self.roc[i]["class_1_classified_as_0"] = 0
            self.roc[i]["class_0_correct"] = 0
            self.roc[i]["class_1_correct"] = 0
            self.roc[i]["total_incorrect"] = 0

    def get_data(self):

        # ------------------- Split data and get iterators--------------- #
        if self.SPLIT_DATA == True:
            split_data(DATA_FILE, 0.75)

        self.data_iterator = IteratorClass()
        self.train_iterator, self.valid_iterator, self.test_iterator, self.vocab = (
            self.data_iterator.train_iterator,
            self.data_iterator.valid_iterator,
            self.data_iterator.test_iterator,
            self.data_iterator.TEXT.vocab,
        )

    def get_iterators(self):

        self.data_iterator.get_iterators()
        self.train_iterator, self.valid_iterator, self.test_iterator = (
            self.data_iterator.train_iterator,
            self.data_iterator.valid_iterator,
            self.data_iterator.test_iterator,
        )

    def train_model(self):

        # --------------------- Neural Network parameters --------------- #

        INPUT_DIM = len(self.vocab)
        EMBEDDING_DIM = 1024
        HIDDEN_DIM = 128
        OUTPUT_DIM = 1
        N_LAYERS = 1
        BIDIRECTIONAL = True
        DROPOUT = 0.01
        N_EPOCHS = 50
        USE_PRETRAINED_EMBEDDING = False

        self.model = RNN(
            INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
        )

        if USE_PRETRAINED_EMBEDDING:
            pretrained_embeddings = self.vocab.vectors
            print("Shape of vocabulary", pretrained_embeddings.shape)
            self.model.embedding.weight.data.copy_(pretrained_embeddings)

        self.optimizer = optim.Adam(self.model.parameters())
        # class_weights = torch.FloatTensor([3])
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.model = nn.DataParallel(self.model, device_ids = [0,1,2])
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

        for epoch in range(N_EPOCHS):

            self.get_iterators()
            t0 = time.time()
            train_loss, train_acc = train(
                self.model, self.train_iterator, self.optimizer, self.criterion
            )
            if DEBUG == False:
                valid_loss, valid_acc = evaluate(
                    self.model, self.valid_iterator, self.criterion
                )
            else:
                valid_loss = 0
                valid_acc = 0
            print("Time spent ", time.time() - t0)
            self.model_metrics["training_accuracy"].append(train_acc)
            self.model_metrics["validation_accuracy"].append(valid_acc)
            print(
                f"| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |"
            )

    def evaluate_model(self):

        print("Evaluating model performance")

        if DEBUG == False:
            test_loss, test_acc = evaluate(
                self.model, self.test_iterator, self.criterion
            )
            print(" Test Loss: and Acc: -- ", test_loss, test_acc)

        get_unknown_length(self.test_iterator)

    def predict(self):

        self.model.eval()
        self.get_iterators()
        self.reset_model_metrics()

        predictions_array = []
        penultimate_layer_array = []

        with torch.no_grad():

            for ct, batch in enumerate(self.test_iterator):

                data = batch.text
                label = batch.label
                sentences = batch.sentences
                # tweet_id = batch.user_id
                # record_id = batch.id

                data = data.to(device)
                label = label.to(device)

                print("Length of data in predict ", len(data))

                if PACKED_FLAG:
                    data, label, batch_lengths, record_id = get_batch_lengths(
                        data, label, record_id
                    )
                    (
                        predictions,
                        attention_weights,
                        penultimate_layer,
                        score,
                    ) = self.model(data, batch_lengths)
                    # print(list(zip(record_id.tolist(), label.cpu().tolist(), predictions.squeeze().tolist(), score.squeeze().tolist())))
                    predictions_array.extend(
                        list(
                            zip(
                                record_id.tolist(),
                                label.cpu().tolist(),
                                score.squeeze().tolist(),
                                penultimate_layer.tolist(),
                                attention_weights.cpu().detach().numpy(),
                            )
                        )
                    )

                else:
                    (
                        predictions,
                        attention_weights,
                        penultimate_layer,
                        score,
                    ) = self.model(data)
                    self.compute_accuracy_and_f1_score(
                        zip(score.round(), label.type(torch.cuda.FloatTensor))
                    )
                    predictions_array.extend(
                        list(
                            zip(
                                sentences,
                                label.cpu().tolist(),
                                score.squeeze().tolist(),
                                penultimate_layer.tolist(),
                                attention_weights.cpu().detach().numpy(),
                            )
                        )
                    )

                    self.compute_roc(score, label.type(torch.cuda.FloatTensor))

                print("Batch ", ct)

        self.write_model_metrics()
        # self.create_results_df(predictions_array)
        self.df = pd.DataFrame(
            predictions_array,
            columns=["text", "label", "score", "penultimate", "attention_weights"],
        )
        self.vis(
            np.array(list(self.df["penultimate"])),
            "proj",
            "Result of penultimate layer projections",
        )

    def create_results_df(self, predictions_array):
        self.test_df = pd.read_csv(DATA_FOLDER + "/Test.csv", header=None)
        self.test_df.columns = ["record_id", "text", "user_id", "label"]
        # self.df.set_index('record_id', inplace=True)
        self.df = pd.DataFrame(
            predictions_array,
            columns=["record_id", "label", "score", "penultimate", "attention_weights"],
        )
        self.df = self.df.merge(
            self.test_df[["record_id", "text", "user_id"]], how="left"
        )
        # self.df['label'] = self.df['label'].apply( lambda row: abs( 1 - row) )
        self.df["label"] = abs(1 - self.df["label"])
        self.df["score"] = 1 - self.df["score"]
        self.df["confidence"] = self.df.apply(
            lambda row: row["score"]
            if (round(row["score"]) == 1)
            else 1 - row["score"],
            axis=1,
        )
        self.df["score_tag"] = self.df.apply(
            lambda row: 0 if round(row["score"]) == row["label"] else 5, axis=1
        )
        print(self.df.head(50))
        self.df["normalized_attention_weights"] = self.df.apply(
            lambda row: self.get_normalized_weights(
                list(zip(row["text"].split(), row["attention_weights"]))
            ),
            axis=1,
        )

        incorrect = 0
        total = len(self.df)
        for row_id, row in self.df.iterrows():
            if row["label"] != round(row["score"]):
                incorrect += 1

        accuracy = (total - incorrect) / total
        print("Manual accuracy calculation", accuracy)

    @staticmethod
    def get_normalized_weights(weight_list):
        filtered_list = list(filter(lambda x: x[0] != "<pad>", weight_list))
        total_weight = sum([elem[1] for elem in filtered_list])
        return_list = list(map(lambda x: (x[0], x[1] / total_weight), filtered_list))
        # print(return_list)
        return return_list

    def save_model(self):
        torch.save(self.model, "saved_model.pt")

    def load_model(self):
        self.model = torch.load("saved_model.pt")
        self.model.eval()
        # class_weights = torch.FloatTensor([3])
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = self.criterion.to(device)

    def compute_roc(self, score, label):

        for threshold in self.xrange:

            thresholded_score = ModelClass.threshold_v(score.cpu(), threshold)
            thresholded_score = thresholded_score.squeeze(1)

            for i in zip(thresholded_score, label):

                if i[0] != i[1]:

                    if i[1] == 0.0:
                        self.roc[threshold]["class_0_classified_as_1"] += 1
                    if i[1] == 1.0:
                        self.roc[threshold]["class_1_classified_as_0"] += 1

                    self.roc[threshold]["total_incorrect"] += 1
                else:
                    if i[1] == 0.0:
                        self.roc[threshold]["class_0_correct"] += 1
                    elif i[1] == 1.0:
                        self.roc[threshold]["class_1_correct"] += 1

    def compute_accuracy_and_f1_score(self, zipped_list):
        # zipped_list[0] is score and zipped_list[1] is label

        ct = 0
        for i in zipped_list:

            ct += 1
            if i[0] != i[1]:

                if i[1] == 0.0:
                    self.model_metrics["class_0_classified_as_1"] += 1
                if i[1] == 1.0:
                    self.model_metrics["class_1_classified_as_0"] += 1

                self.model_metrics["total_incorrect"] += 1
            else:
                if i[1] == 0.0:
                    self.model_metrics["class_0_correct"] += 1
                elif i[1] == 1.0:
                    self.model_metrics["class_1_correct"] += 1

        self.model_metrics["total"] += ct
        print(
            "Prediction metrics ",
            self.model_metrics["total"],
            self.model_metrics["total_incorrect"],
            self.model_metrics["class_0_correct"],
            self.model_metrics["class_1_correct"],
        )

    def reset_model_metrics(self):
        self.model_metrics["total"] = 0
        self.model_metrics["total_incorrect"] = 0
        self.model_metrics["class_0_correct"] = 0
        self.model_metrics["class_0_classified_as_1"] = 0
        self.model_metrics["class_1_correct"] = 0
        self.model_metrics["class_1_classified_as_0"] = 0

    def write_model_metrics(self):
        self.model_metrics["TP"] = self.model_metrics["class_0_correct"]
        self.model_metrics["TN"] = self.model_metrics["class_1_correct"]
        self.model_metrics["FP"] = self.model_metrics["class_0_classified_as_1"]
        self.model_metrics["FN"] = self.model_metrics["class_1_classified_as_0"]
        self.model_metrics["Accuracy"] = (
            (self.model_metrics["total"] - self.model_metrics["total_incorrect"])
            * 100.0
            / self.model_metrics["total"]
        )
        self.model_metrics["Precision"] = (
            self.model_metrics["TP"]
            * 100.0
            / (self.model_metrics["TP"] + self.model_metrics["FP"])
        )
        self.model_metrics["Recall"] = (
            self.model_metrics["TP"]
            * 100.0
            / (self.model_metrics["TP"] + self.model_metrics["FN"])
        )
        self.model_metrics["F1-score"] = (
            2.0
            * self.model_metrics["Precision"]
            * self.model_metrics["Recall"]
            / (self.model_metrics["Precision"] + self.model_metrics["Recall"])
        )

        f = open("model_metrics_large.json", "a")
        json.dump(self.model_metrics, f)
        f.close()

        for elem in self.roc:
            print(elem)
            metrics = self.roc[elem]
            tp = metrics["class_0_correct"]
            tn = metrics["class_1_correct"]
            fp = metrics["class_1_classified_as_0"]
            fn = metrics["class_0_classified_as_1"]

            print(tp, tn, fp, fn)

            self.roc[elem]["TPR"] = tp / (tp + fn)
            self.roc[elem]["FPR"] = fp / (fp + tn)

        f = open("roc_metrics.json", "a")
        json.dump(self.roc, f)
        f.close()


# ----------------------------- Parameters -----------------------#

DEBUG = False
PACKED_FLAG = False
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda'

DATA_FOLDER = "./curated_small"
DATA_FILE = "./curated_small/Curated_userid.csv"

DATA_FOLDER = "./curated"
DATA_FILE = "./curated/Training_data.csv"


def train_new_model(GET_NEW_DATA=False):
    attention_model = ModelClass(GET_NEW_DATA)
    attention_model.get_data()
    attention_model.train_model()
    print("Calling evaluate")
    attention_model.evaluate_model()
    attention_model.predict()
    attention_model.save_model()


def use_saved_model(GET_NEW_DATA=False):
    attention_model = ModelClass(GET_NEW_DATA)
    attention_model.get_data()
    attention_model.load_model()
    # attention_model.evaluate_model()
    attention_model.predict()


train_new_model()
