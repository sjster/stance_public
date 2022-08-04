import plotting as pl
import numpy as np


def get_average(metric_dict, name, keyname):

    elem_mean = np.mean(metric_dict[keyname])
    print("Average for ", name, " with ", keyname, " is ", elem_mean)
    return elem_mean


def get_boxplots_single(filename, output_filename, title):

    # --------------- Glove -------------- #
    glove = pl.get_values(["Accuracy", "Precision", "Recall", "F1-score"], filename)
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(data, filename=output_filename, title=title)


def get_boxplots():

    # --------------- Glove -------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(data, filename="Glove.pdf", title="Glove")

    # ---------------- Glove twitter ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(data, filename="Glove_twitter.pdf", title="Glove.twitter")

    # ---------------- Charngrams ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_charngrams.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(data, filename="Charngrams.pdf", title="Charngrams")

    # ---------------- Elmo ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(
        data,
        filename="Elmo_small_training_data_5_epochs.pdf",
        title="Elmo with smaller training data",
    )

    # ---------------- Elmo ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_25.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(
        data,
        filename="Elmo_small_training_data_25_epochs.pdf",
        title="Elmo with smaller training data",
    )

    # ---------------- Elmo ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_large.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(
        data,
        filename="Elmo_large_training_data_5_epochs.pdf",
        title="Elmo with larger training data",
    )

    # ---------------- Elmo ------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_large_25.json",
    )
    print(glove)

    data = [
        {"y": glove["Accuracy"], "name": "Accuracy"},
        {"y": glove["Precision"], "name": "Precision"},
        {"y": glove["Recall"], "name": "Recall"},
        {"y": glove["F1-score"], "name": "F1-score"},
    ]

    pl.get_boxplot(
        data,
        filename="Elmo_large_training_data_25_epochs.pdf",
        title="Elmo with larger training data",
    )


def get_boxplots_grouped_by_type():

    # --------------- Glove -------------- #
    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove.json",
    )

    # ---------------- Glove twitter ------------- #
    glove_twitter = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter.json",
    )

    # ---------------- Charngrams ------------- #
    charngrams = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_charngrams.json",
    )

    # --------------- Glove -------------- #
    glove_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_large.json",
    )

    # ---------------- Glove twitter ------------- #
    glove_twitter_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter_large.json",
    )

    # ---------------- Charngrams ------------- #
    charngrams_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_charngrams_large.json",
    )

    # ---------------- Elmo ------------- #
    elmo_small = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics.json",
    )

    # ---------------- Elmo ------------- #
    elmo_small_25 = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_25.json",
    )

    # ---------------- Elmo ------------- #
    elmo_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_large.json",
    )

    # ---------------- Elmo ------------- #
    elmo_large_25 = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_large_25.json",
    )

    data = [
        {"y": glove["Accuracy"], "name": "Glove"},
        {"y": glove_twitter["Accuracy"], "name": "Glove.twitter"},
        {"y": charngrams["Accuracy"], "name": "Charngrams"},
        {"y": glove_large["Accuracy"], "name": "Glove, large data"},
        {"y": glove_twitter_large["Accuracy"], "name": "Glove.twitter large data"},
        {"y": charngrams_large["Accuracy"], "name": "Charngrams, large data"},
        {"y": elmo_small_25["Accuracy"], "name": "Elmo, small data"},
        {"y": elmo_large_25["Accuracy"], "name": "Elmo, large data"},
    ]

    pl.get_boxplot(
        data,
        filename="Boxplot_accuracy.pdf",
        title="Comparison of accuracy for all the models",
    )

    data = [
        {"y": glove["F1-score"], "name": "Glove"},
        {"y": glove_twitter["F1-score"], "name": "Glove.twitter"},
        {"y": charngrams["F1-score"], "name": "Charngrams"},
        {"y": glove_large["F1-score"], "name": "Glove, large data"},
        {"y": glove_twitter_large["F1-score"], "name": "Glove.twitter large data"},
        {"y": charngrams_large["F1-score"], "name": "Charngrams, large data"},
        {"y": elmo_small_25["F1-score"], "name": "Elmo, small data"},
        {"y": elmo_large_25["F1-score"], "name": "Elmo, large data"},
    ]

    pl.get_boxplot(
        data,
        filename="Boxplot_F1_score.pdf",
        title="Comparison of F1-score for all the models",
    )

    data = [
        {"y": glove["Precision"], "name": "Glove"},
        {"y": glove_twitter["Precision"], "name": "Glove.twitter"},
        {"y": charngrams["Precision"], "name": "Charngrams"},
        {"y": glove_large["Precision"], "name": "Glove, large data"},
        {"y": glove_twitter_large["Precision"], "name": "Glove.twitter large data"},
        {"y": charngrams_large["Precision"], "name": "Charngrams, large data"},
        {"y": elmo_small_25["Precision"], "name": "Elmo, small data"},
        {"y": elmo_large_25["Precision"], "name": "Elmo, large data"},
    ]

    pl.get_boxplot(
        data,
        filename="Boxplot_Precision.pdf",
        title="Comparison of Precision for all the models",
    )

    data = [
        {"y": glove["Recall"], "name": "Glove"},
        {"y": glove_twitter["Recall"], "name": "Glove.twitter"},
        {"y": charngrams["Recall"], "name": "Charngrams"},
        {"y": glove_large["Recall"], "name": "Glove, large data"},
        {"y": glove_twitter_large["Recall"], "name": "Glove.twitter large data"},
        {"y": charngrams_large["Recall"], "name": "Charngrams, large data"},
        {"y": elmo_small_25["Recall"], "name": "Elmo, small data"},
        {"y": elmo_large_25["Recall"], "name": "Elmo, large data"},
    ]

    pl.get_boxplot(
        data,
        filename="Boxplot_Recall.pdf",
        title="Comparison of Recall for all the models",
    )


def get_history_single(filename, output_filename, title):

    # --------------------- Glove ------------------ #
    glove = pl.get_values(["training_accuracy", "validation_accuracy"], filename)
    print(glove)

    training = [
        {"y": elem, "name": "Training run", "color": "salmon"}
        for elem in glove["training_accuracy"]
    ]
    validation = [
        {"y": elem, "name": "Validation run", "color": "steelblue"}
        for elem in glove["validation_accuracy"]
    ]
    training.extend(validation)

    pl.get_lineplot(
        training,
        filename=output_filename,
        title=title,
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )


def get_history():

    # --------------------- Glove ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_glove.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Glove_accuracy.pdf",
        title="Training and validation accuracy for Glove",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Glove twitter ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Glove_twitter_accuracy.pdf",
        title="Training and validation accuracy for Glove.twitter",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Charngrams ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_charngrams.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Charngrams_accuracy.pdf",
        title="Training and validation accuracy for Charngrams",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Glove ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_glove_large.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Glove_large_accuracy.pdf",
        title="Training and validation accuracy for Glove with larger dataset",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Glove ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter_large.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Glove_twitter_large_accuracy.pdf",
        title="Training and validation accuracy for Glove.twitter with larger dataset",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Glove ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_attention_rnn/model_metrics_charngrams_large.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Charngrams_large_accuracy.pdf",
        title="Training and validation accuracy for Charngrams with larger dataset",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Elmo ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_elmo/model_metrics_25.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Elmo_small_accuracy.pdf",
        title="Training and validation accuracy for Elmo on the smaller dataset",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )

    # --------------------- Elmo large ------------------ #
    glove = pl.get_values(
        ["training_accuracy", "validation_accuracy"],
        "./Pytorch_elmo/model_metrics_large_25.json",
    )
    print(glove)

    data = [
        {
            "y": glove["training_accuracy"][0],
            "name": "Training run 1",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][1],
            "name": "Training run 2",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][2],
            "name": "Training run 3",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][3],
            "name": "Training run 4",
            "color": "salmon",
        },
        {
            "y": glove["training_accuracy"][4],
            "name": "Training run 5",
            "color": "salmon",
        },
        {
            "y": glove["validation_accuracy"][0],
            "name": "Validation run 1",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][1],
            "name": "Validation run 2",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][2],
            "name": "Validation run 3",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][3],
            "name": "Validation run 4",
            "color": "steelblue",
        },
        {
            "y": glove["validation_accuracy"][4],
            "name": "Validation run 5",
            "color": "steelblue",
        },
    ]

    pl.get_lineplot(
        data,
        filename="Elmo_large_accuracy.pdf",
        title="Training and validation accuracy for Elmo on the larger dataset",
        xaxis_title="Number of epochs",
        yaxis_title="Accuracy",
    )


def get_average_metrics():

    glove = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove.json",
    )
    glove_average = get_average(glove, "glove", "Accuracy")
    glove_average = get_average(glove, "glove", "F1-score")
    glove_twitter = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter.json",
    )
    glove_twitter_average = get_average(glove_twitter, "glove_twitter", "Accuracy")
    glove_twitter_average = get_average(glove_twitter, "glove_twitter", "F1-score")
    charngrams = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_charngrams.json",
    )
    charngrams_average = get_average(charngrams, "charngrams", "Accuracy")
    charngrams_average = get_average(charngrams, "charngrams", "F1-score")

    glove_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_large.json",
    )
    glove_large_average = get_average(glove_large, "glove_large", "Accuracy")
    glove_large_average = get_average(glove_large, "glove_large", "F1-score")
    glove_twitter_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_glove_twitter_large.json",
    )
    glove_twitter_large_average = get_average(
        glove_twitter_large, "glove_twitter_large", "Accuracy"
    )
    glove_twitter_large_average = get_average(
        glove_twitter_large, "glove_twitter_large", "F1-score"
    )
    charngrams_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_attention_rnn/model_metrics_charngrams_large.json",
    )
    charngrams_large_average = get_average(
        charngrams_large, "charngrams_large", "Accuracy"
    )
    charngrams_large_average = get_average(
        charngrams_large, "charngrams_large", "F1-score"
    )

    elmo_small = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_25.json",
    )
    elmo_small_average = get_average(elmo_small, "elmo small", "Accuracy")
    elmo_small_average = get_average(elmo_small, "elmo small", "F1-score")
    elmo_large = pl.get_values(
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "./Pytorch_elmo/model_metrics_large_25.json",
    )
    elmo_large_average = get_average(elmo_large, "elmo large", "Accuracy")
    elmo_large_average = get_average(elmo_large, "elmo large", "F1-score")


def get_roc_metrics():

    roc = pl.read_roc_json("./Pytorch_attention_rnn/roc_metrics.json")
    # 0 is cutoff point, 1 is TPR, 2 is FPR
    y = [elem[1] for elem in roc]
    x = [elem[2] for elem in roc]

    roc = pl.read_roc_json("./Pytorch_elmo/roc_metrics.json")
    # 0 is cutoff point, 1 is TPR, 2 is FPR
    y2 = [elem[1] for elem in roc]
    x2 = [elem[2] for elem in roc]
    data = [
        {"y": y, "x": x, "name": "Charngrams", "color": "salmon"},
        {"y": y2, "x": x2, "name": "Elmo", "color": "steelblue"},
    ]
    pl.get_lineplot(data, filename="Combined_ROC.png", title="ROC curves")


# get_history()
# get_average_metrics()
# get_boxplots_grouped_by_type()
# get_roc_metrics()
# get_history_single('/home/srijith/.tej/jobs/attn_pytorch/stage/model_metrics_charngrams.json','test_history.pdf','Training and validation accuracy for Charngrams')
