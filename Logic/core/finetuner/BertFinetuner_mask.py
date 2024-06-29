import json

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn import BCEWithLogitsLoss

import wandb
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres,
                                                                   problem_type="multi_label_classification")
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.multi_label_ = None
        self.multi_label_CM = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.DF = None
        self.X, self.Y = None, None
        self.X_, self.Y_ = [], []  # will add train val test sequentially

        self.top_n_genres = top_n_genres
        self.top_n_genres_values = None
        self.file_path = file_path

        self.label_encoder = LabelEncoder()

        # self.setup_wandb()

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/IMDB_movies.json", "r") as f:
            data = json.load(f)
        # or can do
        # with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/preprocess.json", "r") as f:
        #   data = json.load(f)

        # with open(self.file_path, "r") as f:
        #   data = json.load(f)

        self.DF = pd.DataFrame(data)[["id", "first_page_summary", "genres"]]

        # filter some values
        self.DF = self.DF[self.DF["genres"].apply(len) > 0]
        self.DF = self.DF[self.DF["genres"].notnull()]
        self.DF = self.DF[self.DF["id"].notnull()]
        self.DF = self.DF[self.DF["first_page_summary"].notnull()]

        # TODO : for testing
        self.DF = self.DF[:3000]

        print("load_dataset complete!")

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        # for plot
        explode = self.DF["genres"].explode()
        genre_values = explode.value_counts()

        values_counts = {}
        for index, i in self.DF.iterrows():
            if i is None: continue
            if i["genres"] is None: continue
            for g in i["genres"]:
                values_counts[g] = values_counts.get(g, 0) + 1

        values_counts = sorted(values_counts.items(), key=lambda x: x[1], reverse=True)
        top_n_items = values_counts[:self.top_n_genres]
        self.top_n_genres_values = [g for g, _ in top_n_items]
        # print(len(self.DF))

        #  for later
        self.multi_label_ = MultiLabelBinarizer(classes=self.top_n_genres_values)
        self.DF["top_n_counts"] = self.DF["genres"].apply(
            lambda x: len([i for i in x if i in self.top_n_genres_values]))

        # remove the unimportant
        self.DF = self.DF[self.DF["top_n_counts"] != 0]

        # filtering works but the effect is too low!
        # print(len(self.DF))

        # preprocess for input
        self.X = self.DF["first_page_summary"]
        self.Y = self.DF["genres"]

        # self.Y = [i[0] for i in self.DF["genres"]]
        # self.Y = self.label_encoder.fit_transform(self.Y)

        explode_ = self.DF["genres"].explode()
        genre_values_ = explode_.value_counts()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        genre_values.plot(kind='bar')
        plt.title("Non-filtered Genre Distribution")

        plt.subplot(1, 2, 2)
        genre_values_.plot(kind='bar')
        plt.title("Filtered Genre Distribution")

        plt.tight_layout()
        plt.show()

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        x_train_test, x_val, y_train_test, y_val = train_test_split(self.X, self.Y, test_size=val_size)
        x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test,
                                                            test_size=test_size / (1 - val_size))
        self.X_.append(x_train)
        self.X_.append(x_val)
        self.X_.append(x_test)
        self.Y_.append(y_train)
        self.Y_.append(y_val)
        self.Y_.append(y_test)

        print(f"split dateset completed!")
        print(f"train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def setup_wandb(self):
        wandb.login(key="21ac596995775183edf41f2e3e7c8f89b268f67f")
        wandb.init(project='bert', entity='entity',
                   config={'top_n_genres': self.top_n_genres})

    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
    """

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic
        T_labels = self.multi_label_.fit_transform(
            [list(set(i).intersection(set(self.top_n_genres_values))) for i in self.Y_[0]])
        V_labels = self.multi_label_.fit_transform(
            [list(set(i).intersection(set(self.top_n_genres_values))) for i in self.Y_[1]])

        T_summary, V_summary = list(self.X_[0]), list(self.X_[1])
        T_encoding = self.tokenizer(T_summary, truncation=True, padding=True)
        V_encoding = self.tokenizer(V_summary, truncation=True, padding=True)

        T_set, V_set = self.create_dataset(T_encoding, T_labels), self.create_dataset(V_encoding, V_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        # our main trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=T_set,
            eval_dataset=V_set,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        print("fine_tune_bert completed")

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        y_pred, y_true = (torch.sigmoid(torch.tensor(pred.predictions)) > 1 / 2).int().numpy(), pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic
        T_summary = list(self.X_[2])
        T_labels = self.multi_label_.transform(
            [list(set(i).intersection(set(self.top_n_genres_values))) for i in self.Y_[2]])
        T_encoding = self.tokenizer(T_summary, truncation=True, padding=True)
        T_set = self.create_dataset(T_encoding, T_labels)

        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )

        metrics = trainer.evaluate(T_set)
        print("evaluate_model completed!")
        print(metrics)

        y_pred = (torch.sigmoid(torch.tensor(trainer.predict(T_set).predictions)) > 1 / 2).int().numpy()
        y_true = T_labels
        self.multi_label_CM = multilabel_confusion_matrix(y_true, y_pred)

        n, m = 2, 2
        fig, axes = plt.subplots(n, m, figsize=(15, 10))
        axes = axes.flatten()
        for idx, i in enumerate(self.multi_label_CM):
            if idx >= n * m: break
            sns.heatmap(i, annot=True, fmt='d', ax=axes[idx], xticklabels=["F", "T"], yticklabels=["F", "T"])
            axes[idx].set_xlabel('pred')
            axes[idx].set_ylabel('actual')
            axes[idx].set_title(self.top_n_genres_values[idx])

        plt.tight_layout()
        plt.show()

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.tokenizer.save_pretrained(model_name)
        self.model.save_pretrained(model_name)

    def load_model(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, i):
        """
        Get a single item from the dataset.

        Args:
            i (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)


"""wandb: 
wandb: Run history:
wandb:           eval/accuracy ▁▇▆▇██
wandb:                 eval/f1 ▃▁▃▅██
wandb:               eval/loss █▆▄▂▁▁
wandb:          eval/precision ▁▂▇▇██
wandb:             eval/recall ▄▁▃▄▇█
wandb:            eval/runtime ▂██▂▃▁
wandb: eval/samples_per_second █▁▁█▄▇
wandb:   eval/steps_per_second ▄▁▁▄▃█
wandb:             train/epoch ▁▃▅▆██
wandb:       train/global_step ▂▄▅▇██▁
wandb: 
wandb: Run summary:
wandb:            eval/accuracy 0.31985
wandb:                  eval/f1 0.56744
wandb:                eval/loss 0.49925
wandb:           eval/precision 0.70674
wandb:              eval/recall 0.50852
wandb:             eval/runtime 55.0693
wandb:  eval/samples_per_second 14.818
wandb:    eval/steps_per_second 1.852
wandb:               total_flos 107631765151680.0
wandb:              train/epoch 5.0
wandb:        train/global_step 0
wandb:               train_loss 0.57805
wandb:            train_runtime 1936.9302
wandb: train_samples_per_second 1.404
wandb:   train_steps_per_second 0.088"""