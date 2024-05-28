import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_loader = DataLoader(ReviewDataSet(x, y), batch_size=self.batch_size, shuffle=False)
        best_f1 = None
        for i in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for values, labels in tqdm(train_loader):
                values, labels = values.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(values)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            loss, _, _, F1SM = self._eval_epoch(self.test_loader, self.model)
            print(f'Epoch {i + 1}/{self.num_epochs}, Loss: {running_loss / len(train_loader)}, Eval Loss: {loss}, F1 Score: {F1SM}')

            if best_f1 is None:
                best_f1 = F1SM
                self.best_model = self.model.state_dict()
            elif F1SM > best_f1:
                best_f1 = F1SM
                self.best_model = self.model.state_dict()

        self.model.load_state_dict(self.best_model)
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        predicted = []

        with torch.no_grad():
            for values, labels in DataLoader(ReviewDataSet(x, np.zeros(len(x))), batch_size=self.batch_size, shuffle=False):
                values = values.to(self.device)
                outputs = self.model(values)
                _, p = torch.max(outputs, 1)
                predicted.extend(p.cpu().numpy())

        return predicted

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """

        model.eval()
        loss = 0.0
        predicted, true = [], []
        with torch.no_grad():
            for values, labels in dataloader:
                values, labels = values.to(self.device), labels.to(self.device)
                outputs = model(values)
                loss = self.criterion(outputs, labels)
                loss += loss.item() / len(dataloader)
                _, p = torch.max(outputs, 1)
                predicted.extend(p.cpu().numpy())
                true.extend(labels.cpu().numpy())

        return loss, predicted, true, f1_score(true, predicted, average='macro')

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        self.test_loader = DataLoader(ReviewDataSet(X_test, y_test), batch_size=self.batch_size)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        return classification_report(y, self.predict(x))


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader()
    loader.load_data()
    loader.get_embeddings()
    x_train, x_test, y_train, y_test = loader.split_data()

    classifier = DeepModelClassifier(in_features=len(x_train[0]), num_classes=2, batch_size=32, num_epochs=10)
    classifier.set_test_dataloader(x_test, y_test)
    classifier.fit(x_train, y_train)
    print(classifier.prediction_report(x_test, y_test))
