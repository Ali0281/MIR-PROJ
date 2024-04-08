from typing import List

import numpy as np
from spacy_loggers import wandb


class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        # TODO
        precision_total = 0.0
        count = min(len(actual), len(predicted))

        for i in range(count):
            ret = set(predicted[i])
            rel = set(actual[i])
            if len(ret) != 0: precision_total += len(ret.intersection(rel)) / len(ret)

        return precision_total / count if count > 0 else 0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        # TODO :
        recall_total = 0.0
        count = min(len(actual), len(predicted))

        for i in range(count):
            ret = set(predicted[i])
            rel = set(actual[i])
            if len(rel) != 0:
                recall_total += len(ret.intersection(rel)) / len(rel)
            else:
                recall_total += 1

        return recall_total / count if count > 0 else 0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        # TODO
        f1_total = 0.0
        count = min(len(actual), len(predicted))

        for i in range(count):
            ret = set(predicted[i])
            rel = set(actual[i])
            P = len(ret.intersection(rel)) / len(ret) if len(ret) != 0 else 0
            R = len(ret.intersection(rel)) / len(rel) if len(rel) != 0 else 1
            f1_total += 2(P * R) / (P + R) if P + R != 0 else 0

        return f1_total / count if count > 0 else 0

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        # TODO
        AP = []
        count = min(len(actual), len(predicted))
        for i in range(count):
            all_till_now, correct_till_now, cumulative, flag = 0, 0, 0, False
            for item in predicted[i]:
                all_till_now += 1
                if item in set(actual[i]):
                    flag = True
                    correct_till_now += 1
                    cumulative += correct_till_now / all_till_now
            if flag:
                AP.append(cumulative / correct_till_now)

        # TODO : note : this and MAP cant be both floats while having the same input ! so i returned a list of floats in this part
        return np.array(AP)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        # TODO
        return self.calculate_AP(actual, predicted).mean()

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = [predicted.get(0)[1]]
        # TODO : note : فرض کردیم که پردیکشن و اکچوعال به جای اسرینگ یک اسریتگ و رلونس اسکور دارند که رلونس اسکور اکچوعال مهم است و ترتیب پردیکشن!
        # TODO: Calculate DCG here
        count = min(len(actual), len(predicted))

        for i in range(count):
            relevance_actual = [j[1] for j in actual[i]]
            items_actual = [j[0] for j in actual[i]]
            items_prediction = [j[0] for j in predicted[i]]

            iter_dcg = []
            for index, item in enumerate(items_prediction):
                if item not in items_actual: continue
                if index == 0:
                    iter_dcg.append(relevance_actual[items_actual.index(item)])
                else:
                    iter_dcg.append(iter_dcg[-1] + relevance_actual[items_actual.index(item)] / np.log2(index))
            DCG.append(iter_dcg)
        return DCG

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = []
        # TODO : note : same note as above
        # TODO: Calculate NDCG here

        count = min(len(actual), len(predicted))

        DCG = self.cacluate_DCG(actual, predicted)
        for i in range(count):
            relevance_actual = [j[1] for j in actual[i]]
            reverse_sorted_relevance_actual = sorted(relevance_actual, reverse=True)

            iter_best_dcg = []
            for index, value in enumerate(reverse_sorted_relevance_actual):
                if index == 0:
                    iter_best_dcg.append(value)
                else:
                    iter_best_dcg.append(iter_best_dcg[-1] + value / np.log2(index))
            # TODO : note : i shortened the iter best in case of "continue" happened
            NDCG.append(np.array(DCG[i]) - np.array(iter_best_dcg[:len(DCG[i])]))

        return NDCG

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RR = []
        # TODO: Calculate RR here
        # TOOD : note : i did the same as i did with AP
        count = min(len(actual), len(predicted))

        for i in range(count):
            all_till_now = 0
            for item in predicted[i]:
                all_till_now += 1
                if item in set(actual[i]):
                    RR.append(1 / all_till_now)
                    break

        return np.array(RR)

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        # TODO: Calculate MRR here
        return self.cacluate_RR(actual, predicted).mean()

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        # TODO: Print the evaluation metrics
        print("precision = ", precision)
        print("recall = ", recall)
        print("f1 = ", f1)
        print("ap = ", ap)
        print("map = ", map)
        print("dcg = ", dcg)
        print("ndcg = ", ndcg)
        print("rr = ", rr)
        print("mrr =", mrr)

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        # probably from this ?! - https://docs.wandb.ai/guides/track/log
        # TODO: Log the evaluation metrics using Wandb
        wandb.log({"precision": precision, "recall": recall, "f1": f1, "ap": ap, "map": map, "dcg": dcg, "ndcg": ndcg,
                   "rr": rr, "mrr": mrr})

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        # call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
