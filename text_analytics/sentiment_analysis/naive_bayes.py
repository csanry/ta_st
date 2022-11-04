import argparse
import logging
import pickle
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from text_analytics.config import BASE_SCORER, CV_SPLIT, DATA_PATH, MODEL_PATH
from text_analytics.helpers import (
    calculate_report_metrics,
    evaluate_tuning,
    save_confusion_matrix,
    save_roc_curve,
)

warnings.filterwarnings("ignore")


class NaiveBayesReviews:
    def __init__(
        self,
        X_train: pd.DataFrame = None,
        y_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        scorer: dict = BASE_SCORER,
        cv_split: StratifiedShuffleSplit = CV_SPLIT,
        vectoriser: Optional = None,
        model: MultinomialNB = None,
        hpt: Optional = None,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scorer = scorer
        self.cv_split = cv_split

        if vectoriser is None:
            vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=0.005)

        self.vec = vectoriser

        if model is None:
            model = MultinomialNB()

        self.model = model
        self.trainer = None
        self.best_model = None

        self.timestamp = datetime.now().strftime("%d_%H_%M")
        self.file_name = f"sent_nb_{self.timestamp}"

        if hpt is None:
            hpt = {"nb__alpha": np.arange(0.1, 2, 0.1)}
        self.hpt = hpt

    def save_model(self) -> None:
        save_path = MODEL_PATH / f"{self.file_name}.pkl"

        with open(save_path, "wb") as file:
            pickle.dump(self.best_model, file)

        print(f"Model saved to {save_path}")

    def load_model(self, file_name: str) -> None:

        model_path = MODEL_PATH / f"{file_name}.pkl"
        with open(model_path, "rb") as file:
            self.best_model = pickle.load(file)

        print(f"Model loaded from {model_path}")

    def train(self):

        # build the pipeline
        nb_pipe = Pipeline([("vec", self.vec), ("nb", self.model)])

        # cross validate model with RandomizedSearch
        self.trainer = GridSearchCV(
            estimator=nb_pipe,
            param_grid=self.hpt,
            scoring=self.scorer,
            refit="F_score",
            cv=self.cv_split,
            return_train_score=True,
            verbose=10,
        )

        self.trainer.fit(self.X_train, self.y_train)

        self.best_model = self.trainer.best_estimator_

        return self.trainer, self.best_model

    def evaluate(self) -> None:
        try:
            evaluate_tuning(tuner=self.trainer)
        except AttributeError as err:
            print(
                f"{type(err)}: Fitting must be done before evaluating of hyperparameter tuning process"
            )
            raise

        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        report = calculate_report_metrics(
            y_test=self.y_test, y_pred=y_pred, y_pred_prob=y_pred_proba
        )

        save_confusion_matrix(cf_matrix=report["cf_matrix"], model_name=self.file_name)

        save_roc_curve(
            fpr=report.get("roc")[0],
            tpr=report.get("roc")[1],
            model_name=self.file_name,
            auc=report.get("auroc"),
        )

    def predict_csv(
        self, csv_to_predict: pd.DataFrame, reviews_column: str = "preprocessed_review"
    ) -> pd.DataFrame:

        if reviews_column not in csv_to_predict.columns:
            raise ValueError("Column not in dataframe")

        X = csv_to_predict[reviews_column]

        try:
            csv_to_predict["predicted_class"] = self.best_model.predict(X)
        except BaseException as err:
            print(f"Unexpected err: {err}, Type: {type(err)}")
            raise

        return csv_to_predict

    def predict_single_review(self, article: List[str]) -> Tuple[str, List]:

        vectorised_features = self.best_model[0].transform(article)
        explainable_tokens = self.best_model[0].inverse_transform(vectorised_features)[
            0
        ]

        prediction = self.best_model.predict(article)

        logits = dict(
            zip(
                self.best_model[0].get_feature_names_out(),
                self.best_model[1].feature_log_prob_[1 if prediction > 0.5 else 0],
            )
        )

        token_scores = sorted(
            [f"{token}, {logits.get(token):.05f}" for token in explainable_tokens],
            key=lambda score: abs(float(score.split(", ")[1])),
            reverse=True,
        )

        return "Positive" if prediction > 0.5 else "Negative", token_scores


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", default=15, required=False)
    parser.add_argument(
        "--vec", choices=["count", "tfidf"], default="tfidf", required=False
    )
    args = parser.parse_args()

    logger.info("Reading in files")
    train = pd.read_parquet(DATA_PATH / "sentiment_train.parquet")
    test = pd.read_parquet(DATA_PATH / "sentiment_test.parquet")

    X_train, y_train = train["preprocessed_review"], train["class"]
    X_test, y_test = test["preprocessed_review"], test["class"]

    logger.info(
        f"Instantiating vectoriser: {'TfidfVectorizer' if str(args.vec) == 'tfidf' else 'CountVectoriser'}"
    )
    vectoriser = (
        TfidfVectorizer(ngram_range=(1, 2), min_df=0.005)
        if str(args.vec) == "tfidf"
        else CountVectorizer(ngram_range=(1, 2), min_df=0.005)
    )

    nb = NaiveBayesReviews(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        vectoriser=vectoriser,
    )

    logger.info("Training model")
    nb.train()

    logger.info("Evaluating model")
    nb.evaluate()

    logger.info("Saving model")
    nb.save_model()
