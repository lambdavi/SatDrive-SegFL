import numpy as np


class Metrics:
    """
    Base class for calculating evaluation metrics.

    Args:
        n_classes (int): Number of classes.
        name (str): Name of the metrics.

    Attributes:
        n_classes (int): Number of classes.
        name (str): Name of the metrics.
        confusion_matrix (ndarray): Confusion matrix.
        total_samples (int): Total number of samples.
        results (dict): Evaluation results.

    Methods:
        reset(self): Resets the confusion matrix and total samples count.
    """
    def __init__(self, n_classes, name):
        super().__init__()
        self.n_classes = n_classes
        self.name = name
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.results = {}

    def reset(self):
        """
        Resets the confusion matrix and total samples count.
        """
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0


class StreamClsMetrics(Metrics):
    """
    Class for evaluating classification metrics in a streaming manner.

    Args:
        n_classes (int): Number of classes.
        name (str): Name of the metrics.

    Inherits:
        Metrics: Base class for calculating evaluation metrics.

    Methods:
        update(self, label, prediction): Updates the metrics with a new label and prediction.
        get_results(self): Calculates and returns the evaluation results.
        __str__(self): Returns a string representation of the evaluation results.
    """
    def __init__(self, n_classes, name):
        super().__init__(n_classes=n_classes, name=name)

    def update(self, label, prediction):
        """
        Updates the metrics with a new label and prediction.

        Args:
            label (int): True label.
            prediction (int): Predicted label.
        """
        self.confusion_matrix[label[0]][prediction[0]] += 1
        self.total_samples += 1

    def get_results(self):
        """
        Calculates and returns the evaluation results.

        Returns:
            dict: Evaluation results including total samples, overall accuracy, mean accuracy, and class accuracy.
        """
        eps = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + eps)
        acc_cls = np.mean(acc_cls_c[mask])
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        self.results = {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Class Acc": cls_acc
        }

    def __str__(self):
        """
        Returns a string representation of the evaluation results.

        Returns:
            str: String representation of the evaluation results.
        """
        string = "\n"
        ignore = ["Class Acc", "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in self.results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)
        string += 'Class Acc:\n'
        for k, v in self.results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))
        return string


class StreamSegMetrics(Metrics):
    """
    Class for evaluating segmentation metrics in a streaming manner.

    Args:
        n_classes (int): Number of classes.
        name (str): Name of the metrics.

    Inherits:
        Metrics: Base class for calculating evaluation metrics.

    Methods:
        update(self, label_trues, label_preds): Updates the metrics with new labels and predictions.
        get_results(self): Calculates and returns the evaluation results.
        confusion_matrix_to_text(self): Converts the confusion matrix to a text representation.
        __str__(self): Returns a string representation of the evaluation results.
    """
    def __init__(self, n_classes, name):
        super().__init__(n_classes=n_classes, name=name)

    def _fast_hist(self, label_true, label_pred):
        """
        Computes the histogram of true and predicted labels.

        Args:
            label_true (ndarray): True labels.
            label_pred (ndarray): Predicted labels.

        Returns:
            ndarray: Histogram of true and predicted labels.
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        """
        Updates the metrics with new labels and predictions.

        Args:
            label_trues (list or ndarray): List of true labels.
            label_preds (list or ndarray): List of predicted labels.
        """
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def get_results(self):
        """
        Calculates and returns the evaluation results.

        Returns:
            dict: Evaluation results including total samples, overall accuracy, mean accuracy, mean precision,
                  frequency-weighted accuracy, mean IoU, class IoU, class accuracy, and class precision.
        """

        eps = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + eps)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + eps)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + eps)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        self.results = {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Mean Precision": precision_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Class Prec": cls_prec,
        }

        return self.results

    def confusion_matrix_to_text(self):
        """
        Converts the confusion matrix to a text representation.

        Returns:
            str: Text representation of the confusion matrix.
        """
        string = []
        for i in range(self.n_classes):
            string.append(f"{i} : {self.confusion_matrix[i].tolist()}")
        return "\n" + "\n".join(string)

    def __str__(self):
        """
        Returns a string representation of the evaluation results.

        Returns:
            str: String representation of the evaluation results.
        """
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in self.results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in self.results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in self.results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Prec:\n'
        for k, v in self.results['Class Prec'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string
