class LabelTrafoToBinary:
    def __call__(self, labels):
        labels = (labels > 0).astype(labels.dtype)
        return labels
