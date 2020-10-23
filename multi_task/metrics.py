# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

# from multi_task.losses import l1_loss_instance
# from multi_task.losses import l1_loss_instance
import numpy as np
from sklearn.metrics import balanced_accuracy_score


class RunningMetric(object):
    def __init__(self, metric_type, n_classes=None):
        self._metric_type = metric_type
        self.num_updates = 0.0
        if metric_type == 'ACC':
            self.accuracy = 0.0
        if metric_type == 'ACC_BLNCD':
            self.accuracy_balanced = 0.0
            self.accuracy = 0.0
            self.num_batches = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
        if metric_type == 'IOU':
            assert n_classes is not None, 'ERROR: n_classes is needed for IOU'
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))

    def reset(self):
        self.num_updates = 0.0
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
        if self._metric_type == 'ACC_BLNCD':
            self.accuracy_balanced = 0.0
            self.accuracy = 0.0
            self.num_batches = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
        if self._metric_type == 'IOU':
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))

    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes ** 2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum()).item()
            self.num_updates += predictions.shape[0]

        if self._metric_type == 'ACC_BLNCD':
            predictions = pred.data.max(1, keepdim=True)[1].squeeze()
            y_true = gt.data.view_as(predictions).cpu().numpy()
            predictions = predictions.cpu().numpy()
            self.accuracy_balanced += balanced_accuracy_score(y_true, predictions)
            # if (predictions.min() not in [0, 1]) or\
            #     (predictions.max() not in [0, 1]) or\
            #     (y_true.min() not in [0, 1]) or\
            #     (y_true.max() not in [0, 1]):
            #     print('!!!!!')

            self.num_batches += 1  # different semantics than in ACC: there "num_samples_seen", here "num_batches_seen"
            self.accuracy += (predictions == y_true).sum()
            self.num_updates += predictions.shape[0]

        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti != 250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum(np.abs(gti[mask] - _pred.astype(np.int32)[mask]))
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_result(self):
        if self._metric_type == 'ACC':
            return {'acc': self.accuracy / self.num_updates}
        if self._metric_type == 'ACC_BLNCD':
            return {'acc_blncd': self.accuracy_balanced / self.num_batches,
                    'acc': self.accuracy / self.num_updates}
        if self._metric_type == 'L1':
            return {'l1': self.l1 / self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (
                    self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(
                self.confusion_matrix))
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc': acc_cls, 'mIOU': mean_iou}


def get_metrics(params):
    met = {}
    if 'cityscapes' == params['dataset']:
        if 'S' in params['tasks']:
            met['S'] = RunningMetric(metric_type='IOU', n_classes=19)
        if 'I' in params['tasks']:
            met['I'] = RunningMetric(metric_type='L1')
        if 'D' in params['tasks']:
            met['D'] = RunningMetric(metric_type='L1')
    if params['dataset'] in ['celeba', 'mnist',
                             'cifar10', 'cifar10_singletask', 'cifarfashionmnist',
                             'imagenette_singletask', 'imagenet_val']:
        if 'metric_type' in params:
            metric_type = params['metric_type']
        else:
            metric_type = 'ACC'
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type=metric_type)

    return met
