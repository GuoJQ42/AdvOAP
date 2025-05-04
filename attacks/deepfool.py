# https://github.com/BorealisAI/advertorch

from abc import ABCMeta

import torch

def replicate_input(x):
    return x.detach().clone()


def replicate_input_withgrad(x):
    return x.detach().clone().requires_grad_()

def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor[ii] = clamp(
            batch_tensor[ii], -vector[ii], vector[ii])
    """
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()

def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


class Attack(object):
    """
    Abstract base class for all attack classes.

    :param predict: forward pass function.
    :param loss_fn: loss function that takes .
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.

    """

    __metaclass__ = ABCMeta

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        """Create an Attack instance."""
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """Virtual method for generating the adversarial examples.

        :param x: the model's input tensor.
        :param **kwargs: optional parameters used by child classes.
        :return: adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    def _get_predicted_label(self, x):
        """
        Compute predicted labels given x. Used to prevent label leaking
        during adversarial training.

        :param x: the model's input tensor.
        :return: tensor containing predicted labels.
        """
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x = replicate_input(x)
        y = replicate_input(y)
        return x, y









class DeepfoolLinfAttack(Attack, LabelMixin):
    """
    A simple and fast gradient-based adversarial attack.
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
    "DeepFool: a simple and accurate method to fool deep neural
    networks", https://arxiv.org/abs/1511.04599

    :param predict: forward pass function.
    :param num_classes: number of classes considered
    :param nb_iter: number of iterations.
    :eps=0.1
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    def __init__(self, predict, num_classes=None, nb_iter=50, eps=0.1,
                 overshoot=0.02, clip_min=0., clip_max=1., loss_fn=None,
                 targeted=False):
        """
        Deepfool Linf Attack implementation in pytorch.
        """

        super(DeepfoolLinfAttack, self).__init__(predict, loss_fn=loss_fn,
                                                 clip_min=clip_min,
                                                 clip_max=clip_max)

        self.predict = predict
        self.num_classes = num_classes
        self.nb_iter = nb_iter
        self.eps = eps
        self.overshoot = overshoot
        self.targeted = targeted

    def is_adv(self, logits, y):
        # criterion
        y_hat = logits.argmax(-1)
        is_adv = y_hat != y
        return is_adv

    def get_deltas_logits(self, x, k, classes):
        # definition of loss_fn
        N = len(classes)
        rows = range(N)
        i0 = classes[:, 0]

        logits = self.predict(x)
        ik = classes[:, k]
        l0 = logits[rows, i0]
        lk = logits[rows, ik]
        delta_logits = lk - l0

        return {'sum_deltas': delta_logits.sum(),
                'deltas': delta_logits,
                'logits': logits}

    def get_grads(self, x, k, classes):
        deltas_logits = self.get_deltas_logits(x, k, classes)
        deltas_logits['sum_deltas'].backward()
        deltas_logits['grads'] = x.grad.clone()
        x.grad.data.zero_()
        return deltas_logits

    def get_distances(self, deltas, grads):
        return abs(deltas) / (
            grads.flatten(start_dim=2, end_dim=-1).abs().sum(axis=-1) + 1e-8)

    def get_perturbations(self, distances, grads):
        return self.atleast_kd(distances, grads.ndim) * grads.sign()

    def atleast_kd(self, x, k):
        shape = x.shape + (1,) * (k - x.ndim)
        return x.reshape(shape)

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x = replicate_input_withgrad(x)
        y = replicate_input(y)
        return x, y

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        x.requires_grad_()

        logits = self.predict(x)

        # get the classes
        classes = logits.argsort(axis=-1).flip(-1).detach()
        if self.num_classes is None:
            self.num_classes = logits.shape[-1]
        else:
            self.num_classes = min(self.num_classes, logits.shape[-1])

        N = len(x)
        rows = range(N)

        x0 = x
        p_total = torch.zeros_like(x)
        for _ in range(self.nb_iter):
            # let's first get the logits using k = 1 to see if we are done
            diffs = [self.get_grads(x, 1, classes)]

            is_adv = self.is_adv(diffs[0]['logits'], y)
            if is_adv.all():
                break

            diffs += [self.get_grads(x, k, classes) for k in range(2, self.num_classes)] # noqa

            deltas = torch.stack([d['deltas'] for d in diffs], dim=-1)
            grads = torch.stack([d['grads'] for d in diffs], dim=1)
            assert deltas.shape == (N, self.num_classes - 1)
            assert grads.shape == (N, self.num_classes - 1) + x0.shape[1:]

            # calculate the distances
            # compute f_k / ||w_k||
            distances = self.get_distances(deltas, grads)
            assert distances.shape == (N, self.num_classes - 1)

            # determine the best directions
            best = distances.argmin(axis=1)  # compute \hat{l}
            distances = distances[rows, best]
            deltas = deltas[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (N,)
            assert deltas.shape == (N,)
            assert grads.shape == x0.shape

            # apply perturbation
            distances = distances + 1e-4  # for numerical stability
            p_step = self.get_perturbations(distances, grads)  # =r_i
            assert p_step.shape == x0.shape

            p_total += p_step
            p_total = batch_clamp(self.eps, p_total)

            # don't do anything for those that are already adversarial
            x = torch.where(
                self.atleast_kd(is_adv, x.ndim),
                x,
                x0 + (1.0 + self.overshoot) * p_total,
            )  # =x_{i+1}

            x = clamp(x, min=self.clip_min, max=self.clip_max).clone().detach().requires_grad_() # noqa

        return x.detach()