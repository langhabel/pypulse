from abc import abstractmethod

import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import theano.sparse as S
from sgd_optimizer import sgd
from model import Model
from convergence_criteria import ConvergenceCriterion
from l1_optimizer import L1Optimizer

theano.config.optimizer = 'fast_run'
theano.config.floatX = 'float64'


class SGDL1Optimizer(L1Optimizer):
    def __init__(self,
                 model: Model):
        super().__init__(model)
        self.w = theano.shared(np.zeros(1, dtype=theano.config.floatX), name='w')
        self.l1 = theano.shared(np.zeros(1, dtype=theano.config.floatX), name='l1',
                                broadcastable=self.w.broadcastable)
        self.data_length = theano.shared(1.0, name='data_length')
        self.learning_rate = theano.shared(1.0, name='learning_rate')
        self.F = S.csr_dmatrix('F')  # (labels, features)
        self.label_idx = T.iscalar('label_idx')  # ()
        self.loss = model.objective((self.F, self.label_idx, self.w, self.data_length))


    @abstractmethod
    def optimize(self,
                 feature_matrices,
                 matching_labels_indices,
                 weights,
                 l1_vector,
                 convergence_criterion: ConvergenceCriterion) -> np.ndarray:
        pass


class AdagradL1Optimizer(SGDL1Optimizer):
    def __init__(self,
                 model: Model):

        super().__init__(model)

        updates, self.u_acc, self.q_acc, self.gg_acc = adagrad_l1_update(self.loss,
                                                                         self.w,
                                                                         self.l1,
                                                                         learning_rate=self.learning_rate)

        self.training_op = theano.function(inputs=[self.F, self.label_idx],
                                           outputs=self.loss,
                                           updates=updates)

    def optimize(self,
                 feature_matrices,
                 matching_labels_indices,
                 weights,
                 l1_vector,
                 convergence_criterion: ConvergenceCriterion,
                 hot_starting_indices=None,
                 learning_rate=1.0,
                 initial_gradient_squared_accumulator_value=1e-7,
                 max_training_epochs=1000,
                 loss_convergence_factor=1e-5):
        self.w.set_value(weights)
        self.l1.set_value(l1_vector)
        self.learning_rate.set_value(learning_rate)

        if hot_starting_indices is None or len(hot_starting_indices) == 0:
            # init accumulators in first execution
            self.u_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
            self.q_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
            self.gg_acc.set_value(
                np.full(weights.shape, initial_gradient_squared_accumulator_value,
                        dtype=theano.config.floatX))
        else:
            self.u_acc.set_value(select_and_pad(self.u_acc.get_value(), hot_starting_indices))
            self.q_acc.set_value(select_and_pad(self.q_acc.get_value(), hot_starting_indices))
            self.gg_acc.set_value(select_and_pad(self.gg_acc.get_value(), hot_starting_indices,
                                                 padding=initial_gradient_squared_accumulator_value))

        sgd.optimize(feature_matrices,
                     matching_labels_indices,
                     self.training_op,
                     convergence_criterion,
                     max_training_epochs,
                     loss_convergence_factor)

        return self.w.get_value()


class AdadeltaL1Optimizer(SGDL1Optimizer):
    def __init__(self,
                 model: Model):

        super().__init__(model)
        self.rho = theano.shared(1.0, name='rho')
        self.eps = theano.shared(1.0, name='eps')

        updates, self.u_acc, self.q_acc, self.delta_acc, self.magnitudes_acc = adadelta_l1_update(
            self.loss,
            self.w,
            self.l1,
            learning_rate=self.learning_rate,
            rho=self.rho,
            eps=self.eps)

        self.training_op = theano.function(inputs=[self.F, self.label_idx],
                                           outputs=self.loss,
                                           updates=updates)

    def optimize(self,
                 feature_matrices,
                 matching_labels_indices,
                 weights,
                 l1_vector,
                 convergence_criterion: ConvergenceCriterion,
                 hot_starting_indices=None,
                 learning_rate=1.0,
                 eps=1e-3,
                 rho=0.99,
                 max_training_epochs=1000,
                 loss_convergence_factor=1e-5):
        self.w.set_value(weights)
        self.l1.set_value(l1_vector)
        self.learning_rate.set_value(learning_rate)
        self.eps.set_value(eps)
        self.rho.set_value(rho)

        if hot_starting_indices is None or len(hot_starting_indices) == 0:
            # init accumulators in first execution
            self.u_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
            self.q_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
            self.delta_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
            self.magnitudes_acc.set_value(np.zeros(weights.shape, dtype=theano.config.floatX))
        else:
            self.u_acc.set_value(select_and_pad(self.u_acc.get_value(), hot_starting_indices))
            self.q_acc.set_value(select_and_pad(self.q_acc.get_value(), hot_starting_indices))
            self.delta_acc.set_value(
                select_and_pad(self.delta_acc.get_value(), hot_starting_indices))
            self.magnitudes_acc.set_value(
                select_and_pad(self.magnitudes_acc.get_value(), hot_starting_indices))

        sgd.optimize(feature_matrices,
                     matching_labels_indices,
                     self.training_op,
                     convergence_criterion,
                     max_training_epochs,
                     loss_convergence_factor)

        return self.w.get_value()


def adagrad_l1_update(loss, w, l1, learning_rate=1.0):
    """Implements Adagrad SGD with Cumulative Penalty L1 Regularization.

    Tsuruoka et al. 2009: Stochastic Gradient Descent Training for L1-regularized Log-linear
    Models with Cumulative Penalty

    Duchi et al. 2011: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
    """
    g = theano.grad(loss, w)  # (tef_n, )
    updates = OrderedDict()

    # for each gradient-weights vector pair compile the following

    # empty initializations, later of size (tef_n, )
    # maximum penalty accumulator
    u = theano.shared(np.zeros(1, dtype=theano.config.floatX), broadcastable=w.broadcastable)

    # actual accumulated penalties
    q = theano.shared(np.zeros(1, dtype=theano.config.floatX), broadcastable=w.broadcastable)

    # squared gradients accumulator
    gg = theano.shared(np.zeros(1, dtype=theano.config.floatX), broadcastable=w.broadcastable)

    # adagrad learning rate upgrade
    gradient_squared_accu_new = gg + g ** 2  # (tef_n, )
    adagrad_learning_rate = learning_rate / T.sqrt(gradient_squared_accu_new)

    # total possible accumulated l1 penalty
    u_new = u + l1 * adagrad_learning_rate

    # update rule: w_k+1/2,i = w_k,i - adagrad_lr_k * g_i
    w_tmp = w - adagrad_learning_rate * g

    # apply penalties
    if T.gt(l1, 0):
        # if w_k+1/2,i > 0: w_k+1,i = max(0, w_k+1/2,i - (u_k,i + q_k-1,i)) else if: ...
        w_update = T.switch(T.gt(w_tmp, 0.),
                            T.maximum(w_tmp - (u_new + q), 0.),  # w_tmp > 0
                            T.switch(T.lt(w_tmp, 0.0),
                                     T.minimum(w_tmp + (u_new - q), 0.),  # w_tmp < 0
                                     0.)  # w_tmp == 0
                            )
    else:
        w_update = w_tmp

    # return updates (key: shared variable, value: symbolic variable)
    updates[w] = w_update
    updates[gg] = gradient_squared_accu_new
    updates[u] = u_new
    # actually accumulated penalty
    updates[q] = q + w_update - w_tmp

    return updates, u, q, gg


def adadelta_l1_update(loss, w, l1, learning_rate=1.0, rho=0.99, eps=1e-3):
    """Implements Adagrad SGD with Cumulative Penalty L1 Regularization.

    Tsuruoka et al. 2009: Stochastic Gradient Descent Training for L1-regularized Log-linear
    Models with Cumulative Penalty
    """
    g = theano.grad(loss, w)  # [(tef_n, )]
    updates = OrderedDict()

    # maximum penalty accumulator
    u = theano.shared(np.zeros(1, dtype=theano.config.floatX), broadcastable=w.broadcastable)

    # actual accumulated penalties
    q = theano.shared(np.zeros(1, dtype=theano.config.floatX), broadcastable=w.broadcastable)

    # running average of squared gradients
    magnitudes_acc = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                                   broadcastable=w.broadcastable)

    # running average of parameter updates
    delta_acc = theano.shared(np.zeros(1, dtype=theano.config.floatX),
                              broadcastable=w.broadcastable)

    # decaying average of past squared gradients
    magnitudes_acc_new = rho * magnitudes_acc + (T.constant(1) - rho) * g ** 2

    # get learning rate by division of root mean squared (ratio accu updates - gradients)
    adadelta_learning_rate = T.sqrt(delta_acc + eps) / T.sqrt(
        magnitudes_acc_new + eps) * learning_rate

    # adadelta update
    update = adadelta_learning_rate * g

    # decaying average of squared parameter updates
    delta_acc_new = rho * delta_acc + (T.constant(1) - rho) * update ** 2

    # accumulate maximal achievable penalties
    u_new = u + l1 * adadelta_learning_rate

    # update rule: w_k+1/2,i = w_k,i - adagrad_lr_k * g_i
    w_tmp = w - update

    # apply penalties
    if T.gt(l1, 0):
        # if w_k+1/2,i > 0: w_k+1,i = max(0, w_k+1/2,i - (u_k,i + q_k-1,i)) else if: ...
        w_update = T.switch(T.gt(w_tmp, 0.),
                            T.maximum(w_tmp - (u_new + q), 0.),  # w_tmp > 0
                            T.switch(T.lt(w_tmp, 0.0),
                                     T.minimum(w_tmp + (u_new - q), 0.),  # w_tmp < 0
                                     0.)  # w_tmp == 0
                            )
    else:
        w_update = w_tmp

    # return updates (key: shared variable, value: symbolic variable)
    updates[w] = w_update
    updates[u] = u_new
    updates[q] = q + w_update - w_tmp
    updates[delta_acc] = delta_acc_new
    updates[magnitudes_acc] = magnitudes_acc_new

    return updates, u, q, delta_acc, magnitudes_acc


def select_and_pad(x: np.ndarray,
                   selection_idx: np.ndarray,
                   padding=0.0,
                   dtype=theano.config.floatX):
    return np.hstack((x[selection_idx], np.full(len(x) - len(selection_idx), padding, dtype=dtype)))
