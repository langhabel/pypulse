import time
import logging
import numpy as np
from theano.compile.function_module import Function
from typing import List
from convergence_criteria import ConvergenceCriterion

logger = logging.getLogger('sgd')


def optimize(data_points: List[any],
             matching_labels_indices: np.array,
             training_op: Function,
             convergence_criterion: ConvergenceCriterion,
             max_training_epochs=100,
             convergence_by_loss=1e-5,
             seed=0):
    start = time.time()
    np.random.seed(seed)
    n = len(data_points)

    logger.info('Run SGD optimization...')

    epoch = 0
    total_loss = 0.0
    while epoch < max_training_epochs:
        total_loss = 0.0

        # shuffle mini-batches
        indices = np.arange(n)
        np.random.shuffle(indices)

        for idx in indices:
            loss = training_op(data_points[idx], matching_labels_indices[idx])
            total_loss += loss

        epoch += 1

        logger.info('Loss in epoch {0}: {1:1.4f}'.format(epoch, total_loss))

        if convergence_criterion.is_converged():
            logger.info('Convergence by criterion.')
            break
        elif total_loss < convergence_by_loss:
            logger.info('Convergence by loss.')
            break

    logger.info('Final loss after epoch {0}: {1:1.4f} -- {2:5.2f}s'.format(epoch,
                                                                           total_loss,
                                                                           time.time() - start))
