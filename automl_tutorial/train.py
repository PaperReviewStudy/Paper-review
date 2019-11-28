import os
from sacred import Experiment
from sacred.observers import MongoObserver
import models

import argparse
ex = Experiment('juntae')
ex.observers.append(MongoObserver.create(url='mongodb+srv://juntae:qlalfqjgh1!@cluster0-meoyr.gcp.mongodb.net/test?retryWrites=true&w=majority',
                                         db_name='experiments'))

def get_params():
    parser = argparse.ArgumentParser()

    #data hyperparamters
    parser.add_argument("--batch_size", type=int, default=64)

    #model hyperparameter
    parser.add_argument("--hidden_dim1", type=int, default=64)
    parser.add_argument("--hidden_dim2", type=int, default=128)
    parser.add_argument("--hidden_dim3", type=int, default=256)
    parser.add_argument("--hidden_dim4", type=int, default=512)

    #optimizer hyperparameter
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default='Adam')

    #learn hyperparameter
    parser.add_argument("--epoch", type=int, default=10)

    args = parser.parse_args()
    return args


@ex.config
def hyperparam():
    """hyperparam"""

    args = get_params()

    #data hyperparameter
    """@nni.variable(nni.choice(32, 64, 128, 256, 512, 1024), name=args.batch_size)"""
    args.batch_size = args.batch_size

    #model hyperparameter
    """@nni.variable(nni.choice(32, 64, 128, 512, 1024), name=args.hidden_dim1)"""
    args.hidden_dim1 = args.hidden_dim1
    """@nni.variable(nni.choice(32, 64, 128, 512, 1024), name=args.hidden_dim2)"""
    args.hidden_dim2 = args.hidden_dim2
    """@nni.variable(nni.choice(32, 64, 128, 512, 1024), name=args.hidden_dim3)"""
    args.hidden_dim3 = args.hidden_dim3
    """@nni.variable(nni.choice(32, 64, 128, 512, 1024), name=args.hidden_dim4)"""
    args.hidden_dim4 = args.hidden_dim4


    #optimizer hyperparamter
    """@nni.variable(nni.loguniform(0.0001, 0.1), name=args.lr)"""
    args.lr = args.lr
    """@nni.variable(nni.choice('adam', 'SGD'), name=args.optimizer)"""
    args.optimizer = args.optimizer

    #learn hyperparamter
    """@nni.variable(nni.choice(100), name=args.epoch)"""
    args.epoch = args.epoch

    print("hyperparams", args)

@ex.automain
def run(args):
    test_loss = models.train_model(args, ex)

    ex.log_scalar('loss', test_loss)
    return test_loss
