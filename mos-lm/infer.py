import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gc

import data
import model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=-1,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--n_experts', type=int, default=10,
                    help='number of experts')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')

parser.add_argument('--model_file', type=str,  default='',
                    help='path to load the model from')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the log')

parser.add_argument('--log-file', type=str,  default='',
                    help='path to save the log')
parser.add_argument('--mmd_kernel_alpha', type=float,  default=0.5,
                    help='mmd kernel')
parser.add_argument('--mmd_lambda', type=float,  default=0.1,
                    help='mmd kernel')
parser.add_argument('--moment', action='store_true',
                    help='using moment regularization')
parser.add_argument('--moment_split', type=int, default=2000,
                    help='threshold for rare and popular words')
parser.add_argument('--moment_lambda', type=float, default=0.01,
                    help='lambda')
parser.add_argument('--adv', action='store_false',
                    help='using adversarial regularization')
parser.add_argument('--adv_bias', type=int, default=1000,
                    help='threshold for rare and popular words')
parser.add_argument('--adv_lambda', type=float, default=0.02,
                    help='lambda')
parser.add_argument('--adv_lr', type=float,  default=0.01,
                    help='adv learning rate')
parser.add_argument('--adv_wdecay', type=float,  default=1.2e-6,
                    help='adv weight decay')

parser.add_argument('--switch', type=int, default=200,
                    help='switch to asgd')
parser.add_argument('--epsilon', type=float, default=0.005,
                    help='switch to finetune')
parser.add_argument('--gaussian', type=float, default=0.0,
                    help='gaussian dropout')

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nhidlast, args.nlayers,
                       args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop,
                       args.tied, args.dropoutl, args.n_experts, args.epsilon, args.gaussian)
model.load_state_dict(torch.load(args.model_file))
model.gaussian = args.gaussian;model.dropouti = args.dropouti

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model

total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()
if args.adv:
    rate = (ntokens - args.adv_bias) * 1.0 / ntokens
    adv_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([rate, 1 - rate]).cuda())
    adv_hidden = nn.Linear(args.emsize, 2).cuda()
    adv_targets = torch.LongTensor(np.array([0] * args.adv_bias + [1] * (ntokens - args.adv_bias))).cuda()
    #adv_targets = Variable(adv_targets)
    adv_targets.requires_grad = True
    adv_hidden.weight.data.uniform_(-0.1, 0.1)
###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

# Loop over epochs.
lr = args.lr
stored_loss = evaluate(val_data)
logging('-' * 89)
logging('| Valid loss {:5.2f} | '
        'valid ppl {:8.2f}'.format(stored_loss, math.exp(stored_loss)))
logging('-' * 89)
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| Test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)

