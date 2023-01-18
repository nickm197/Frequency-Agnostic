import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
#from torch.autograd import Variable
import data

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str,
                    help='name of model to eval')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
parser.add_argument('--val', action='store_true',
                    help='set for validation error, test by default')
parser.add_argument('--lamb', type=float, default=0.02,
                    help='decay parameter lambda')
parser.add_argument('--epsilon', type=float, default=0.001,
                    help='stabilization parameter epsilon')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate eta')
parser.add_argument('--ms', action='store_true',
                    help='uses mean squared gradients instead of sum squared')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size for gradient statistics')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence/truncation length')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='maximum number of training batches for gradient statistics')
# parser.add_argument('--n_experts', type=int, default=10, help='number of experts')


args = parser.parse_args()

if args.gpu>=0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False

model_name=args.model

print('loading')

corpus = data.Corpus(args.data)
eval_batch_size = 1
test_batch_size = 1

lr = args.lr
lamb = args.lamb
epsilon = args.epsilon

def gradstat():

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0

    for param in model.parameters():
        param.MS = 0. * param.data
    while i < train_data.size(0) - 1 - 1:
        seq_len = args.bptt
        model.use_dropout = False# model.eval()

        data, targets = get_batch(train_data, i, args)
        targets = targets.view(-1)
        hidden = repackage_hidden(hidden)

        #assumes model has atleast 2 returns, and first is output and second is hidden
        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

        loss.backward()

        for param in model.parameters():
            try:
                param.MS = param.MS + param.grad.data*param.grad.data
            except:
                pass
        total_loss += loss.data

        batch += 1


        i += seq_len
        if args.max_batches>0:
            if batch>= args.max_batches:
                break
    gsum = 0

    for param in model.parameters():
        try:
            if args.ms:
                param.MS = torch.sqrt(param.MS/batch)
            else:
                param.MS = torch.sqrt(param.MS)
            gsum+=torch.mean(param.MS)
        except:
            pass

    for param in model.parameters():
        try:
            param.decrate = param.MS/gsum
        except:
            pass

def evaluate():

    #clips decay rates at 1/lamb
    #otherwise scaled decay rates can be greater than 1
    #would cause decay updates to overshoot
    for param in model.parameters():
        try:
            if args.cuda:
                decratenp = param.decrate.cpu().numpy()
                ind = np.nonzero(decratenp>(1/lamb))
                decratenp[ind] = (1/lamb)
                param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)
                param.data0 = 1*param.data
            else:
                decratenp = param.decrate.numpy()
                ind = np.nonzero(decratenp>(1/lamb))
                decratenp[ind] = (1/lamb)
                param.decrate = torch.from_numpy(decratenp).type(torch.FloatTensor)
                param.data0 = 1*param.data
        except:
            pass

    total_loss = 0

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    last = False
    seq_len= args.bptt
    seq_len0 = seq_len
    #loops through data
    while i < eval_data.size(0) - 1 - 1:

        model.use_dropout = False#model.eval()
        #gets last chunk of seqlence if seqlen doesn't divide full sequence cleanly
        if (i+seq_len)>=eval_data.size(0):
            if last:
                break
            seq_len = eval_data.size(0)-i-1
            last = True

        data, targets = get_batch(eval_data, i, args)
        targets = targets.view(-1)

        hidden = repackage_hidden(hidden)

        model.zero_grad()

        #assumes model has at least 2 returns, and first is output and second is hidden
        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        #compute gradient on sequence segment loss
        loss.backward()

        #update rule
        for param in model.parameters():
            try:
                dW = lamb*param.decrate*(param.data0-param.data)-lr*param.grad.data/(param.MS+epsilon)
                param.data+=dW
            except:
                pass

        #seq_len/seq_len0 will be 1 except for last sequence
        #for last sequence, we downweight if sequence is shorter
        total_loss += (seq_len/seq_len0)*loss.data
        batch += (seq_len/seq_len0)

        i += seq_len

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)

    perp = torch.exp(total_loss/batch)
    if args.cuda:
        return perp.cpu().numpy()
    else:
        return perp.numpy()

#load model
with open(model_name, 'rb') as f:
    model = torch.load(f)

ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

if args.val== True:
    eval_data= val_data
else:
    eval_data=test_data
train_data = batchify(corpus.train, args.batch_size, args)

print('collecting gradient statistics')
#collect gradient statistics on training data
model.train()
gradstat()
#model.eval()
#change batch size to 1 for dynamic eval
args.batch_size=1
print('running dynamic evaluation')
#apply dynamic evaluation
loss = evaluate()
print('perplexity loss: ' + str(loss[0]))
