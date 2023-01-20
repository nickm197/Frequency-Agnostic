import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
#from torch.autograd import Variable
import data
import pickle

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
parser.add_argument('--text_file', type=str,
                    help='filename of the text to test')
# parser.add_argument('--n_experts', type=int, default=10, help='number of experts')

global tokens, lps

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
text = data.TestSet(args.text_file, corpus.dictionary)
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

    tokens = []
    lps = []

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

    model.use_dropout = False  # model.eval()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    #loops through data
    src = input[0]
    indices = [src.item()]
    print('src', src, indices, corpus.dictionary.idx2word[src.item()])
    #while i < input.size(0) - 1:# - 1:
    model.zero_grad()
    while i < 40:# - 1:

        #for d in data[0].tolist():
        #      print(d, corpus.dictionary.idx2word[d])

        #hidden = repackage_hidden(hidden)

        src = torch.LongTensor([indices]).cuda()

        #print('src', src, indices)

        hidden = model.init_hidden(i+1)

        #assumes model has at least 2 returns, and first is output and second is hidden
        log_prob, hidden = model(src, hidden)
        loss = log_prob.view(-1, log_prob.size(2))[0, input[i+1].item()]
        #loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), torch.LongTensor([input[i+1]]).cuda())

        tokens.append(corpus.dictionary.idx2word[input[i+1].item()])
        lps.append(loss.data.item())
        #print(log_prob.size())
        print(input[i+1].item(), corpus.dictionary.idx2word[input[i+1].item()], loss.data.item())

        indices.append(input[i+1].item())

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
        total_loss += loss.data
        batch += 1

        i += 1

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)
    name = 'tokens.lps'
    print(name)
    with open(name, 'wb') as file:
        pickle.dump((tokens,  lps), file)

    perp = total_loss/batch
    print(batch, log_prob)
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
train_data = batchify(corpus.train, args.batch_size, args)
text_data = batchify(text.text, eval_batch_size, args)

print('collecting gradient statistics')
#collect gradient statistics on training data
model.train()
gradstat()
#model.eval()
#change batch size to 1 for dynamic eval
args.batch_size=1
print('running dynamic evaluation')
#apply dynamic evaluation
eval_data= val_data
args.bptt=1


input = corpus.valid

loss = evaluate()
print('-' * 89)
print('| Valid loss {:5.2f} | '
        'valid ppl {:8.2f}'.format(loss, math.exp(-loss)))
print('-' * 89)
#eval_data=test_data
#loss = evaluate()
#print('=' * 89)
#print('| Test loss {:5.2f} | test ppl {:8.2f}'.format(
#    loss, math.exp(loss)))
#print('=' * 89)
#eval_data=text_data
#loss = evaluate()
#print('=' * 89)
#print('| Text loss {:5.2f} | text ppl {:8.2f}'.format(
#    loss, math.exp(loss)))
#print('=' * 89)