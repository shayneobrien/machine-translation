""" 
python seq2seq.py --devid -1 --nhid 200 --nlayers 1 --epochs 15 --lr 1e-3 --wd 0 --bsz 64 --clip 5 --maxlen 20 --minfreq 5 --beam 100

NOTE: Not masking artificially deflates perplexity due to assignment of probability mass to pad token, which is never seen in real world
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext import data
from torchtext import datasets
from torch.autograd import Variable

import spacy, random, argparse
import numpy as np
from tqdm import tqdm, tqdm_notebook

# Some utility functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

global USE_CUDA
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=DEVICE)

    parser.add_argument("--nhid", type=int, default=200)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--clip", type=float, default=5)
    
    parser.add_argument("--maxlen", type=float, default=20)
    parser.add_argument("--minfreq", type=float, default=5)
    parser.add_argument("--beam", type = int, default=100)
    
    return parser.parse_args()

args = parse_args()

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token = '<s>', eos_token = '</s>') # only target needs BOS/EOS
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), filter_pred=lambda x: len(vars(x)['src']) <= args.maxlen and len(vars(x)['trg']) <= args.maxlen)

DE.build_vocab(train.src, min_freq=args.minfreq)
EN.build_vocab(train.trg, min_freq=args.minfreq)

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=args.bsz, device=args.devid, repeat=False, sort_key=lambda x: len(x.src))

def str_to_tensor(string, src_lang = DE):
    string = string.split()
    word_ids = [src_lang.vocab.stoi[word] for word in string]
    word_tensor = Variable(torch.LongTensor(word_ids))
    if USE_CUDA:
        return word_tensor.cuda()
    else:
        return word_tensor
    
def tensor_to_kaggle(tensor, trg_lang = EN):
    return '|'.join([trg_lang.vocab.itos[word_id] for word_id in tensor])
    
def tensor_to_str(tensor, trg_lang = EN):
    return ' '.join([trg_lang.vocab.itos[word_id] for word_id in tensor])

class Encoder(nn.Module):
    def __init__(self, src_vsize, hidden_dim, n_layers = args.nlayers):
        super(Encoder, self).__init__()
        
        self.src_vsize = src_vsize
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embeddings = nn.Embedding(src_vsize, hidden_dim, padding_idx = DE.vocab.stoi[DE.pad_token])
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers = n_layers, batch_first = False)
        
    def forward(self, src_words):
        embedded = self.embeddings(src_words)
        out, hdn = self.lstm(embedded)
        return out, hdn

class Decoder(nn.Module):
    def __init__(self, hidden_dim, trg_vsize, n_layers = args.nlayers):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.trg_vsize = trg_vsize
        self.n_layers = n_layers
        
        self.embeddings = nn.Embedding(trg_vsize, hidden_dim, padding_idx = EN.vocab.stoi[EN.pad_token])
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers = n_layers, batch_first = False)
        self.proj = nn.Linear(hidden_dim, trg_vsize)
        
    def forward(self, trg_words, hidden):
        embedded = self.embeddings(trg_words)
        out, hdn = self.lstm(embedded, hidden)
        output = self.proj(out)
        return output, hdn
    
class Seq2Seq(nn.Module):
    def __init__(self, src_vsize, trg_vsize, hidden_dim, n_layers = args.nlayers):
        super(Seq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.encoder = Encoder(src_vsize, hidden_dim)
        self.decoder = Decoder(hidden_dim, trg_vsize)

class Trainer:
    def __init__(self, train_iter, val_iter):
        """ Initialize trainer class with Torchtext iterators """
        self.train_iter = train_iter
        self.val_iter = val_iter
        
    def train(self, num_epochs, model, lr = args.lr, weight_decay = args.wd, clip = args.clip):
        """ Train using Adam """
        best_ppl = 75
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params = parameters, lr = lr, weight_decay = weight_decay)
        
        all_losses = []
        for epoch in tqdm(range(1, num_epochs + 1)):

            epoch_loss = []
            for batch in tqdm(self.train_iter):
                
                optimizer.zero_grad()

                batch_loss = self.train_batch(batch, model)
                batch_loss.backward()

                nn.utils.clip_grad_norm(model.parameters(), clip)
                
                optimizer.step()

                epoch_loss.append(batch_loss.data[0])
                                
                if len(epoch_loss) % 100 == 0:
                    step = len(epoch_loss)
                    cur_loss = np.mean(epoch_loss)
                    train_ppl = np.exp(np.mean(epoch_loss))
                    print('Step: {0} | Loss: {1} | Train PPL: {2}'.format(step, cur_loss, train_ppl))
                    print('Wie würde eine solche Zukunft aussehen ? -->', self.translate('Wie würde eine solche Zukunft aussehen ?', model))
                
            epoch_loss = np.mean(epoch_loss)
            train_ppl = np.exp(epoch_loss)
            val_ppl = self.validate(model)

            print('Epoch: {0} | Loss: {1} | Train PPL: {2} | Val PPL: {3}'.format(epoch, epoch_loss, train_ppl, val_ppl))
            all_losses.append(epoch_loss)
            
            # early stopping
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_model = model
        
        torch.save(best_model.cpu(), best_model.__class__.__name__ + ".pth")
        return best_model.cpu(), all_losses        
                
    def train_batch(self, batch, model):
        """ Get train batch using teacher forcing (prev. true target is next word input) always. 
            Results in large speed-up. """
        
        # Get target length, create shift tensor (since we take word n-1 to predict word n)
        target_length = batch.trg.size()[0]
        shift = Variable(torch.LongTensor(batch.batch_size).fill_(1)).unsqueeze(0)
        if USE_CUDA:
            shift = shift.cuda()

        # Run words through encoder
        encoder_outputs, encoder_hidden = model.encoder(batch.src)

        # Get outputs for batch, using encoder hidden as initialization for decoder hidden
        decoder_outputs, decoder_hidden = model.decoder(batch.trg, encoder_hidden)

        # Reshape outputs, add shift tensor to targets
        preds = decoder_outputs.view(target_length * batch.batch_size, -1)
        targets = torch.cat((batch.trg[1:], shift), dim = 0).view(-1)

        # Compute loss in a batch (more efficient than loop)
        loss = F.cross_entropy(preds, targets)
        return loss
    
    def translate(self, string, model, maxlength = None):  
        """ Predict translation for an input string """
        # Make string a tensor
        tensor = str_to_tensor(string)
        tensor = tensor.unsqueeze(1)
        if USE_CUDA:
            tensor = tensor.cuda()

        # Run words through encoder
        encoder_outputs, decoder_hidden = model.encoder(tensor)

        # First token must always start of sentence <s>
        decoder_inputs = Variable(torch.LongTensor([EN.vocab.stoi[EN.init_token]])).unsqueeze(0)
        if USE_CUDA: 
            decoder_inputs = decoder_inputs.cuda()

        # if no maxlength, let it be 3*length original
        maxlength = maxlength if maxlength else 3 * tensor.shape[0]
        out_string = []

        # Predict words until an <eos> token or maxlength
        for trg_word_idx in range(maxlength):
            decoder_output, decoder_hidden = model.decoder(decoder_inputs, decoder_hidden)

            # Get most likely word index (highest value) from output
            prob_dist = F.log_softmax(decoder_output, dim = 2)
            top_probs, top_word_idx = prob_dist.data.topk(1, dim = 2)
            ni = top_word_idx.squeeze(0)

            decoder_inputs = Variable(ni) # Chosen word is next input
            out_string.append(ni[0][0])

            # Stop at end of sentence (not necessary when using known targets)
            if ni[0][0] == EN.vocab.stoi[EN.eos_token]: 
                break

        out_string = tensor_to_str(out_string)
        return out_string
    
    def evaluate_kaggle(self, string, model, ngrams = 3, context = 0, top_k = args.beam):
        """ Beam search the best starting trigrams for Kaggle input sentences """
        # Convert string to tensor for embedding lookups
        tensor = str_to_tensor(string)
        tensor = tensor.unsqueeze(1)
        if USE_CUDA:
            tensor = tensor.cuda()

        # Run words through encoder to get init hidden for decoder
        encoder_outputs, encoder_hidden = model.encoder(tensor)

        # Start collecting hiddens, prepare initial input variables
        decoder_inputs = Variable(torch.LongTensor([EN.vocab.stoi[EN.init_token]])).unsqueeze(0)
        if USE_CUDA: 
            decoder_inputs = decoder_inputs.cuda()

        # Compute the top K first words, so that we have something to work with
        decoder_output, decoder_hidden = model.decoder(decoder_inputs, encoder_hidden)
        prob_dist = F.log_softmax(decoder_output, dim = 2)
        top_probs, top_word_idx = prob_dist.data.topk(top_k, dim = 2)
        decoder_inputs = Variable(top_word_idx)
        if USE_CUDA:
            decoder_inputs = decoder_inputs.cuda()

        # Begin table to keep our outputs, output_probs
        outputs = [[word] for word in list(decoder_inputs.data[0][0])]
        output_probs = list(top_probs[0][0])

        # For using the correct hidden to predict next word. Initially it is 100x copy
        all_hiddens = [decoder_hidden for _ in range(top_k)]

        # Get top_k beams for 
        for _ in range(1, ngrams+context):
            beam_search_idx, beam_search_probs = [], []
            for k in range(top_k):
                decoder_output, new_hdn = model.decoder(decoder_inputs[:, :, k], all_hiddens[k])
                prob_dist = F.log_softmax(decoder_output, dim = 2)
                top_probs, top_word_idx = prob_dist.data.topk(top_k, dim = 2)
                beam_search_idx.append(list(top_word_idx[0][0]))
                beam_search_probs.append(list(top_probs[0][0]))
                all_hiddens[k] = new_hdn

            # Top K words idx
            next_word_idx = np.argsort(np.hstack(beam_search_probs))[::-1][:top_k] 

            # Backpointers to the input word that each top word was drawn from
            back_pointers = [int(np.floor(word / top_k)) for word in next_word_idx] 

            # Update output list with new decoder inputs and their corresponding probabilities
            next_words = [np.hstack(beam_search_idx)[ids] for ids in next_word_idx]
            next_probs = [np.hstack(beam_search_probs)[ids] for ids in next_word_idx]
            decoder_inputs = Variable(torch.LongTensor([int(word) for word in next_words])).unsqueeze(0).unsqueeze(0)
            if USE_CUDA:
                decoder_inputs = decoder_inputs.cuda()

            # update hiddens, outputs
            all_hiddens = [all_hiddens[pointer] for pointer in back_pointers]
            outputs = [outputs[pointer] + [word] for pointer, word in zip(back_pointers, next_words)]
            output_probs = [output_probs[pointer] + new_p for pointer, new_p in zip(back_pointers, next_probs)]

        prob_sort_idx = np.argsort(output_probs)[::-1]
        outputs = [outputs[idx] for idx in prob_sort_idx]
        outputs = [output[:ngrams] for output in outputs]
        out = [tensor_to_kaggle(tsr) for tsr in outputs]
        return ' '.join(out)
        
    def validate(self, model):
        """ Compute validation set perplexity """
        loss = []
        for batch in tqdm(self.val_iter):
            batch_loss = self.train_batch(batch, model)
            loss.append(batch_loss.data[0])
        
        val_ppl = np.exp(np.mean(loss))
        return val_ppl
    
    def write_kaggle(self, test_file, model):
        """ Write outputs to kaggle """
        with open(test_file, 'r') as fh:
            datasource = fh.read().splitlines()
        
        print('Evaluating on {0}...'.format(test_file))
        with open('output.txt', 'w') as fh:
            fh.write('id,word\n')
            for idx, string in tqdm(enumerate(datasource)):
                output = self.evaluate_kaggle(string, model)
                output = str(idx+1) + ',' + self.escape_kaggle(output) + '\n'
                fh.write(output)
        print('File saved.')
        
    def escape_kaggle(self, l):
        """ So kaggle doesn't yell at you when submitting results """
        return l.replace("\"", "<quote>").replace(",", "<comma>")

if __name__ == '__main__':
    model = Seq2Seq(src_vsize = len(DE.vocab.itos), trg_vsize = len(EN.vocab.itos), hidden_dim = args.nhid)
    trainer = Trainer(train_iter, val_iter)
    if USE_CUDA:
        model = model.cuda()
    print('Using cuda: ', np.all([parameter.is_cuda for parameter in model.parameters()]))
    model, all_losses = trainer.train(args.epochs, model)
    if USE_CUDA: # weird bug if not re-cuda'd
        model = model.cuda()
    trainer.write_kaggle('../data/source_test.txt', model)

