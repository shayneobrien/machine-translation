Introduction
============

In this repo we train neural machine translation (NMT) systems
using end-to-end networks on the IWSLT-2016 dataset. This corpus
consists of Ted Talks translated between German and English. The
utterances of each language are aligned, which allows us to use it to
train translation systems. We implement (1) a sequence to sequence model
as described by (Sutskever et al., 2014); (2) an attention model as
introduced by (Bahdanau et. al., 2014) with dot-product attention
computation as per (Luong et al., 2015); (3) beam search to improve
translation quality, and; (4) visualizations of how the attention
mechanism makes a soft-alignment between words in the source and target
sentences. We also experiment with the effects of batch size, masking
padding, and teacher forcing on the validation set perplexity.

Problem Description
===================

In machine translation, the objective is to translate sentences from a
source language into a target language. In NMT, we approach this task by
jointly training end-to-end deep neural networks to predict the
likelihood of a sequence of output target tokens given a sequence of
input tokens of a different language. More formally, we define machine
translation as translating a source sentence <img src="/tex/ab80de912d15e049e5b345e3a41299f6.svg?invert_in_darkmode&sanitize=true" align=middle width=17.57554259999999pt height=22.465723500000017pt/> = <img src="/tex/a50c3a6cce0c5b640cc5bef1d62b99bd.svg?invert_in_darkmode&sanitize=true" align=middle width=14.393129849999989pt height=14.15524440000002pt/>,<img src="/tex/e378afcd7cae11e7306c61a9c35bf6cf.svg?invert_in_darkmode&sanitize=true" align=middle width=19.17798959999999pt height=14.15524440000002pt/>,
<img src="/tex/1b20726963975821d8d4f2ccfaba578b.svg?invert_in_darkmode&sanitize=true" align=middle width=16.19904164999999pt height=14.15524440000002pt/> = <img src="/tex/5d0660143caf7fd94fbe95b2a7a0cfe3.svg?invert_in_darkmode&sanitize=true" align=middle width=26.473118099999986pt height=34.337843099999986pt/> into a target sentence <img src="/tex/97790d793f190b3b985b582fea9ceb20.svg?invert_in_darkmode&sanitize=true" align=middle width=16.78561829999999pt height=22.465723500000017pt/> = <img src="/tex/add566ef276cab0dc7347620a8377612.svg?invert_in_darkmode&sanitize=true" align=middle width=14.206684799999989pt height=14.15524440000002pt/>,<img src="/tex/e378afcd7cae11e7306c61a9c35bf6cf.svg?invert_in_darkmode&sanitize=true" align=middle width=19.17798959999999pt height=14.15524440000002pt/>, <img src="/tex/d9b0b456e5794bd828f354f519cfa033.svg?invert_in_darkmode&sanitize=true" align=middle width=19.505804999999988pt height=14.15524440000002pt/>
= <img src="/tex/4120691aa89db70efa4601d06566afef.svg?invert_in_darkmode&sanitize=true" align=middle width=25.74555224999999pt height=34.337843099999986pt/>. Using this notation, we can define a translation system
as the function

<p align="center"><img src="/tex/6b50ee1af5fe304d8a00a96bc42d1b1d.svg?invert_in_darkmode&sanitize=true" align=middle width=122.34571964999999pt height=19.68035685pt/></p>

where <img src="/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> are the parameters of the model specifying the
probability distribution. We learn <img src="/tex/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode&sanitize=true" align=middle width=8.17352744999999pt height=22.831056599999986pt/> from data consisting of
aligned sentences in the source and target languages. In the case of
this problem set, German is our source language and English is our
target language.

Models and Algorithms
=====================

In the encoder-decoder framework, an encoder reads an input sequence
into a context vector <img src="/tex/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode&sanitize=true" align=middle width=7.11380504999999pt height=14.15524440000002pt/> and a decoder is trained to predict an output
token <img src="/tex/f8e685d6750d68bb9b2cd1a5e6616288.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=22.831056599999986pt/> given the context vector and all previously predicted
tokens <img src="/tex/23b0fe6e6c241feca9c011752da61cfc.svg?invert_in_darkmode&sanitize=true" align=middle width=98.26478474999998pt height=24.65753399999998pt/>. In
this way, the decoder defines a probability distribution over a
translation <img src="/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/>. Formally,

<p align="center"><img src="/tex/c178ace4d0158a28ee4eb0a6a6a24098.svg?invert_in_darkmode&sanitize=true" align=middle width=237.63298185pt height=47.60747145pt/></p>

During generation for unlabeled input sequences, tokens are usually
output until an end-of-sentence token is output or a specified maximum
length is reached.

In the context of this homework, this conditional probability is modeled
using recurrent neural networks (RNN). With this being the case, we let
<img src="/tex/aca72261c2e5f86e7a1bbbfa53001ec7.svg?invert_in_darkmode&sanitize=true" align=middle width=140.06848845pt height=24.65753399999998pt/> =
<img src="/tex/3c8614514ceb18a1bd59aa6389166220.svg?invert_in_darkmode&sanitize=true" align=middle width=86.64015524999999pt height=24.65753399999998pt/> for simplicity where <img src="/tex/1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode&sanitize=true" align=middle width=12.67127234999999pt height=14.15524440000002pt/> is the hidden state of
the RNN and <img src="/tex/f93ce33e511096ed626b4719d50f17d2.svg?invert_in_darkmode&sanitize=true" align=middle width=8.367621899999993pt height=14.15524440000002pt/> is a nonlinear, possibly multi-layered function that
outputs the probability of <img src="/tex/71c0437a67c94e48f18cc11d0c17a38c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=14.15524440000002pt/> (Neubig, 2017).

Encoder-Decoder
---------------

Inspired by (Cho et al., 2014) and (Sutskever et al., 2014), the
name encoder-decoder comes from the idea that we can use a model of two
RNNs to translate between languages: one that processes an input
sequence <img src="/tex/5201385589993766eea584cd3aa6fa13.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92464304999999pt height=22.465723500000017pt/> and “encodes” its information as a vector of real numbers
(hidden state), and another that is used to “decode” this vector to
predict the corresponding target sequence <img src="/tex/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode&sanitize=true" align=middle width=13.08219659999999pt height=22.465723500000017pt/>. In particular if we let
the encoder be expressed as RNN<img src="/tex/4667fb2573950c167856981c0bb49055.svg?invert_in_darkmode&sanitize=true" align=middle width=17.100001049999992pt height=29.190975000000005pt/>(<img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/>), then the decoder is
expressed as RNN<img src="/tex/5c0083fc276b49ee2618dabcab942ccb.svg?invert_in_darkmode&sanitize=true" align=middle width=16.51095104999999pt height=29.190975000000005pt/>(<img src="/tex/211dca2f7e396e7b572b4982e8ab3d19.svg?invert_in_darkmode&sanitize=true" align=middle width=4.5662248499999905pt height=14.611911599999981pt/>) and we can represent the overall model
for time step <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> as follows:

<p align="center"><img src="/tex/0888222cb88593339e7f0bfec2429de8.svg?invert_in_darkmode&sanitize=true" align=middle width=298.97227304999996pt height=192.9830298pt/></p>

where <img src="/tex/f5ddc45b18292a607cca0a2db7eb662a.svg?invert_in_darkmode&sanitize=true" align=middle width=32.85334139999999pt height=34.337843099999986pt/> is the source language embedding lookup,
<img src="/tex/458d9d90ef190d726d1652951b90d55f.svg?invert_in_darkmode&sanitize=true" align=middle width=27.602227949999993pt height=34.337843099999986pt/> is the encoder hidden state, <img src="/tex/681dd07f1866eda4408b1a0a2dcb90b2.svg?invert_in_darkmode&sanitize=true" align=middle width=32.26429139999999pt height=34.337843099999986pt/>
is the source language embedding lookup, <img src="/tex/5727fff37f71f3377e71f485ada37266.svg?invert_in_darkmode&sanitize=true" align=middle width=27.013177949999992pt height=34.337843099999986pt/> is the
decoder hidden state, and <img src="/tex/9ee1dfac44c56e3eac8a4fe64d4df0a5.svg?invert_in_darkmode&sanitize=true" align=middle width=27.013177949999992pt height=34.337843099999986pt/> is the softmax to turn
<img src="/tex/556fbe3167a9544674a95c1130889847.svg?invert_in_darkmode&sanitize=true" align=middle width=53.26899764999999pt height=31.598183099999996pt/>’s hidden state into a probability.

Note that <img src="/tex/7776e8ae45c1ebf7043f1ae9fb0aa36e.svg?invert_in_darkmode&sanitize=true" align=middle width=27.013177949999992pt height=34.337843099999986pt/> is initialized to <img src="/tex/431a3b404ad94813fff78fa19ac56232.svg?invert_in_darkmode&sanitize=true" align=middle width=10.747741949999993pt height=30.520578000000025pt/> and
<img src="/tex/3a7e0c9ce8ca8394de05391e148899c9.svg?invert_in_darkmode&sanitize=true" align=middle width=28.544986799999993pt height=34.337843099999986pt/> = <img src="/tex/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode&sanitize=true" align=middle width=7.11380504999999pt height=14.15524440000002pt/> the encoder has seen all words in the
source sentence. In this setup, we are feeding the true <img src="/tex/a6b173a311cd703aa85ef9c1d23f808f.svg?invert_in_darkmode&sanitize=true" align=middle width=29.44649729999999pt height=14.15524440000002pt/> target
as the input to the decoder for time step <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/>. This is known as teacher
forcing. When we are not employing teacher forcing, as is the case in
some of our experiments, <img src="/tex/681dd07f1866eda4408b1a0a2dcb90b2.svg?invert_in_darkmode&sanitize=true" align=middle width=32.26429139999999pt height=34.337843099999986pt/> <img src="/tex/810e7d96f19ed5daeacf60ac74b421a3.svg?invert_in_darkmode&sanitize=true" align=middle width=49.12350464999999pt height=22.465723500000017pt/>
where <img src="/tex/a6b173a311cd703aa85ef9c1d23f808f.svg?invert_in_darkmode&sanitize=true" align=middle width=29.44649729999999pt height=14.15524440000002pt/> = <img src="/tex/afb1352be3173ee3f13f293c26c35163.svg?invert_in_darkmode&sanitize=true" align=middle width=40.01724209999999pt height=34.337843099999986pt/>, the most likely output
token from the previous time step ().

Attention Decoder
-----------------

To incorporate attention into our decoder, we redefine the conditional
probability of <img src="/tex/71c0437a67c94e48f18cc11d0c17a38c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=14.15524440000002pt/> to be

<p align="center"><img src="/tex/e2ebc2effa230349f1f1515e4490c2e1.svg?invert_in_darkmode&sanitize=true" align=middle width=282.85694415pt height=16.438356pt/></p>

Here and unlike the formulation of <img src="/tex/6c8cb668f0b6017420cd007ea25c635c.svg?invert_in_darkmode&sanitize=true" align=middle width=34.49782544999999pt height=24.65753399999998pt/> in previous sections, the
probability is conditioned on a distinct context vector <img src="/tex/d3dcc43716c71f7ed358fd35a5f0a4ae.svg?invert_in_darkmode&sanitize=true" align=middle width=12.079597199999991pt height=14.15524440000002pt/> for each
target token <img src="/tex/71c0437a67c94e48f18cc11d0c17a38c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=14.15524440000002pt/>. The context vector <img src="/tex/d3dcc43716c71f7ed358fd35a5f0a4ae.svg?invert_in_darkmode&sanitize=true" align=middle width=12.079597199999991pt height=14.15524440000002pt/> depends on a sequence of
annotations <img src="/tex/5a95dbebd5e79e850a576db54f501ab8.svg?invert_in_darkmode&sanitize=true" align=middle width=16.02366149999999pt height=22.831056599999986pt/>, <img src="/tex/e378afcd7cae11e7306c61a9c35bf6cf.svg?invert_in_darkmode&sanitize=true" align=middle width=19.17798959999999pt height=14.15524440000002pt/>, <img src="/tex/d5231d154ec0bd2718bd57ba9a2c0698.svg?invert_in_darkmode&sanitize=true" align=middle width=14.436907649999991pt height=22.831056599999986pt/> to which an encoder maps the input
sequence. The context vector <img src="/tex/d3dcc43716c71f7ed358fd35a5f0a4ae.svg?invert_in_darkmode&sanitize=true" align=middle width=12.079597199999991pt height=14.15524440000002pt/> is then computed as a weighted sum of
these annotations:

<p align="center"><img src="/tex/8db681ff12c16a4fc821a922659407be.svg?invert_in_darkmode&sanitize=true" align=middle width=99.2865654pt height=50.04352485pt/></p>

where

<p align="center"><img src="/tex/ab2bee1e27b641a05cbfcfbbc5d288dc.svg?invert_in_darkmode&sanitize=true" align=middle width=138.10286985pt height=42.70067175pt/></p>

and

<p align="center"><img src="/tex/41aeaa8cac2c5e9403001c87d4f3d281.svg?invert_in_darkmode&sanitize=true" align=middle width=120.60128189999999pt height=17.031940199999998pt/></p>

is an <img src="/tex/d1c406257f326833378a145850af142e.svg?invert_in_darkmode&sanitize=true" align=middle width=75.76817159999999pt height=22.831056599999986pt/> <img src="/tex/bc23b377024c6b5b641825b4aecd9ac2.svg?invert_in_darkmode&sanitize=true" align=middle width=43.839604049999984pt height=22.831056599999986pt/> which scores how well the inputs around
position <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/> and the output at position <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> match. The alignment model
<img src="/tex/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.32879834999999pt height=22.465723500000017pt/> is parametrized as a simple feedforward neural network that is
jointly trained with all other components of the translation system.
Note that the alignment is not a latent variable, which allows
backpropagation through this gradient due to its differentiability.
Intuitively, <img src="/tex/d4a3e1c2f994878f0e0e132140309baa.svg?invert_in_darkmode&sanitize=true" align=middle width=21.585948449999993pt height=14.15524440000002pt/> reflects the importance of <img src="/tex/6d22be1359e204374e6f0b45e318d561.svg?invert_in_darkmode&sanitize=true" align=middle width=15.57562379999999pt height=22.831056599999986pt/> with respect
to <img src="/tex/7f55d81dc342c12cbe8394d0b8f0f0f6.svg?invert_in_darkmode&sanitize=true" align=middle width=29.49784034999999pt height=14.15524440000002pt/> in deciding <img src="/tex/1f1c28e0a1b1708c6889fb006c886784.svg?invert_in_darkmode&sanitize=true" align=middle width=12.67127234999999pt height=14.15524440000002pt/> and generating <img src="/tex/71c0437a67c94e48f18cc11d0c17a38c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.61992929999999pt height=14.15524440000002pt/>. It is believed that
giving the decoder an attention mechanism relieves the encoder from
having to encode all information in the source sequence needed for
translation to a fixed-length vector (Bahdanau et al., 2014).

Beam Search
-----------

Since the model is only trained to predict the probability of the next
word at a given time step in the target sentence, a decoding procedure
is needed to produce an actual translation, where the goal is to
maximize the probability of all words in the target sentence:

<p align="center"><img src="/tex/c178ace4d0158a28ee4eb0a6a6a24098.svg?invert_in_darkmode&sanitize=true" align=middle width=237.63298185pt height=47.60747145pt/></p>

The simplest approach is to proceed in a greedy fashion, taking the most
likely word under the model at each time step and feeding it back in as
the next input until the end-of-sentence token is produced:

<p align="center"><img src="/tex/ab4bc454f98d846654860e69bf0d9932.svg?invert_in_darkmode&sanitize=true" align=middle width=119.0842521pt height=23.9884194pt/></p>

This is not guaranteed to produce the highest-scoring sentence, though,
since in some cases the selection of a lower-probability word at a given
time-step will lead to downstream predictions that produce a
higher-scoring sentence overall. To account for this, we follow
(Sutskever et al., 2014) and use a beam search – instead of greedily
accumulating a single series of tokens across time-steps in the target
sentence, we instead keep track of a fixed-size set of candidate
decodings (the “beam”). At each step, we extend each candidate with all
words in the vocabulary, update the cumulative score for each new
candidate given the probabilities produced by the model, and then skim
off the <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> highest-scoring candidates to keep the size of the beam
constant. (Otherwise it would be a breadth-first search of all possible
permutations.) After the new beam is pruned, any candidates ending with
the end-of-sentence token are removed from the beam and added to a set
of completed candidates, and the beam size <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> is reduced by 1. This
continued until the beam is empty, and <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> complete candidate sentences
have been produced.

If the final set of candidate sentences is sorted on the summed
log-probabilities of each word, the procedure will strongly favor short
sentences, since the addition of each new word will drive down the joint
probability of the sentence. A range of strategies for length
normalization have been proposed, but we take the simplest approach,
which is to divide the total log-probability of the sentence by the
number of words to get an average per-word log-probability.

<p align="center"><img src="/tex/e46119df2534610ed192ad1152988d60.svg?invert_in_darkmode&sanitize=true" align=middle width=207.76043474999997pt height=19.68035685pt/></p>

Following (Sutskever et al., 2014) we use beam size of 10. And, in the
interest of speed – when extending candidates with new words at each
time-step, we only consider the 1000 most frequent words under the
model, which gives a significant speedup at little or no cost in
accuracy.

Experiments
===========

We experimented with a number of modifications to the baseline attention
model:

1.  In addition to the standard approach for calculating attention where
    the hidden layer $h_t^e$ is dotted with the hidden states at each
    time-step from the encoder:

    <p align="center"><img src="http://latex.codecogs.com/gif.latex?$\text{attn\_score}(H^{(f)},&space;h_t^{(e)})&space;:=&space;H_j^{(f)&space;\intercal}h_t^{(e)}$" title="$\text{attn\_score}(H^{(f)}, h_t^{(e)}) := H_j^{(f) \intercal}h_t^{(e)}$" /></p>

    We also implemented the attention scoring as a multi-layer
    perceptron as described by (Bahdanau et al., 2014):

    <p align="center"><img src="http://latex.codecogs.com/gif.latex?$$\text{attn\_score}(h_t^{(e)},&space;h_j^{(f)})&space;:=&space;w_{a2}^{\intercal}&space;\text{tanh}(W_{a1}[h_t^{(e)};h_j^{(f)}])$$" title="$$\text{attn\_score}(h_t^{(e)}, h_j^{(f)}) := w_{a2}^{\intercal} \text{tanh}(W_{a1}[h_t^{(e)};h_j^{(f)}])$$" /></p>

2.  We tried a range of learning rate schedulers, which train for a fixed
    number of epochs (e.g., {4, 8, 10}) and then decay the learning rate by 
    a factor of 0.5 every _N_ epochs after
    that. We tried _N_ = {2, 3, 4}.

3.  We implemented the decoder as a bidirectional LSTM as described by
    (Bahdanau et al., 2014).

4.  Last, we experimented with a wide range of hyperparameter settings:

    -   Word embedding dimensions: 200, 300, 500.

    -   LSTM hidden layers: 1, 2, 4.

    -   LSTM hidden layer dimensions: 200, 500, 1000.

    -   SGD and Adam optimizers.

    -   Batch lengths of 32 and 64.

    -   Clipping gradients to 5.

Though we didn’t have the resources to conduct a full grid search across
all combinations of these architectures and parameters, our best
performing models used 2-layer LSTMs with dot-product attention. All
final models were optimized with SGD starting with a learning rate of 1.
Early stopping was used to identify the best-performing model on the
validation set.

*Model* | *Accuracy* |
:---: | :---: |
Encoder-Decoder | 15.612
Dot Attention – 300d embeddings, 500d LSTMs, Bidirectional encoder | 9.680
Dot Attention – 500d embeddings, 1000d LSTMs | 9.672
Dot Attention – 200d embeddings, 200d LSTMs | 9.636
Dot Attention – 300d embeddings, 500d LSTMs | **9.501**

Generally, we found that it was difficult to find a single learning rate
scheduler that worked well for different hyperparameter settings – a
decay rate that seemed to work well for the 500-unit LSTM seemed too
slow for the 200-unit model, etc. Given indefinite resources, ideally we
would run a broad grid search to pick the best model.

See the following sections for visualizations of the attention weights and comparisons between the translations produced by the beam coder and the Google Translate predictions.

Translation examples
====================

We can compare the translations produced by the beam search decoder to
the predictions from Google translate:

*Source* | *Our Model* | *Google*
:---: | :---: | :---: |
Arbeit kam später, Heiraten kam später, Kinder kamen später, selbst der Tod kam später. | work came later , `<unk>` came later later , children came later later , even the death came later . | Work came later, marriages came later, children came later, even death came later
Das ist es , was Psychologen einen Aha-Moment nennen. | That ’s what psychologists call a `<unk>` call . | That’s what psychologists call an aha moment.
Dies ist nicht meine Meinung. Das sind Fakten.| This is not my opinion . These are facts . | This is not my opinion. These are facts.
In den 20ern sollte man sich also weiterbilden über den Körper und die eigenen Möglichkeiten | So in the `<unk>` , you should be thinking about the body and the own possibilities . | In the 20’s you should continue to educate yourself about the body and your own possibilities
Wie würde eine solche Zukunft aussehen? | What would such a future look like ? | What would such a future look like?

Attention visualizations
========================

We can visualize the dot-product attention by plotting the weights
assigned to the encoder hidden states for each step during decoding.

<p align="center">
  'Dies ist nicht meine Meinung. Das sind Fakten' --> 'This is not my opinion. These are facts.'
  <img src="imgs/att1.png">
</p>

<p align="center">
  'Oh, yeah! Ihr seid alle unglaublich' --> 'Oh, yeah! You're all incredible.'
  <img src="imgs/att2.png">
</p>

Conclusion
==========

We trained two classes of models – a basic encoder-decoder architecture
as described by (Sutskever et al., 2014) and a series of models that use
dot-product attention to give the decoder a more flexible representation
of the input. Though we experimented with more complex architectures such as
bidirectional LSTMs with multilayer perceptrons for attention weighting, our best-performing models used the basic dot-product attention.

References
==========

Bahdanau, D., Cho, K. & Bengio, Y. Neural Machine Translation by Jointly
Learning to Align and Translate. doi:10.1146/annurev.neuro.26.041002.131047. 2014.

Cho, K. et al. Learning Phrase Representations using RNN Encoder-Decoder
for Statistical Machine Translation. doi:10.3115/v1/D14-1179. 2014.

Luong, M.-T., Pham, H. & Manning, C. D. Effective Approaches to
Attention-based Neural Machine Translation. doi:10.18653/v1/D15-1166. 2015.

Neubig, G. Neural Machine Translation and Sequence-to-sequence Models: A
Tutorial. 2017.

Sutskever, I., Vinyals, O. & Le, Q. V. Sequence to sequence learning
with neural networks. Advances in Neural Information Processing Systems
(NIPS), pages 3104–3112. 2014.
