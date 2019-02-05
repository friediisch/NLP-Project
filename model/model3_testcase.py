import spacy as sp
import os
from pathlib import Path
import json
import sys
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Span
import numpy as np 
import keras
import editdistance
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Add
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.layers import Lambda
from time import time
import gensim
import sys
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from random import shuffle
from time import sleep
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
import utils
import pandas as pd
from keras.layers import Dropout
from keras import regularizers

title = 'CAN RECURRENT NEURAL NETWORKS WARP TIME'


abstract = \
'''Successful  recurrent  models  such  as  long  short-term  memories  (LSTMs)  and
gated recurrent units (GRUs) use
ad hoc
gating mechanisms.  Empirically these
models have been found to improve the learning of medium to long term temporal
dependencies and to help with vanishing gradient issues.
We  prove  that  learnable  gates  in  a  recurrent  model  formally  provide
quasi-
invariance  to  general  time  transformations
in  the  input  data.   We  recover  part
of the LSTM architecture from a simple axiomatic approach.
This result leads to a new way of initializing gate biases in LSTMs and GRUs. Ex-
perimentally, this new
chrono initialization
is shown to greatly improve learning
of long term dependencies, with minimal implementation effort.'''

text =  \
'''
Published as a conference paper at ICLR 2018
explains why this is reasonable in most cases, when facing medium term dependencies, but
fails when facing long to very long term dependencies.
∙
We test the empirical benefits of the new initialization on both synthetic and real world data
(Section 3). We observe substantial improvement with long-term dependencies, and slight
gains or no change when short-term dependencies dominate.
1
F
ROM TIME WARPING INVARIANCE TO GATING
When tackling sequential learning problems,  being resilient to a change in time scale is crucial.
Lack of resilience to time rescaling implies that we can make a problem arbitrarily difficult simply
by changing the unit of measurement of time.  Ordinary recurrent neural networks are highly non-
resilient to time rescaling: a task can be rendered impossible for an ordinary recurrent neural network
to learn, simply by inserting a fixed, small number of zeros or whitespaces between all elements of
the input sequence.   An explanation is that,  with a given number of recurrent units,  the class of
functions representable by an ordinary recurrent network is not invariant to time rescaling.
Ideally, one would like a recurrent model to be able to learn from time-warped input data
푥
(
푐
(
푡
))
as
easily as it learns from data
푥
(
푡
)
, at least if the time warping
푐
(
푡
)
is not overly complex. The change
of time
푐
may represent not only time rescalings, but, for instance, accelerations or decelerations of
the phenomena in the input data.
We call a class of models
invariant to time warping
, if for any model in the class with input data
푥
(
푡
)
, and for any time warping
푐
(
푡
)
, there is another (or the same) model in the class that behaves
on data
푥
(
푐
(
푡
))
in the same way the original model behaves on
푥
(
푡
)
.  (In practice, this will only
be possible if the warping
푐
is not too complex.)  We will show that this is deeply linked to having
gating mechanisms in the model.
Invariance to time rescaling
Let us first discuss the simpler case of a linear time rescaling. Formally, this is a linear transformation
of time, that is
푐
:
R
+
−→
R
+
푡
↦−→
훼푡
(1)
with
훼 >
0
. For instance, receiving a new input character every
10
time steps only, would correspond
to
훼
= 0
.
1
.
Studying time transformations is easier in the continuous-time setting.  The discrete time equation
of a basic recurrent network with hidden state
ℎ
푡
,
ℎ
푡
+1
= tanh (
푊
푥
푥
푡
+
푊
ℎ
ℎ
푡
+
푏
)
(2)
can be seen as a time-discretized version of the continuous-time equation
1
d
ℎ
(
푡
)
d
푡
= tanh
(︀
푊
푥
푥
(
푡
) +
푊
ℎ
ℎ
(
푡
) +
푏
)︀
−
ℎ
(
푡
)
(3)
namely, (2) is the Taylor expansion
ℎ
(
푡
+
훿푡
)
≈
ℎ
(
푡
) +
훿푡
d
ℎ
(
푡
)
d
푡
with discretization step
훿푡
= 1
.
Now imagine that we want to describe time-rescaled data
푥
(
훼푡
)
with a model from the same class.
Substituting
푡
←
푐
(
푡
) =
훼푡
,
푥
(
푡
)
←
푥
(
훼푡
)
and
ℎ
(
푡
)
←
ℎ
(
훼푡
)
and rewriting (3) in terms of the new
variables, the time-rescaled model satisfies
2
d
ℎ
(
푡
)
d
푡
=
훼
tanh
(︀
푊
푥
푥
(
푡
) +
푊
ℎ
ℎ
(
푡
) +
푏
)︀
−
훼ℎ
(
푡
)
.
(4)
However, when translated back to a discrete-time model, this no longer describes an ordinary RNN
but a
leaky
RNN (Jaeger, 2002, §8.1). Indeed, taking the Taylor expansion of
ℎ
(
푡
+
훿푡
)
with
훿푡
= 1
in (4) yields the recurrent model
ℎ
푡
+1
=
훼
tanh (
푊
푥
푥
푡
+
푊
ℎ
ℎ
푡
+
푏
) + (1
−
훼
)
ℎ
푡
(5)
1
We will use indices
ℎ
푡
for discrete time and brackets
ℎ
(
푡
)
for continuous time.
2
More precisely, introduce a new time variable
 
and set the model and data with variable
 
to
퐻
(
 
)
:
=
ℎ
(
푐
(
 
))
and
푋
(
 
)
:
=
푥
(
푐
(
 
))
. Then compute
d
퐻
(
 
)
d
 
. Then rename
퐻
to
ℎ
,
푋
to
푥
and
 
to
푡
to match the
original notation.
2
Published as a conference paper at ICLR 2018
Thus, a straightforward way to ensure that a class of (continuous-time) models is able to represent
input data
푥
(
훼푡
)
in the same way that it can represent input data
푥
(
푡
)
,  is to take a leaky model
in which
훼 >
0
is a learnable parameter,  corresponding to the coefficient of the time rescaling.
Namely, the class of ordinary recurrent networks is not invariant to time rescaling, while the class
of leaky RNNs (5) is.
Learning
훼
amounts to learning the global characteristic timescale of the problem at hand.  More
precisely,
1
/훼
ought to be interpreted as the characteristic forgetting time of the neural network.
3
Invariance to time warpings
In all generality, we would like recurrent networks to be resilient not only to time rescaling, but to
all sorts of time transformations of the inputs, such as variable accelerations or decelerations.
An eligible time transformation, or
time warping
, is any increasing differentiable function
푐
from
R
+
to
R
+
.  This amounts to facing input data
푥
(
푐
(
푡
))
instead of
푥
(
푡
)
.  Applying a time warping
푡
←
푐
(
푡
)
to the model and data in equation (3) and reasoning as above yields
d
ℎ
(
푡
)
d
푡
=
d
푐
(
푡
)
d
푡
tanh
(︀
푊
푥
푥
(
푡
) +
푊
ℎ
ℎ
(
푡
) +
푏
)︀
−
d
푐
(
푡
)
d
푡
ℎ
(
푡
)
.
(6)
Ideally, one would like a model to be able to learn from input data
푥
(
푐
(
푡
))
as easily as it learns from
data
푥
(
푡
)
, at least if the time warping
푐
(
푡
)
is not overly complex.
To be invariant to time warpings, a class of (continuous-time) models has to be able to represent
Equation (6) for any time warping
푐
(
푡
)
.  Moreover, the time warping is unknown a priori, so would
have to be learned.
Ordinary recurrent networks do not constitute a model class that is invariant to time rescalings, as
seen above. A fortiori, this model class is not invariant to time warpings either.
For time warping invariance,  one has to introduce a
learnable
function
푔
that will represent the
derivative
4
of the time warping,
d
푐
(
푡
)
d
푡
in (6).   For instance
푔
may be a recurrent neural network
taking the
푥
’s as input.
5
Thus we get a class of recurrent networks defined by the equation
d
ℎ
(
푡
)
d
푡
=
푔
(
푡
) tanh
(︀
푊
푥
푥
(
푡
) +
푊
ℎ
ℎ
(
푡
) +
푏
)︀
−
푔
(
푡
)
ℎ
(
푡
)
(7)
where
푔
belongs to a large class (universal approximator) of functions of the inputs.
The class of recurrent models (7) is
quasi
-invariant to time warpings. The quality of the invariance
will depend on the learning power of the learnable function
푔
:  a function
푔
that can represent any
function  of  the  data  would  define  a  class  of  recurrent  models  that  is  perfectly  invariant  to  time
warpings; however, a specific model for
푔
(e.g., neural networks of a given size) can only represent
a specific, albeit large, class of time warpings, and so will only provide quasi-invariance.
Heuristically,
푔
(
푡
)
acts as a time-dependent version of the fixed
훼
in (4). Just like
1
/훼
above,
1
/푔
(
푡
0
)
represents the local forgetting time of the network at time
푡
0
:  the network will effectively retain
information about the inputs at
푡
0
for a duration of the order of magnitude of
1
/푔
(
푡
0
)
(assuming
푔
(
푡
)
does not change too much around
푡
0
).
Let us translate back this equation to the more computationally realistic case of discrete time, using
a Taylor expansion with step size
훿푡
= 1
, so that
d
ℎ
(
푡
)
d
푡
=
···
becomes
ℎ
푡
+1
=
ℎ
푡
+
···
. Then the
model (7) becomes
ℎ
푡
+1
=
푔
푡
tanh (
푊
푥
푥
푡
+
푊
ℎ
ℎ
푡
+
푏
) + (1
−
푔
푡
)
ℎ
푡
.
(8)
3
Namely, in the “free” regime if inputs stop after a certain time
푡
0
,
푥
(
푡
) = 0
for
푡 > 푡
0
, with
푏
= 0
and
푊
ℎ
= 0
, the solution of (4) is
ℎ
(
푡
) =
푒
−
훼
(
푡
−
푡
0
)
ℎ
(
푡
0
)
, and so the network retains information from the past
푡 < 푡
0
during a time proportional to
1
/훼
.
4
It  is,  of  course,  algebraically  equivalent  to  introduce  a  function
푔
that  learns  the  derivative  of
푐
,  or  to
introduce a function
퐺
that learns
푐
.  However, only the derivative of
푐
appears in (6).  Therefore the choice
to work with
d
푐
(
푡
)
d
푡
is more convenient. Moreover, it may also make learning easier, because the simplest case
of a time warping is a time rescaling, for which
d
푐
(
푡
)
d
푡
=
훼
is a constant.  Time warpings
푐
are increasing by
definition: this translates as
푔 >
0
.
5
The time warping has to be learned only based on the data seen so far.
3
Published as a conference paper at ICLR 2018
where
푔
푡
itself is a function of the inputs.
This model is the simplest extension of the RNN model that provides invariance to time warpings.
6
It is a basic gated recurrent network, with input gating
푔
푡
and forget gating
(1
−
푔
푡
)
.
Here
푔
푡
has to be able to learn an arbitrary function of the past inputs
푥
; for instance, take for
푔
푡
the
output of a recurrent network with hidden state
ℎ
푔
:
푔
푡
=
휎
(
푊
푔푥
푥
푡
+
푊
푔ℎ
ℎ
푔
푡
+
푏
푔
)
(9)
with sigmoid activation function
휎
(more on the choice of sigmoid below).  Current architectures
just reuse for
ℎ
푔
the states
ℎ
of the main network (or, equivalently, relabel
ℎ
←
(
ℎ,ℎ
푔
)
to be the
union of both recurrent networks and do not make the distinction).
The  model  (8)  provides  invariance  to
global
time  warpings,  making  all  units  face  the  same  di-
lation/contraction  of  time.   One  might,  instead,  endow  every  unit
푖
with  its  own  local  contrac-
tion/dilation function
푔
푖
. This offers more flexibility (gates have been introduced for several reasons
beyond time warpings (Hochreiter, 1991)), especially if several unknown timescales coexist in the
signal:  for instance, in a multilayer model, each layer may have its own characteristic timescales
corresponding to different levels of abstraction from the signal. This yields a model
ℎ
푖
푡
+1
=
푔
푖
푡
tanh
(︁
푊
푖
푥
푥
푡
+
푊
푖
ℎ
ℎ
푡
+
푏
푖
)︁
+ (1
−
푔
푖
푡
)
ℎ
푖
푡
(10)
with
ℎ
푖
and
(
푊
푖
푥
,푊
푖
ℎ
,푏
푖
)
being respectively the activation and the incoming parameters of unit
푖
,
and with each
푔
푖
a function of both inputs and units.
Equation 10 defines a simple form of gated recurrent network, that closely resembles the evolution
equation of
cell
units in LSTMs, and of hidden units in GRUs.
In (10), the forget gate is tied to the input gate (
푔
푖
푡
and
1
−
푔
푖
푡
). Such a setting has been successfully
used before (e.g. (Lample et al., 2016)) and saves some parameters, but we are not aware of system-
atic comparisons. Below, we
initialize
LSTMs this way but do not enforce the constraint throughout
training.
Continuous time versus discrete time
Of course, the analogy between continuous and discrete time breaks down if the Taylor expansion is
not valid. The Taylor expansion is valid when the derivative of the time warping is not too large, say,
when
훼
.
1
or
푔
푡
.
1
(then (8) and (7) are close). Intuitively, for continuous-time data, the physical
time increment corresponding to each time step
푡
→
푡
+ 1
of the discrete-time recurrent model
should be smaller than the speed at which the data changes, otherwise the situation is hopeless.  So
discrete-time gated models are invariant to time warpings that stretch time (such as interspersing the
data with blanks or having long-term dependencies), but obviously not to those that make things
happen too fast for the model.
Besides, since time warpings are monotonous, we have
d
푐
(
푡
)
d
푡
>
0
, i.e.,
푔
푡
>
0
. The two constraints
푔
푡
>
0
and
푔
푡
<
1
square nicely with the use of a sigmoid for the gate function
푔
.
2
T
IME WARPINGS AND GATE INITIALIZATION
If  we  happen  to  know  that  the  sequential  data  we  are  facing  have  temporal  dependencies  in  an
approximate range
[
 
min
, 
max
]
, it seems reasonable to use a model with memory (forgetting time)
lying approximately in the same temporal range. As mentioned in Section 1, this amounts to having
values of
푔
in the range
[︁
1
 
max
,
1
 
min
]︁
.
The biases
푏
푔
of the gates
푔
greatly impact the order of magnitude of the values of
푔
(
푡
)
over time.
If the values of both inputs and hidden layers are centered over time,
푔
(
푡
)
will typically take values
6
Even more:  the weights
(
푊
푥
,푊
ℎ
,푏
)
are the same for
ℎ
(
푡
)
in (3) and
ℎ
(
푐
(
푡
))
in (6).  This means that
in
principle it is not necessary to re-train the model for the time-warped data
.  (Assuming, of course, that
푔
푡
can
learn the time warping efficiently.) The variable copy task (Section 3) arguably illustrates this. So the definition
of time warping invariance could be strengthened to use the
same
model before and after warping.
4
Published as a conference paper at ICLR 2018
centered around
휎
(
푏
푔
)
. Values of
휎
(
푏
푔
)
in the desired range
[︁
1
 
max
,
1
 
min
]︁
are obtained by choosing
the biases
푏
푔
between
−
log(
 
max
−
1)
and
−
log(
 
min
−
1)
. This is a loose prescription: we only
want to control the order of magnitude of the memory range of the neural networks. Furthermore, we
don’t want to bound
푔
(
푡
)
too tightly to some value forever:  if rare events occur, abruplty changing
the time scale can be useful. Therefore we suggest to use these values as initial values only.
This suggests a practical initialization for the bias of the gates of recurrent networks such as (10):
when characteristic timescales of the sequential data at hand are expected to lie between
 
min
and
 
max
, initialize the biases of
푔
as
−
log(
풰
([
 
min
, 
max
])
−
1)
where
풰
is the uniform distribution
7
.
For LSTMs, using a variant of (Graves et al., 2013):
푖
푡
=
휎
(
푊
푥푖
푥
푡
+
푊
ℎ푖
ℎ
푡
−
1
+
푏
푖
)
(11)
푓
푡
=
휎
(
푊
푥푓
푥
푡
+
푊
ℎ푓
ℎ
푡
−
1
+
푏
푓
)
(12)
푐
푡
=
푓
푡
푐
푡
−
1
+
푖
푡
tanh(
푊
푥푐
푥
푡
+
푊
ℎ푐
ℎ
푡
−
1
+
푏
푐
)
(13)
표
푡
=
휎
(
푊
푥표
푥
푡
+
푊
ℎ표
ℎ
푡
−
1
+
푏
표
)
(14)
ℎ
푡
=
표
푡
tanh(
푐
푡
)
,
(15)
the correspondence between between the gates in (10) and those in (13) is as follows:
1
−
푔
푡
corre-
sponds to
푓
푡
, and
푔
푡
to
푖
푡
.  To obtain a time range around
 
for unit
푖
, we must both ensure that
푓
푖
푡
lies around
1
−
1
/ 
, and that
푖
푡
lies around
1
/ 
. When facing time dependencies with largest time
range
 
max
, this suggests to initialize LSTM gate biases to
푏
푓
∼
log(
풰
([1
, 
max
−
1]))
푏
푖
=
−
푏
푓
(16)
with
풰
the uniform distribution and
 
max
the expected range of long-term dependencies to be cap-
tured.
Hereafter, we refer to this as the
chrono initialization
.
3
E
XPERIMENTS
First,  we test the theoretical arguments by explicitly introducing random time warpings in some
data, and comparing the robustness of gated and ungated architectures.
Next,  the  chrono  LSTM  initialization  is  tested  against  the  standard  initialization  on  a  variety  of
both synthetic and real world problems. It heavily outperforms standard LSTM initialization on all
synthetic tasks, and outperforms or competes with it on real world problems.
The synthetic tasks are taken from previous test suites for RNNs, specifically designed to test the
efficiency of learning when faced with long term dependencies (Hochreiter & Schmidhuber, 1997;
Le et al., 2015; Graves et al., 2014; Martens & Sutskever, 2011; Arjovsky et al., 2016).
In  addition  (Appendix  A),  we  test  the  chrono  initialization  on  next  character  prediction  on  the
Text8 (Mahoney, 2011) dataset, and on next word prediction on the Penn Treebank dataset (Mikolov
et al., 2012).  Single layer LSTMs with various layer sizes are used for all experiments, except for
the word level prediction, where we use the best model from (Zilly et al., 2016), a
10
layer deep
recurrent highway network (RHN).
Pure warpings and paddings.
To test the theoretical relationship between gating and robustness
to time warpings, various recurrent architectures are compared on a task where the only challenge
comes from warping.
The unwarped task is simple: remember the previous character of a random sequence of characters.
Without time warping or padding, this is an extremely easy task and all recurrent architectures are
successful.  The only difficulty will come from warping; this way, we explicitly test the robustness
of various architectures to time warping and nothing else.
7
When the characteristic timescales of the sequential data at hand are completetly unknown, a possibility is
to draw, for each gate, a random time range
 
according to some probability distribution on
N
with slow decay
(such as
P
(
 
=
푘
)
∝
1
푘
log (
푘
+1)
2
) and initialize biases to
log(
 
)
.
5
Published as a conference paper at ICLR 2018
0
0
.
5
1
1
.
5
2
10      20      30      40      50      60      70      80      90     100
Loss
after
3
ep o chs
Maximum
warping
RNN
leaky
RNN
gated
RNN
0
0
.
5
1
1
.
5
2
10      20      30      40      50      60      70      80      90     100
0
0
.
5
1
1
.
5
2
2
.
5
10      20      30      40      50      60      70      80      90     100
Loss
after
3
ep o chs
Maximum
warping
RNN
leaky
RNN
gated
RNN
0
0
.
5
1
1
.
5
2
2
.
5
10      20      30      40      50      60      70      80      90     100
0
0
.
2
0
.
4
0
.
6
0
.
8
1
1
.
2
1
.
4
10      20      30      40      50      60      70      80      90     100
Loss
after
3
ep o chs
Maximum
warping
RNN
leaky
RNN
gated
RNN
0
0
.
2
0
.
4
0
.
6
0
.
8
1
1
.
2
1
.
4
10      20      30      40      50      60      70      80      90     100
0
0
.
5
1
1
.
5
2
2
.
5
10      20      30      40      50      60      70      80      90     100
Loss
after
3
ep o chs
Maximum
warping
RNN
leaky
RNN
gated
RNN
0
0
.
5
1
1
.
5
2
2
.
5
10      20      30      40      50      60      70      80      90     100
Figure  1:  Performance  of  different  recurrent  architectures  on  warped  and  padded  sequences  se-
quences.   From top left to bottom right:  uniform time warping of length
maximum_warping
,
uniform padding of length
maximum_warping
, variable time warping and variable time padding,
from
1
to
maximum_warping
.  (For uniform padding/warpings, the leaky RNN and gated RNN
curves overlap, with loss
0
.) Lower is better.
Unwarped task example:
Input:
All human beings are born free and equal
Output:
All human beings are born free and equa
Uniform warping example (warping
×
4
):
Input:
AAAAllllllll    hhhhuuuummmmaaaannnn
Output:
AAAAllllllll    hhhhuuuummmmaaaa
Variable warping example (random warping
×
1
–
×
4
):
Input:
Allllll  hhhummmmaannn bbbbeeiiingssss
Output:
AAAlllll   huuuummaaan    bbeeeingggg
Figure 2: A task involving pure warping.
Uniformly time-warped tasks are produced by repeating each character
maximum_warping
times
both in the input and output sequence, for some fixed number
maximum_warping
.
Variably time-warped tasks are produced similarly, but each character is repeated a random number
of times uniformly drawn between
1
and
maximum_warping
.  The same warping is used for the
input and output sequence (so that the desired output is indeed a function of the input). This exactly
corresponds to transforming input
푥
(
푡
)
into
푥
(
푐
(
푡
))
with
푐
a random, piecewise affine time warping.
Fig. 2 gives an illustration.
For each value of
maximum_warping
, the train dataset consists of
50
,
000
length-
500
randomly
warped random sequences, with either uniform or variable time warpings.  The alphabet is of size
10
(including a dummy symbol). Contiguous characters are enforced to be different. After warping,
each sequence is truncated to length
500
. Test datasets of
10
,
000
sequences are generated similarily.
The criterion to be minimized is the cross entropy in predicting the next character of the output
sequence.
6
Published as a conference paper at ICLR 2018
0
0
.
01
0
.
02
0
.
03
0
.
04
0
.
05
0
.
06
0
.
07
10000      20000      30000      40000      50000      60000
NLL
Loss
Numb er
of
iterations
Chrono
750
Constant
1
Memoryless
0
0
.
005
0
.
01
0
.
015
10000     20000     30000     40000     50000     60000
NLL
Loss
Numb er
of
iterations
Chrono
3000
Constant
1
Memoryless
0
0
.
005
0
.
01
0
.
015
0
.
02
0
.
025
0
.
03
0
.
035
0
.
04
10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
NLL
Loss
Numb er
of
iterations
Chrono
500
Constant
1
0
0
.
005
0
.
01
0
.
015
0
.
02
0
.
025
0
.
03
0
.
035
0
.
04
10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
NLL
Loss
Numb er
of
iterations
Chrono
1000
Constant
1
Figure 3: Standard initialization (blue) vs. chrono initialization (red) on the copy and variable copy
task.   From  left to  right,  top to  bottom,  standard  copy
 
= 500
and
 
= 2000
,  variable copy
 
= 500
and
 
= 1000
.  Chrono initialization heavily outperforms standard initialization, except
for variable length copy with the smaller
 
where both perform well.
0
0
.
05
0
.
1
0
.
15
0
.
2
0
.
25
100    200    300    400    500    600    700    800    900    1000
MSE
Loss
Numb er
of
iterations
Constant
200
Constant
1
Memoryless
0
0
.
05
0
.
1
0
.
15
0
.
2
500   1000   1500   2000   2500   3000   3500   4000   4500   5000
MSE
Loss
Numb er
of
iterations
Chrono
750
Constant
1
Memoryless
Figure 4: Standard initialization (blue) vs. chrono initialization (red) on the adding task.  From left
to right,
 
= 200
, and
 
= 750
. Chrono initialization heavily outperforms standard initialization.
Note that each sample in the dataset uses a new random sequence from a fixed alphabet, and (for
variable warpings) a new random warping.
A similar, slightly more difficult task uses
padded
sequences instead of warped sequences, obtained
by padding each element in the input sequence with a fixed or variable number of
0
’s (in continuous-
time, this amounts to a time warping of a continuous-time input sequence that is nonzero at certain
points in time only). Each time the input is nonzero, the network has to output the previous nonzero
character seen.
We compare three recurrent architectures:  RNNs (Eq. (2),  a simple,  ungated recurrent network),
leaky RNNs (Eq. (5), where each unit has a constant learnable “gate” between
0
and
1
) and gated
RNNs, with one gate per unit, described by (10). All networks contain
64
recurrent units.
7
Published as a conference paper at ICLR 2018
The point of using gated RNNs (10) (“LSTM-lite” with tied input and forget gates), rather than full
LSTMs, is to explicitly test the relevance of the arguments in Section 1 for time warpings.  Indeed
these LSTM-lite already exhibit perfect robustness to warpings in these tasks.
RMSprop with an
훼
parameter of
0
.
9
and a batch size of
32
is used. For faster convergence, learning
rates are divided by
2
each time the evaluation loss has not decreased after
100
batches. All architec-
tures are trained for
3
full passes through the dataset, and their evaluation losses are compared. Each
setup is run
5
times, and mean, maximum and minimum results among the five trials are reported.
Results on the test set are summarized in Fig. 1.
Gated architectures significantly outperform RNNs as soon as moderate warping coefficients are
involved.   As expected from theory,  leaky RNNs perfectly solve uniform time warpings,  but fail
to achieve optimal behavior with variable warpings, to which they are not invariant.  Gated RNNs,
which are quasi invariant to general time warpings, achieve perfect performance in both setups for
all values of
maximum_warping
.
Synthetic tasks.
For synthetic tasks, optimization is performed using RMSprop (Tieleman & Hin-
ton, 2012) with a learning rate of
10
−
3
and a moving average parameter of
0
.
9
. No gradient clipping
is performed;  this results in a few short-lived spikes in the plots below, which do not affect final
performance.
C
OPY  TASKS
.
The copy task checks whether a model is able to remember information for arbi-
trarily long durations.  We use the setup from (Hochreiter & Schmidhuber, 1997; Arjovsky et al.,
2016), which we summarize here.  Consider an alphabet of
10
characters.  The ninth character is a
dummy character and the tenth character is a signal character. For a given
 
, input sequences consist
of
 
+ 20
characters.  The first
10
characters are drawn uniformly randomly from the first
8
letters
of the alphabet. These first characters are followed by
 
−
1
dummy characters, a
signal
character,
whose aim is to signal the network that it has to provide its outputs, and the last
10
characters are
dummy characters. The target sequence consists of
 
+ 10
dummy characters, followed by the first
10
characters of the input.  This dataset is thus about remembering an input sequence for exactly
 
timesteps. We also provide results for the
variable
copy task setup presented in (Henaff et al., 2016),
where the number of characters between the end of the sequence to copy and the signal character is
drawn at random between
1
and
 
.
The best that a memoryless model can do on the copy task is to predict at random from among
possible characters, yielding a loss of
10 log(8)
 
+20
(Arjovsky et al., 2016).
On those tasks we use LSTMs with
128
units.  For the standard initialization (baseline), the forget
gate biases are set to
1
.  For the new initialization, the forget gate and input gate biases are chosen
according to the chrono initialization (16), with
 
max
=
3
 
2
for the copy task, thus a bit larger than
input length, and
 
max
=
 
for the variable copy task. The results are provided in Figure 3.
Importantly,  our  LSTM  baseline  (with  standard  initialization)  already  performs  better  than  the
LSTM baseline of (Arjovsky et al., 2016), which did not outperform random prediction.  This is
presumably due to slightly larger network size, increased training time, and our using the bias ini-
tialization from (Gers & Schmidhuber, 2000).
On the copy task,  for all the selected
 
’s,  chrono initialization largely outperforms the standard
initialization.  Notably, it does not plateau at the memoryless optimum.  On the variable copy task,
chrono initialization is even with standard initialization for
 
= 500
, but largely outperforms it for
 
= 1000
.
A
DDING  TASK
.
The adding task also follows a setup from (Hochreiter & Schmidhuber, 1997;
Arjovsky et al., 2016).  Each training example consists of two input sequences of length
 
.  The
first one is a sequence of numbers drawn from
풰
([0
,
1])
, the second is a sequence containing zeros
everywhere,  except for two locations,  one in the first half and another in the second half of the
sequence.  The target is a single number,  which is the sum of the numbers contained in the first
sequence at the positions marked in the second sequence.
The best a memoryless model can do on this task is to predict the mean of
2
×풰
([0
,
1])
, namely
1
(Arjovsky et al., 2016). Such a model reaches a mean squared error of
0
.
167
.
8
Published as a conference paper at ICLR 2018
LSTMs with
128
hidden units are used.  The baseline (standard initialization) initializes the forget
biases to
1
.  The chrono initialization uses
 
max
=
 
.  Results are provided in Figure 4.  For all
 
’s, chrono initialization significantly speeds up learning.  Notably it converges
7
times faster for
 
= 750
.
C
ONCLUSION
The self loop feedback gating mechanism of recurrent networks has been derived from first princi-
ples via a postulate of invariance to time warpings. 
'''



model = keras.models.load_model('../data/models/keras/model3.h5')

doc_data = utils.simple_preprocess(title=title, abstract=abstract, text=text)


fulldata, ngrams = utils.create_features(doc_data, labels=False)



docvec = [(
            np.fromstring(instance['title_vec'].strip('[]'), sep=',') +
            np.fromstring(instance['abstract_vec'].strip('[]'), sep=',') + 
            np.fromstring(instance['text_vec'].strip('[]'), sep=',') 
        ) / 3
         for instance in doc_data]
docvec = np.array(docvec)

data = fulldata[:,:300] - docvec

predictions = model.predict(data)

df_ = np.array([ngrams.reshape((-1,)), 
    predictions.reshape((-1,)), 
    np.where(predictions>0.5, 1, 0).reshape((-1,))])



df = pd.DataFrame(df_.T, columns = ['ngram', 'probability','predicted label'])

print(df)

df.to_csv('../data/models/keras/results_model3_test.csv', sep=';')























