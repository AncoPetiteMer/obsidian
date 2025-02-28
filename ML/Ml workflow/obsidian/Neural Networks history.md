# The Deep Learning Odyssey: From AlexNet to Modern AI (2012–Present)
[[Neural Networks history]]

The evolution of neural networks over the past decade reads like an epic saga in technology. In 2012, a breakthrough CNN _conquered_ image recognition, sparking a deep learning revolution. From there, networks learned to remember sequences, generate art, and even challenge the dominance of convolution itself. In this chronological journey, we’ll meet the key architectures—each a hero of their era—detailed in both concept and code. _(We’ll use PyTorch for deep learning code and scikit-learn for a few comparisons.)_ Prepare for technical depth, analogies, a bit of humor, and plenty of insights as we travel from **AlexNet** to **transformers** and **diffusion models**.

## 2012: AlexNet – Deep Learning’s Big Bang


[_Figure](https://viso.ai/deep-learning/alexnet/): **AlexNet architecture** visualized. It consists of five convolutional layers (C1–C5, in orange) that extract features from the input image, followed by three fully-connected layers (FC6–FC8) that perform classification. AlexNet’s 8-layer deep network (60M+ parameters) crushed the 2012 ImageNet competition, halving the error rate of the next best model​_


_._

AlexNet, introduced by Alex Krizhevsky _et al._ in 2012, marks the start of the deep learning era in computer vision. Prior to AlexNet, neural networks were considered too slow or impractical for large-scale tasks. AlexNet proved otherwise by winning the ImageNet image classification challenge of 2012 by a **huge margin** (15.3% vs 26.2% top-5 error​

[viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=In%20that%20competition%2C%20AlexNet%20performed,to%2015.3)

). It demonstrated that **depth matters** – its 5 convolutional + 3 dense layer architecture learned rich hierarchical features from images. Key innovations from the AlexNet paper set the template for future CNNs​

[viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=,just%20like%20regular%20max%20pooling)

:

- **ReLU activation:** Replaced sigmoid/tanh activations with the non-saturating ReLU ($f(x)=\max(0,x)$), drastically accelerating training convergence​
    
    [viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=three%20fully%20connected%20layers.%20,just%20like%20regular%20max%20pooling)
    
    . ReLU doesn’t squish large values to a narrow range, so it preserves gradients well (plus it’s computationally simple – just threshold at 0). _Analogy:_ ReLU is like a diode, only allowing positive signal through – it gives neurons an easy on/off switch and avoids the “yes, but softly” behavior of sigmoids.
- **GPU Training:** Leveraged GPUs (NVIDIA CUDA) to train the network in reasonable time. This was like switching from a bicycle to a rocket – GPUs enabled crunching millions of operations across many cores in parallel.
- **Dropout Regularization:** Used dropout (randomly turning off neurons during training) to prevent overfitting, like forcing an ensemble of many subnetworks to jointly learn robust features.
- **Data Augmentation & LRN:** Employed image augmentations (translations, reflections) to expand effective training data, and Local Response Normalization (now less common) to help generalization.

In code, we can build a simplified AlexNet using PyTorch’s layer modules. Below is a **PyTorch model definition** for a variant of AlexNet (for simplicity, we use fewer channels than the original and omit LRN):
```python
import torch
import torch.nn as nn

class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),          # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),  # flatten conv5 output
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x

model = SimpleAlexNet(num_classes=1000)
print(model)

```



This code defines a CNN with the structure of AlexNet’s feature extractor and classifier. Key details: large initial conv filter (11×11) with stride 4 (to aggressively downsample), multiple conv layers to deepen the network, and two dropout layers in the classifier for regularization. Running `print(model)` would show the architecture stack, verifying the design.

AlexNet’s success showed that **deep convolutional networks + big data + GPUs** are a winning combination. It set off a wave of research extending CNNs (VGG, Inception, etc.), but our story now branches to another lineage of neural nets that handle **sequential data**.

## Recurrent Neural Networks ([[RNN]]s) – Teaching Networks to Remember

After CNNs conquered images, researchers turned to sequences: language, speech, time series. Enter **Recurrent Neural Networks (RNNs)** – networks with loops, designed to handle sequential input of arbitrary length. An RNN processes one element of a sequence at a time, while maintaining a **hidden state** (memory) that carries information from earlier time steps to later ones. This way, RNNs can, in theory, _remember_ prior inputs in the sequence.

**How RNNs work:** Think of an RNN as a recurring cell that reads inputs one by one. At each time step $t$, the cell takes the current input $x_t$ and the previous hidden state $h_{t-1}$ and produces a new hidden state $h_t = f(W \cdot [h_{t-1}, x_t])$ (and possibly an output $y_t$). All time steps share the same network weights $W$. The effect is like having a deep neural network **unrolled in time** – one layer per time step, all with tied weights.

![RNN architecture](https://kvitajakub.github.io/img/rnn-unrolled.svg)

_Figure: **Unrolled RNN** over a sequence. On the left, the RNN is shown as a loop (green box A feeding into itself). On the right, the loop is unrolled into a chain: each copy of cell A processes one input $x_i$ and produces a hidden state $h_i$【9†】. The same weights apply at each step. Backpropagation through time (BPTT) will propagate gradients from the end of the sequence back through each copy (dashed arrows), adjusting the weights._

RNNs are useful whenever context or order matters. For example, understanding a sentence requires remembering earlier words when interpreting later ones. A plain feedforward network can’t naturally do this, but an RNN can maintain a running context in its hidden state. In the early 2010s, RNNs re-emerged (they’ve existed since the ’80s) as powerful models for language modeling, speech recognition, handwriting generation, and more.

However, training RNNs is notoriously tricky due to the **vanishing gradient problem**. This is the same issue early deep nets faced: as we backpropagate errors through many time steps (layers), gradients can get exponentially smaller, effectively preventing learning of long-range dependencies​

[viso.ai](https://viso.ai/deep-learning/alexnet/#:~:text=,during%20backpropagation%20or%20disappear%20completely)

. In plain English, RNNs had the memory of a goldfish – they struggled to carry information from far in the past to influence the present.

Let’s illustrate a simple RNN in code. We’ll use PyTorch’s `nn.RNN` module for a basic example of sequence processing. Say we want an RNN to read a sequence of numbers and produce an output sequence (e.g., identity mapping for demonstration):

```python
import torch.nn as nn

# Define an RNN layer: input_size=1, hidden_size=5, one RNN layer
rnn_layer = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)

# Dummy sequence batch: batch_size=2, sequence_length=4, features=1
inputs = torch.tensor([[[1.0],[2.0],[3.0],[4.0]],
                       [[5.0],[6.0],[7.0],[8.0]]])  # shape (2,4,1)
hidden0 = torch.zeros(1, 2, 5)  # initial hidden (num_layers, batch, hidden_size)

outputs, hn = rnn_layer(inputs, hidden0)
print("Output sequence shape:", outputs.shape)
print("Final hidden state shape:", hn.shape)

```

Here we define an RNN that takes 1-dimensional inputs and has a hidden state of size 5. We feed in a batch of two sequences (each of length 4). The RNN returns `outputs` (the sequence of hidden states at each time step, shape `[batch, seq_len, hidden_size]`) and `hn` (the final hidden state for each layer). Running this would show `outputs.shape = (2,4,5)` and `hn.shape = (1,2,5)`, confirming it processed 4 steps and ended with a 5-dimensional hidden state for each sequence in the batch. In practice, we’d add a linear layer on top of the RNN to map the hidden state to desired output (e.g., a vocabulary distribution in language modeling).

**Backpropagation Through Time (BPTT):** Training RNNs uses BPTT, which is like running standard backprop on the unrolled network. It’s as if the error signal hops in a time machine, traveling backwards through every time step. If gradients shrink at each step, by the time they reach the beginning, they’re minuscule. This is why vanilla RNNs have trouble with long sequences (e.g., trying to learn dependencies 100 steps back is almost hopeless with naive RNN).

_Humor analogy:_ BPTT is like “The Terminator” of machine learning – sending a signal back in time to fix mistakes in the past. But if the signal gets weaker each trip (like a terminator gradually disintegrating), it can’t change anything far back. We needed a better time traveler…

And that hero arrived in the form of LSTMs.

## Long Short-Term Memory (LSTM) – A Network with a Time Machine

To overcome RNN limitations, researchers Hochreiter & Schmidhuber (1997) introduced the **Long Short-Term Memory (LSTM)** network. The name sounds like an oxymoron, but it reflects the goal: retain long-term information over short-term intervals. LSTMs are a special kind of RNN that can learn **long-range dependencies** thanks to a clever gating mechanism that controls information flow.

**LSTM Architecture:** An LSTM cell is like an RNN on steroids. Instead of a single hidden state, it maintains two: a **cell state** $C_t$ (the “long-term memory”) and a hidden state $h_t$ (the “short-term state” or output). The cell has gates that regulate what information to forget from the cell state, what new information to add, and what to output. Specifically, each LSTM cell has:

- **Forget gate** $f_t$: decides which parts of the cell state to erase (outputs values between 0 and 1 for each component of $C_{t-1}$).
- **Input (update) gate** $i_t$ and **candidate** $\tilde{C}_t$: together decide what new information to write to the cell state.
- **Output gate** $o_t$: decides what part of the cell state to output as $h_t$.

At time $t$, the LSTM computes roughly:

f_t = \sigma(W_f \cdot [h_{t-1}, x_t]) # forget gate i_t = \sigma(W_i \cdot [h_{t-1}, x_t]) # input gate \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t]) # candidate cell state C_t = f_t * C_{t-1} + i_t * \tilde{C}_t # new cell state (keep some of old, add new) o_t = \sigma(W_o \cdot [h_{t-1}, x_t]) # output gate h_t = o_t * \tanh(C_t) # new hidden state (filtered cell state)

Here $\sigma$ is the sigmoid function (outputs 0 to 1, perfect for gating), and $\tanh$ squashes inputs between -1 and 1 (to create candidate values). The `*` denotes elementwise multiplication. The LSTM gates allow it to **retain information over long periods**: for example, if $f_t \approx 1$ and $i_t \approx 0$ for many steps, the cell state $C_t$ basically carries on unchanged – remembering whatever was written in it initially. This ability to have nearly constant error flow (via $C_t$) mitigates the vanishing gradient problem. It’s like an **information highway** through time where gradients can flow easily (the highway being the cell state, with forget gate near 1.0 allowing gradient to just pass through).

![LSTM architecture](https://kvitajakub.github.io/img/lstm-peepholes.svg)

_Figure: **LSTM cell diagram** (with a variant including peephole connections). The cell state runs through the top (line across the cell), being adjusted by gates: the forget gate $f_t$ (left) multiplies the old cell state $C_{t-1}$ (line coming from top left) by some factor in [0,1]; the input gate $i_t$ (middle left) decides how much of the new candidate $\tilde{C}_t$ (computed via tanh) to add in; the output gate $o_t$ (right) controls how much of the cell state (passed through tanh) to reveal as $h_t$. Sigmoid ($\sigma$) activations (yellow) produce gate values, and tanh (purple) produces candidate or scaled outputs. The multiplicative interactions (pink ⨉ and +) allow gating【10†】._

_Analogy:_ Think of the LSTM as a smart concierge managing a busy hotel (the cell state is the notepad the concierge keeps). The forget gate decides what old info in the notepad to erase (e.g., cross out yesterday’s news), the input gate decides what new info to write down (e.g., today’s reservations), and the output gate decides what info to share at a given moment (e.g., tell a guest their schedule, which is a filtered piece of the notepad). This way, the concierge (LSTM) maintains relevant info over time and provides it when needed, avoiding being overwhelmed by irrelevant details from the distant past.

In practical terms, LSTMs enabled RNNs to learn things like long sentences, or dependencies in text that span dozens of words. Around 2015–2016, LSTMs became _the_ go-to model for sequence learning. They powered breakthroughs in machine translation (Google’s 2016 Neural Machine Translation system was built on LSTMs), speech recognition (e.g., Siri’s speech models), and more.

To use LSTMs in code is straightforward with PyTorch’s `nn.LSTM`. Here’s an example similar to our vanilla RNN, but using an LSTM:

lstm_layer = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Dummy input: batch of 3 sequences, each of length 5, feature dim 10

```python
inputs = torch.randn(3, 5, 10)   # (batch, seq_len, input_size)
h0 = torch.zeros(1, 3, 20)       # initial hidden state (num_layers, batch, hidden_size)
c0 = torch.zeros(1, 3, 20)       # initial cell state

outputs, (hn, cn) = lstm_layer(inputs, (h0, c0))
print("LSTM output shape:", outputs.shape)
print("Final hidden shape:", hn.shape, "| Final cell shape:", cn.shape)
```

This defines a single-layer LSTM taking 10-dimensional inputs and producing a 20-dimensional hidden state. The input is a random tensor of shape (3,5,10). We provide initial hidden and cell states (both zeros). The output `outputs` will have shape (3,5,20) (hidden state for each of 5 timesteps, for each sequence), while `hn` and `cn` are the final hidden and cell states (each of shape (1,3,20) for 1 layer, 3 sequences, 20 dims). Running this confirms the shapes and that the code executes.

**Real-world uses:** LSTMs (and their kin) have been used in **text generation** (e.g., char-RNNs that generate Shakespeare-like text), **language modeling**, **machine translation** (an encoder LSTM reads a sentence in one language, a decoder LSTM writes in another), **speech recognition**, **music generation** – basically anywhere sequence modeling was needed, LSTMs were dominant through the late 2010s.

However, LSTMs aren’t the only enhanced RNN in town. A slightly leaner variant arrived around 2014: the GRU.

## Gated Recurrent Units ([[GRU Layer]]s) – LSTM’s Leaner Cousin

The **Gated Recurrent Unit (GRU)**, introduced by Cho _et al._ in 2014, is essentially a simplified LSTM. Think of GRU as LSTM’s cousin who decided to “travel light” by packing fewer gates. GRUs have only two gates: a **reset gate** $r_t$ and an **update gate** $z_t$ (sometimes called the forget gate, as it plays a similar role). It combines the input and forget gates of the LSTM into one “update” gate, and it merges the cell state and hidden state into a single state $h_t$.

In a GRU cell, the equations look like:

z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) # update gate (mixes input/forget) r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) # reset gate \tilde{h}_t = \tanh(W_h \cdot [r_t * h_{t-1},\; x_t]) # candidate state (reset applied) h_t = z_t * h_{t-1} + (1 - z_t) * \tilde{h}_t # new state is mix of old state and new candidate

The update gate $z_t$ decides how much of the previous state to keep (if $z_t$ is 1 for a unit, it keeps the old state, no change; if 0, it completely updates to the new candidate). The reset gate $r_t$ controls how much of the past to forget when computing the new candidate $\tilde{h}_t$ (if $r_t$ is 0, we ignore the old hidden state, essentially resetting memory; if 1, we fully use the previous state).

![GRU architecture](https://kvitajakub.github.io/img/gru.svg)

_Figure: **GRU cell**. It has a simplified design with only two gates: reset gate $r_t$ (left) and update gate $z_t$ (middle). The GRU directly produces a new hidden state $\tilde{h}_t$ via a candidate (right tanh node) that uses a reset-modulated previous state. The final $h_t$ is a linear interpolation between the old state $h_{t-1}$ and the new candidate, controlled by $z_t$ (note the $(1 - z_t)$ flowing into the pink + node)【11†】._

The GRU’s design makes it computationally slightly cheaper (fewer matrix multiplications) and conceptually simpler (no separate cell state $C_t$). Empirically, GRUs often perform on par with LSTMs on many tasks, though there’s no universal winner – some tasks prefer one or the other.

In code, using a GRU in PyTorch is as easy as an LSTM, just use `nn.GRU`:

```python
gru_layer = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
outputs, hn = gru_layer(inputs, torch.zeros(1, 3, 20))
print("GRU output shape:", outputs.shape)
print("GRU final hidden shape:", hn.shape)

```

This would output shapes similar to the LSTM example (except there’s no cell state `cn` now).

**Summary of RNN variants:** RNN, LSTM, GRU can be viewed like evolutionary stages:

- **Vanilla RNN**: simple, but prone to short memory.
- **LSTM**: complex gating, long memory, more parameters.
- **GRU**: middle ground – gating for long memory, but fewer parameters than LSTM.

Each was a milestone in enabling networks to handle sequences better. By the late 2010s, however, a completely new approach would overtake even LSTMs for sequence tasks (hint: _“Attention”_). But before we get there, our journey detours into networks that learn without labels and those that _create_ data.

## Autoencoders – Learning Data Compressions and Encodings

As deep learning matured, researchers explored unsupervised learning – getting neural nets to learn useful representations without explicit labels. **Autoencoders (AE)** became a popular tool for this. An autoencoder is a neural network trained to copy its input to its output. That sounds trivial, but the catch is it does so through a **bottleneck** – a middle layer (latent code) smaller than the input. Thus, the network is forced to **compress** the data into a lower-dimensional representation (the code), then decompress it back. Through this process, it hopefully learns meaningful structure about the data.

![](blob:https://chatgpt.com/b5fa03b3-72a6-4fb5-9206-b00bda5ef9ef)

_Figure: **Autoencoder architecture.** The network consists of two parts: an **encoder** that compresses the input $X$ into a latent code $h$ (also called _bottleneck_ or _latent vector_), and a **decoder** that reconstructs $X'$ from $h$. The encoder and decoder are often mirror-symmetric neural networks. By training $X' \approx X$ (minimizing reconstruction error), the autoencoder learns an efficient encoding of the data​_

_[Autoencoders architecture](https://en.wikipedia.org/wiki/Autoencoder#/media/File:Autoencoder_schema.png)_

_._

Think of an autoencoder like a student tasked with taking detailed lecture notes (the input), summarizing them onto an index card (the latent code), and then trying to recreate the original lecture from that summary. If the student can reconstruct the lecture well, it means the summary captured the important points.

The objective function for training an autoencoder is typically to minimize the reconstruction error (e.g., mean squared error for images, or cross-entropy for binary data). Once trained, you usually **discard the decoder** and use the encoder’s output $h$ as a compressed representation (feature vector) for your data.

**Applications:**

- **Dimensionality reduction:** Autoencoders can serve as non-linear versions of PCA. The latent code can be used as a low-dimensional representation of high-D data.
- **Denoising autoencoders:** By training the autoencoder to reconstruct the original input from a **noisy** version of it, the model learns to filter noise. This is useful for noise reduction in images or signals.
- **Anomaly detection:** Train an autoencoder on “normal” data; it will learn to reconstruct normal patterns well. If you feed it anomalous data (e.g., a network intrusion pattern or a manufacturing defect image), it will reconstruct poorly (higher error). Thus, reconstruction error can flag anomalies​
    
    [en.wikipedia.org](https://en.wikipedia.org/wiki/Autoencoder#:~:text=,AutoML)
    
    .
- **Generative modeling (Variational Autoencoders):** An important extension is the VAE (Variational Autoencoder), which adds probabilistic constraints to the latent space so we can sample from it to generate new data. While VAEs are beyond our current scope, they were among the first _generative_ deep models alongside GANs.

Implementing a basic autoencoder in PyTorch is straightforward. Let’s build a simple autoencoder for the MNIST digits (28x28 images compressed to a small code):

```python
import torch.optim as optim

# Define a simple Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: two linear layers
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32)   # compress to 32 dims
        )
        # Decoder: two linear layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()        # output pixels 0-1
        )
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# Initialize model, loss, optimizer
ae = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=1e-3)

# Dummy training loop for illustration (assuming mnist_loader provides 28x28 flattened images):
# for images, _ in mnist_loader:
#     images = images.view(images.size(0), -1)  # flatten
#     optimizer.zero_grad()
#     recon = ae(images)
#     loss = criterion(recon, images)  # compare recon to original
#     loss.backward()
#     optimizer.step()

```

This defines a simple fully-connected autoencoder. The encoder compresses 784-dimensional input to 32, and the decoder tries to reconstruct the 784-dim output. We use a Sigmoid at the output since MNIST pixels are normalized 0–1, and MSE loss to measure reconstruction. (In practice, one might use `BCEWithLogitsLoss` treating it as a probabilistic reconstruction, but MSE is fine for a demo.)

After training, we could examine the 32-D latent vectors for properties, use them for clustering digits, or visualize them with PCA/TSNE to see if the autoencoder learned to differentiate digits in code space.

Autoencoders were a stepping stone to more explicit generative models. Speaking of which, let’s talk about _generating_ data: GANs.

## Generative Adversarial Networks (GANs) – When Neural Nets Play Cat and Mouse

One of the most exciting (and fun) developments in deep learning was **Generative Adversarial Networks (GANs)**, introduced by Ian Goodfellow _et al._ in 2014. GANs brought forth a way to train neural networks to generate _new data_ (images, audio, etc.) that mimics real data. The core idea of a GAN is brilliantly game-like: pit two neural networks against each other in a competition.

A GAN consists of two players:

- The **Generator** ($G$): takes random noise (a random vector) and tries to produce a fake data sample (e.g., an image) that looks real.
- The **Discriminator** ($D$): takes an input (real or generated) and predicts whether it’s real (from the training data) or fake (produced by $G$).

They are trained simultaneously: $D$ tries to become a savvy detective distinguishing real vs fake, and $G$ tries to become a skilled counterfeiter that fools $D$. Formally, $G$ wants to maximize $D(G(z))$ (where $z$ is random noise), while $D$ wants to maximize correctly classifying real vs fake (typically, maximize $\log D(x_{\text{real}}) + \log(1 - D(G(z)))$). This setup is a **minimax game**: $G$ minimizes the same objective $D$ is trying to maximize.

![https://viso.ai/deep-learning/generative-adversarial-networks-gan/](blob:https://chatgpt.com/1ea52bdf-8f83-49ab-9f05-c41e7704742d)

_Figure: **Conceptual illustration of a GAN as a two-player game.** Imagine a tennis match between a forger (Generator) and an inspector (Discriminator). The generator (left player) serves by producing a fake example; the discriminator (right player) tries to return it by identifying it as fake. Over many rounds, the generator learns to serve more convincingly (more realistic fakes), and the discriminator becomes a sharper judge. Eventually, an equilibrium may be reached where the fakes are so good the discriminator is only guessing​_

_[viso.ai](https://viso.ai/deep-learning/generative-adversarial-networks-gan/#:~:text=In%20the%20context%20of%20GANs%2C,generated%20data%20and%20real%20data)_

_._

In training:

- We alternate between training $D$ and training $G$.
- To train $D$: we show it a batch of real data (labelled "real") and a batch of generated data from $G$ (labelled "fake"), and update $D$ to better discriminate.
- To train $G$: we generate a batch of fakes, run them through $D$, compute the loss from the perspective of “the generator wants these to be classified as real,” and update $G$ accordingly (while _not_ updating $D$ in this step).

Over time, $G$ (if successful) learns to model the distribution of real data, producing very realistic samples. $D$ tries to keep up, like an arms race. Many liken this to counterfeiters vs the police: as counterfeiters improve their fake currency, the police improve their detection skills, and so on.

Here’s a **simplified GAN training loop** in PyTorch-like pseudocode, for generating MNIST-digit-like images:

```python
import torch.nn.functional as F

G = Generator()     # e.g., map 100-d noise to 28x28 image
D = Discriminator() # e.g., classify 28x28 image as real/fake
optim_G = optim.Adam(G.parameters(), lr=2e-4)
optim_D = optim.Adam(D.parameters(), lr=2e-4)

for real_imgs, _ in dataloader:  # iterate through real data
    # Train Discriminator
    optim_D.zero_grad()
    z = torch.randn(real_imgs.size(0), 100)           # sample noise
    fake_imgs = G(z).detach()                         # generate fake images (detach so G is not trained here)
    loss_D = - (torch.log(D(real_imgs)).mean() + torch.log(1 - D(fake_imgs)).mean())
    loss_D.backward()
    optim_D.step()

    # Train Generator
    optim_G.zero_grad()
    z = torch.randn(real_imgs.size(0), 100)
    fake_imgs = G(z)                                  # new fake images
    loss_G = - torch.log(D(fake_imgs)).mean()         # generator wants D(fake)=1 (real)
    loss_G.backward()
    optim_G.step()

```

This code sketch uses the original GAN loss. In practice, one might use the non-saturating loss version or even different formulations like Wasserstein GAN, but the essence is: update $D$ to maximize log-likelihood of real vs fake classification; update $G$ to fool $D$. Over many epochs, if balanced well, $G$ starts producing outputs that $D$ cannot distinguish from real.

**Training issues:** GAN training is famously fickle. It requires a delicate balance—if $D$ becomes too good, $G$ gets no meaningful gradient (it always gets caught); if $G$ is too good, $D$ yields wrong labels trivially. Problems like **mode collapse** can happen (where $G$ finds one kind of output that consistently fools $D$ and produces only that, e.g., generating the same face over and over). Many techniques (architectural choices like DCGAN’s conv nets, loss tweaks, feature matching, etc.) were developed to stabilize GAN training.

**Results:** When GANs work, the results can be astounding. GANs have generated:

- Photorealistic images of people who don’t exist (see _thispersondoesnotexist.com_).
- Artwork in various styles.
- Super-resolution images (enhancing details in low-res images).
- Deepfakes (face-swapping in videos).
- Synthetic data for training or simulation (e.g., generating medical images for data augmentation).

GANs essentially gave neural networks a creative toolset. One network’s “imagination” is honed by another’s “critique.” This adversarial training was a game-changer for generative modeling.

GANs dominated the generative scene for a few years, but as we’ll see, newer generative models (like diffusion models) are now stealing the limelight. Before that, we must talk about a fascinating idea by Hinton to rethink CNNs: capsule networks.

## Capsule Networks – Hinton’s Quest to Fix CNNs

Geoffrey Hinton (one of the “godfathers” of deep learning, and incidentally Krizhevsky’s advisor on AlexNet) wasn’t entirely satisfied with CNNs, despite their success. In 2017, Hinton, Sabour, and Frosst introduced **Capsule Networks (CapsNets)** as an attempt to address some weaknesses of CNNs. The core intuition is that a typical CNN loses a lot of **spatial hierarchies** information due to pooling. For example, a CNN might detect facial features (eyes, nose, mouth) but if you jumbled them up in an image (like a Picasso painting with features in wrong places), a CNN might still fire the same “face” neurons. In other words, CNNs are not great at understanding _relative spatial relationships_ – they just know features are present, not whether they’re arranged properly.

Capsule Networks introduce **capsules**, which are groups of neurons that output not just an activation but a vector (or matrix) representing both the presence of a feature _and_ properties of that feature like pose (orientation, position, etc.). The idea is that a “capsule” for an object (say a face) will have high activation if the face is present, and its vector output will encode the pose of that face. Capsules are organized in layers, and **dynamic routing** algorithms decide which lower-level capsules activate higher-level ones.

In a CapsNet (as per the 2017 paper):

- First, convolutional layers create lower-level features.
- Instead of flattening into one huge vector, features are grouped into capsules (e.g., 8-dimensional vectors). A squash non-linearity ensures these vectors have length between 0 and 1, interpreting length as probability of feature presence.
- Then, each capsule in layer **L** tries to predict outputs for capsules in layer **L+1**. If the predictions agree, the higher-level capsule is activated (routing by agreement).
- This routing mechanism replaces max-pooling, aiming to preserve the spatial relationships.

_Analogy:_ Think of each capsule as a little expert specialist. One capsule might detect an eye in an image and encode its position and angle. Another might detect a nose. A higher-level “face” capsule will receive “proposals” from the eye and nose capsules, including their poses. If the eyes and nose agree on a plausible face arrangement, the face capsule activates strongly (i.e., “yes, I see a face, and it’s looking 30° to the left”). If they are in odd positions (one eye far away from where a nose is), the face capsule won’t activate strongly, thus addressing the jumbled-feature issue (the “Picasso problem” where all parts present but in wrong places​

[en.wikipedia.org](https://en.wikipedia.org/wiki/Capsule_neural_network#:~:text=Among%20other%20benefits%2C%20capsnets%20address,capsnets%20exploit%20the%20fact%20that)

).

In practice, the original CapsNet was demonstrated on MNIST digits and showed that it could recognize digits even when parts (strokes) were shifted – something a regular CNN could also do with data augmentation, but CapsNet did with an internal routing mechanism. CapsNet also output **reconstructions** of input images as a form of regularization and to ensure capsules encoded instantiation parameters.

Capsule networks are quite complex to implement from scratch. The dynamic routing involves iterative processes (looping a few times to refine agreements). Here is a very high-level sketch of what one iteration might look like (not full code):

```python
# u_hat: predictions from lower capsules (shape: [num_lower, num_higher, capsule_dim])
# b: raw coupling logits (shape: [num_lower, num_higher], initialized to zero)
# v: outputs of higher capsules (shape: [num_higher, capsule_dim])

for iter in range(num_iterations):
    c = softmax(b, dim=1)            # coupling coefficients for each lower->higher
    s = (c[:, :, None] * u_hat).sum(dim=0)  # weighted sum of predictions for each higher capsule
    v = squash(s)                    # squash to get output of higher capsules
    # measure agreement
    delta_b = (u_hat * v[None, :, :]).sum(dim=2)  # dot product agreement
    b = b + delta_b                  # update coupling logits

```

This pseudocode captures the essence: lower-level capsules make predictions `u_hat` for higher-level ones (via learned transformation matrices). The softmax `c` decides which higher capsule each lower one mostly contributes to. The higher capsule’s tentative state `v` is computed and then used to adjust the couplings `b` (if a lower capsule’s prediction aligns with `v`, `b` increases for that connection). After a few iterations, you get the final `v` as the output of the higher capsule.

Capsule networks are fascinating, but they have not (yet) seen widespread adoption in practice. They are computationally heavy for larger images and datasets, and so far, CNNs with lots of data have kept their edge. Still, Hinton’s capsules inspired people to think about alternative ways of modeling spatial hierarchies, and the idea of neurons with vector outputs representing entity attributes is intriguing for the quest toward more human-like vision understanding.

As our story continues, we return to the convolutional network lineage for one of the **most influential inventions** that made ultra-deep networks feasible: the **Residual Network**.

## ResNets (2015) – Going Deeper with Skip Connections

By 2015, CNNs were dominating vision benchmarks. Every year, researchers increased depth (VGG went to 19 layers, GoogLeNet ~22 layers with Inception modules). But simply stacking more layers was starting to hit diminishing returns and training difficulties. Common intuition was: deeper should be better, but experiments showed very deep networks sometimes performed worse than shallower ones (the infamous “degradation problem”—not to be confused with overfitting, this was on training accuracy!). Something was amiss in training deep nets: the optimizer struggled to propagate gradients through so many layers, even with ReLUs and good init.

**Residual Networks (ResNets)**, introduced by He _et al._ in 2015, solved this by a simple but powerful idea: **skip connections**. A ResNet is built out of “residual blocks” where the input of a block is added to its output (after some convolutional layers). In formula terms, instead of a block trying to learn a direct mapping $H(x)$, it learns a _residual_ mapping $F(x) = H(x) - x$. The block then outputs $y = F(x) + x$. If $H(x)$ was the identity function, it can be easily represented by $F(x)=0$ (just zero out the weights) and $y = x$. This simple reformulation makes it easier for the network to _at least_ learn identity mappings for some layers, which helps avoid the degradation problem.

![https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/](blob:https://chatgpt.com/dfc84efc-4c4b-4052-839b-3bffeea42613)

_Figure: **Residual block** (building unit of ResNet). Instead of learning an unreferenced function $H(x)$, the residual block learns $F(x)$ (the “residual”) and adds the original input $x$ to it. Mathematically: output = $F(x) + x$. If $F(x)$ is zero, the block just outputs $x$ (identity passthrough). This helps preserve gradients and information as it flows through many layers._

The skip connection (also called shortcut connection) allows gradients to flow directly through the skip path (essentially backpropagating through an identity function has no attenuation of gradient). It was like giving the network **highways** to transmit information. Even if some layers in the block are not useful, they can be bypassed. This addressed the vanishing gradient issue for very deep nets and allowed He _et al._ to train **152-layer networks** (!) that significantly outperformed shallower ones on ImageNet.

In code, a residual block can be written in PyTorch as follows (for example, a basic block used in ResNet-18/34):

```python
class BasicBlock(nn.Module):
    expansion = 1  # for compatibility with deeper Bottleneck blocks
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        # If dimensions differ, use a conv to project input to correct shape for addition
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        # main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # add shortcut
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

```

This implements a 2-layer residual block. If the input and output channels (or spatial size due to stride) differ, we use a `downsample` layer (1x1 conv) to reshape the identity to the correct dimensions before adding. Otherwise, the identity is added directly. Note the final `out += identity` and then ReLU. This addition is the magic – it lets the network propagate $x$ or the gradient of $x$ easily. A ResNet is built by stacking many of these blocks (with occasional downsampling via stride 2 blocks to reduce spatial dimension).

Residual connections were quickly adopted not just in image classification but across deep learning:

- **Computer Vision:** ResNets became the backbone for many tasks (detection, segmentation). Variants like ResNeXt, Wide ResNet, etc., tweaked the formula but kept skip connections.
- **Language/NLP:** While LSTMs were king for sequences, people even built deep RNNs with residual connections between time steps or layers, to ease training.
- **Highway Networks:** An earlier idea (Srivastava _et al._) introduced gating on skip connections (inspired by LSTMs), but ResNet’s simpler ungated skip performed better in practice​
    
    [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/#:~:text=Skip%20connections%20were%20introduced%20in,gates%20%E2%80%93%20simplicity%20wins%20here)
    
    – simplicity won out.
- **Normalization and skips**: Later architectures combined skip connections with normalization (Layer Norm) and attention… which brings us to the next revolution.

ResNets essentially allowed neural networks to go _ultra-deep_. But our story doesn’t end with stacking conv layers. In 2017, a paper came out that would redefine how we handle sequences (and even images) entirely, by throwing out recurrence and convolutions in favor of a new paradigm: **Attention**.

## 2017: [[Transformers]] – “Attention Is All You Need”

In 2017, Vaswani _et al._ published “Attention Is All You Need”​

[jalammar.github.io](https://jalammar.github.io/illustrated-transformer/#:~:text=The%20Transformer%20was%20proposed%20in,knowledge%20of%20the%20subject%20matter)

, introducing the **Transformer** architecture. This model was initially proposed for machine translation, but it has since taken over the field of natural language processing (and made inroads into many other domains). The Transformer’s tagline was bold: dispense with recurrence and convolution entirely, and rely solely on _attention mechanisms_ to handle sequences. It turned out to be a brilliant success, enabling both faster training (parallelizable) and better performance on long sequences than RNNs.

**Key idea – Self-Attention:** Instead of processing words sequentially as RNNs do, the Transformer looks at a sequence **holistically**. It uses _self-attention_ to allow each position in the sequence to directly attend to (i.e., consult) all other positions. This means, for example, when encoding a word in a sentence, the model can immediately draw information from words even 10 positions away, without having to pass through intermediate steps. It’s like every word has a direct line of communication to every other word, mediated by an attention mechanism that decides what to pay attention to.

**Transformer Architecture Overview:** The original transformer is an **encoder-decoder** model:

- The **encoder** takes an input sequence (e.g., a sentence in English) and outputs a series of vector representations (one per input token).
- The **decoder** takes those representations and generates an output sequence (e.g., the translation in French), one token at a time, while attending to the encoder outputs.

Each encoder and decoder is built from **stacked layers** (e.g., 6 each in the base model). A layer primarily has:

1. **Multi-Head Self-Attention:** The layer allows the model to attend to itself (or for decoder, attend to both its own output and encoder outputs). Multi-head means the attention is done independently multiple times (with different learned projections), so the model can attend to different aspects of the sequence in parallel.
2. **Feed-Forward Network:** A position-wise dense network (applied to each sequence position separately) that processes the attention outputs further.
3. **Add & Norm:** Residual connections (yes, ResNet strikes again) around the sub-layers and layer normalization for stable training.

Crucially, because the model doesn’t inherently know positions (unlike an RNN that processes in order), Transformers add **positional encodings** to the inputs – a deterministic or learned vector that indicates the token’s position in the sequence, so order information isn’t lost when you allow all-to-all attention.

_Analogy:_ The Transformer is like a group discussion (attention) rather than a line of people passing a message along (RNN). In an RNN, person 1 whispers to person 2, who whispers to 3, etc., which is slow and information can degrade. In a Transformer, every person in the group can listen to everyone else at once, and they do this in multiple “heads” – think of it like each person having multiple ears tuned to different conversations. After the discussion (attention), each person (token representation) updates their own understanding (feed-forward network). Residual connections ensure they can keep some of their original thought if needed.

The formula for one head of scaled dot-product attention is:

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{Q K^T}{\sqrt{d_k}}\Big) V Attention(Q,K,V)=softmax(dk​​QKT​)V

where $Q$ (query), $K$ (key), $V$ (value) are matrices representing the set of queries, keys, and values (often all derived from the sequence itself in self-attention). The dot products $Q K^T$ compute similarity between each query and each key (i.e., how much should token _i_ attend to token _j_), scaled by $\sqrt{d_k}$ to stabilize gradients. The softmax turns those similarities into attention weights that sum to 1 over all positions. Multiplying by $V$ sums up the values weighted by these attention weights, producing a weighted combination of values for each query. In self-attention, $Q, K, V$ are usually the same (just the embedding matrix of the sequence, projected).

**Multi-head** means we do this $h$ times with different linear projections of the inputs (so each head can focus on different patterns), then concatenate the results.

In code, PyTorch provides `nn.MultiheadAttention` which does a lot of this under the hood. Here’s a conceptual example using it:

```python
import math
import torch.nn.functional as F

# Suppose we have a sequence of length L=5 with d_model=16 features.
# We'll create a single MultiheadAttention with 2 heads.
MHA = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)

# Dummy data: sequence of 5 tokens (batch_size=1 for simplicity)
x = torch.randn(1, 5, 16)  # (batch, seq, features)

# Self-attention: query, key, value are all x (sequence attends to itself)
attn_output, attn_weights = MHA(x, x, x)
print("Attention output shape:", attn_output.shape)   # (1, 5, 16)
print("Attention weights shape:", attn_weights.shape) # (1, 2, 5, 5) => 2 heads, 5 queries, 5 keys

```

This will output a new sequence of the same shape, where each position’s representation is now a blend of the original positions. The `attn_weights` shows for each head and each query position, the weights over 5 key positions (in a 5x5 matrix per head). In a full Transformer, this operation is followed by the feed-forward network. The decoder would use a masked version of self-attention (to prevent attending to future tokens that haven’t been generated yet) and a cross-attention to encoder outputs.

Why did Transformers revolutionize NLP?

- **Parallelism:** Unlike RNNs, an entire sequence can be processed in parallel during training (since at training time, you have the whole sentence). This makes it much faster to train on GPUs for long sequences.
- **Long-range dependencies:** Self-attention doesn’t care if something is 10 or 100 tokens away; it can still directly attend with one step. Transformers handled long sentences or documents more effectively than RNNs which struggled beyond say 30-50 time steps of context.
- **Scaling:** With more data and compute, transformers just kept getting better. We could scale them to unprecedented sizes (millions to billions of parameters) and they’d utilize it to improve performance.

The original Transformer model rapidly became state-of-the-art in machine translation. But perhaps more importantly, it became the foundation for pretraining large language models that could be fine-tuned: which leads us to **BERT** and **GPT**.

## BERT, GPT, and Modern [[Transformers]] – The Age of Language Models

Following the Transformer architecture, researchers realized that training _huge_ transformer models on large unlabeled text corpora and then fine-tuning them on specific tasks was a winning strategy. This ushered in the era of **pretrained language models**.

Two dominant paradigms emerged:

- **Encoder-based models (e.g., BERT)** – good for understanding tasks (classification, Q&A, etc.).
- **Decoder-based models (e.g., GPT)** – good for generative tasks (text completion, writing).

**BERT (2018)** – _Bidirectional Encoder Representations from Transformers_, introduced by Devlin _et al._ at Google​

[en.wikipedia.org](https://en.wikipedia.org/wiki/BERT_\(language_model\)#:~:text=BERT%20%28language%20model%29%20,2018%20by%20researchers%20at%20Google)

. BERT uses the Transformer _encoder_ stack and is trained with a clever self-supervised objective: _Masked Language Modeling_ (MLM). They take sentences and mask out some words (e.g., 15% of them) and ask the model to predict those missing words. This forces the model to build a deep understanding of context (it has to use both left and right context – hence “bidirectional”). BERT is also trained on a next sentence prediction task (to understand sentence relationships). After this pretraining on a massive corpus (Wikipedia, books), BERT can then be fine-tuned on downstream tasks like sentiment analysis, question answering, named entity recognition, etc., where it broke many state-of-the-art records upon release. Essentially, BERT provided a **universal language understanding model** that one can specialize with relatively little data.

In code, using a pretrained BERT is easy with libraries like Hugging Face Transformers:

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text = "Deep learning is transforming AI."
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  # representation for each token

```

This would download a pre-trained BERT and output hidden states for each token in the input text. Fine-tuning for a specific task would involve adding a task-specific head (e.g., a classifier on top of the `[CLS]` token representation for classification).

**GPT (2018 for GPT-1, GPT-2 in 2019, GPT-3 in 2020)** – _Generative Pre-trained Transformer_, by OpenAI. GPT models use the Transformer _decoder_ stack (which is causal, one-directional). They are trained with the straightforward objective of next-word prediction: given a text so far, predict the next token. This is the classic language modeling setup, but with a transformer. GPT-2 (2019) garnered attention for its ability to generate coherent multi-paragraph texts and was not fully released initially due to “concerns about misuse”. GPT-2 had up to 1.5 billion parameters. Then GPT-3 (2020) blew the doors off with **175 billion parameters**​

[developer.nvidia.com](https://developer.nvidia.com/blog/openai-presents-gpt-3-a-175-billion-parameters-language-model/#:~:text=OpenAI%20researchers%20recently%20released%20a,up%20of%20175%20billion%20parameters)

– a model so large it could perform tasks with few-shot prompting (just by giving it a few examples in the prompt) that normally would have required fine-tuning. GPT-3 could write essays, code, summarize, do basic math, all emerging from the massive training on virtually all of the internet’s text.

OpenAI’s GPT models demonstrated that _scaling up_ and training on enormous data can yield models with surprising capabilities (this scaling notion was theorized in papers like “Scaling Laws for Neural Language Models”). They also set the stage for products like GPT-3 powered APIs, and eventually ChatGPT (which is a fine-tuned GPT model on conversational data).

**Transformer variants and innovations:** After the original Transformer, many variants appeared:

- **Transformer-XL (2019):** introduced recurrence in transformers to handle very long texts beyond a fixed length by caching past hidden states.
- **Reformer, Longformer, etc.:** addressed the $O(n^2)$ complexity of attention for long sequences by approximations or locality-sensitive hashing.
- **XLNet (2019):** a permutation-based language model (autoregressive like GPT but learned to predict tokens in random order, capturing bidirectional context without masking).
- **T5 (2019):** a text-to-text Transformer that unified NLP tasks by representing everything (input and output) as text.
- **Megatron, GShard, etc.:** techniques for training huge models over multiple GPUs/TPUs.

One concrete example using a decoder model (like GPT-2) via Hugging Face might be:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
prompt = "Once upon a time in deep learning,"
inputs = tok(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, top_k=50)
print(tok.decode(outputs[0]))

```

This would generate a continuation of the prompt using GPT-2. It showcases how a decoder-only model is great for text generation. The `generate` method handles the iterative process of feeding the model its own outputs to generate multi-token sequences.

The Transformer revolution in NLP is often likened to a Cambrian explosion. NLP tasks that once had bespoke models and feature engineering were replaced wholesale by fine-tuning a pretrained transformer. As of the early 2020s, transformers power virtually all state-of-the-art language systems.

But transformers haven’t just stayed in text. Researchers soon asked: can we apply the same “attention-only” approach to images? That leads us to the next milestone.

## Vision [[Transformers]] (ViT) – Transformers in the Land of CNNs

Convolutional neural networks had been the undisputed champs of vision for a long time. But in 2020, Dosovitskiy _et al._ introduced the **Vision Transformer (ViT)**, showing that a pure transformer architecture can attain state-of-the-art in image classification, provided you have enough data (or use pre-training). This was a provocative result: could we dispense with convolutions for vision tasks too, in favor of attention?

**How Vision Transformers work:** The main obstacle to applying a transformer to images is that images are 2D grids of pixels, not 1D token sequences. The ViT solution: **split the image into patches** and treat each patch as a “token”. For example, take a 224×224 image, split it into 16×16 patches – that gives you $14 \times 14 = 196$ patches (if using non-overlapping 16×16). Flatten each patch (16_16_3 = 768 values if 3-channel) and linearly project it to a feature vector of dimension $D$ (the model’s hidden size, say 768). These become the input tokens. Also prepend a special classification token (learnable [CLS] embedding, similar to BERT’s [CLS]) as the first token. Add positional embeddings (so the transformer knows patch positions), and then feed these tokens through a standard Transformer encoder.

![](blob:https://chatgpt.com/2918e407-9c3d-490b-a925-8de6041d1423)

_Figure: **Vision Transformer architecture.** The image is split into fixed-size patches (e.g., 16×16 pixels) which are flattened and fed through a linear projection to create patch embeddings. A special [CLS] token is prepended (purple), and its output at the end is used for classification. The sequence of patch embeddings is then processed by a Transformer Encoder (gray box) just like a sequence of word embeddings​_

_[Vision Transformers architecture](https://en.wikipedia.org/wiki/Vision_transformer#/media/File:Vision_Transformer.png)_

_. Finally, an MLP head (yellow) on [CLS] outputs the class. ViTs thus treat images like sequences of word patches._

The ViT doesn’t have any convolution or pooling; it just has the self-attention layers and MLPs. It relies on the attention mechanism to aggregate information from across the image patches.

One might wonder: doesn’t this ignore local structure that CNNs exploit? Indeed, transformers are **ignorant of locality** unless the model learns it or positional embeddings impart some of it. This means ViTs typically need a lot of training data to learn the notion of locality and basic visual features (whereas CNNs have inductive bias for locality and translation invariance). In the ViT paper, they pre-trained on very large datasets (like JFT-300M, which has 300 million images) and then fine-tuned to tasks like ImageNet. With enough data, ViT matched or exceeded CNNs. On smaller datasets, ViTs initially underperformed CNNs, but techniques like data augmentation, regularization, or hybrid models (convolutional stem + transformer) helped close the gap.

By 2021-2022, **Vision Transformers and hybrid models** were matching CNN performance in classification, and were extended to detection and segmentation (with some modifications like hierarchical structure – e.g., Swin Transformer introduced a shifting window approach to restrict attention locally, effectively creating a pyramid akin to CNN feature maps).

Using a ViT via code (huggingface offers ViT models too):

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

image = ...  # PIL image or NumPy array
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
pred_logits = outputs.logits
pred_label = pred_logits.argmax(-1)
print("Predicted label:", model.config.id2label[pred_label.item()])

```

This would load a pretrained ViT (base model, 16×16 patches, 224 input res) and classify an input image.

The success of transformers in vision is a testament to the generality of attention mechanisms. It’s almost poetic: CNNs were inspired by the visual cortex and dominated vision; transformers came from language, but turned out to be so powerful that given enough data they could even see images in a new way (by “reading” them patch by patch).

We’ve now seen how neural networks have evolved to excel at perception (vision, text) either discriminatively or via generative means like GANs. Our final stop in this journey is the latest advances in generative modeling that go beyond GANs – namely, **Diffusion Models** and related modern generative techniques.

## Diffusion Models & Modern Generative Models – The New Wave of Generative AI

In recent years, a new class of generative models has surged to the forefront, especially for image synthesis: **Denoising Diffusion Models** (often just called diffusion models). In 2022, models like **DALL·E 2, Stable Diffusion, Imagen** (all diffusion-based) stunned the world by generating incredibly detailed images from text prompts, arguably surpassing GANs in quality and diversity.

**What are Diffusion Models?** At a high level, diffusion models generate data by **gradually denoising** random noise into a desired output​

[shelf.io](https://shelf.io/blog/diffusion-models-for-machine-learning/#:~:text=Diffusion%20models%20for%20machine%20learning,and%20coherent%2C%20like%20an%20image)

. The process is typically formulated in two parts:

- A **forward process** (diffusion): start from a real image and gradually add noise over many steps until it becomes pure noise (like adding tiny perturbations step by step). This forward process is fixed, often Gaussian noise added at each step.
- A **reverse process** (denoising): a learned model tries to invert this noising process, i.e., start from pure noise and remove noise step by step, moving toward a sample from the data distribution.

The model (often a U-Net convolutional neural network) is trained to predict the noise added at a given step or directly predict the denoised image at the previous step. By training on lots of images, it learns how to gradually paint images out of noise.

_Analogy:_ Diffusion model generation is like developing a Polaroid photo but in reverse​

[shelf.io](https://shelf.io/blog/diffusion-models-for-machine-learning/#:~:text=Imagine%20an%20artist%20starting%20with,and%20even%20new%20scientific%20research)

​

[shelf.io](https://shelf.io/blog/diffusion-models-for-machine-learning/#:~:text=Diffusion%20models%20for%20machine%20learning,and%20coherent%2C%20like%20an%20image)

. Initially, the photo is just gray static (random noise). As time progresses (steps in the model), the image becomes clearer as noise is gently removed, eventually revealing a coherent picture. Another analogy: it’s like sculpting – you start with a block of marble (noise) and chisel away (denoise) little by little until the sculpture (image) appears. This is different from GANs, which attempt to generate an image in essentially one shot through a feedforward pass. Diffusion breaks the problem into many smaller steps, which can make it easier to model complex distributions.

**Why diffusion models excel:**

- They don’t suffer from mode collapse like GANs. Because each training step focuses on slightly denoising, the model effectively spreads its effort across the whole data distribution. Empirically, diffusion models generate more diverse outputs covering the modes of the data better.
- The loss function (often simple L2 loss on noise prediction) is more stable and easier to optimize than the adversarial loss of GANs. There’s no adversary to worry about.
- You can trade off speed vs quality by adjusting number of sampling steps. Fewer steps = faster generation but maybe lower quality, more steps = slower but higher fidelity.
- Diffusion models also allow **guidance**: e.g., _classifier-guided diffusion_ or _classifier-free guidance_ where you can modulate the generation towards certain classes or prompts (that’s how text-to-image works: the model is trained with a text encoder; at generation time the text embedding “guides” the diffusion process).

**Training diffusion models:** One popular formulation is the **Denoising Diffusion Probabilistic Model (DDPM)** by Ho _et al._ (2020). It uses a **schedule** of noise addition (e.g., $\beta_1, \beta_2, \dots, \beta_T$ variances for T steps). The model (a U-Net) is trained to predict the noise $ϵ$ added to an image at an arbitrary step $t$ given the noisy image $x_t$ (and possibly conditional input like text embedding). The loss is $E_{t,x_0,ϵ}[ |;ϵ - ϵ_\theta(x_t, t, \text{conditioning});|^2 ]$. Each training sample involves taking a real image $x_0$, sampling a random timestep $t$, adding noise to get $x_t$, and then asking the model to predict the noise.

**Sampling (reverse process):** Start with $x_T \sim \mathcal{N}(0, I)$ (pure noise). Then for $t=T$ down to $1$, use the model to predict $ϵ_\theta(x_t, t)$, then compute a denoised estimate of $x_{t-1}$. This usually involves the formula: $x_{t-1} = \frac{1}{\sqrt{α_t}}(x_t - \frac{1-α_t}{\sqrt{1- \bar{α}_t}} ϵ_\theta(x_t,t)) + \sigma_t z$ (looks complex but basically one step of Gaussian denoising with some random perturbation $\sigma_t z$ added for stochasticity, where $α_t = 1 - β_t$ and $\bar{α}_t$ is cumulative product). If $T$ is large (e.g., 1000), this closely approximates the true data distribution.

There have been many improvements:

- **Improved sampling**: techniques like DDIM allow deterministic faster sampling.
- **Diffusion in latent space**: e.g., Latent Diffusion Models (used in Stable Diffusion) run diffusion on a compressed latent (like the output of a VAE’s encoder), greatly speeding up computation.
- **Conditional diffusion**: adding conditions like text (via CLIP text encoders), class labels, or other modalities.

**Modern generative model landscape:** Diffusion models have essentially become state-of-the-art for image generation, outperforming GANs on metrics like FID for many tasks​

[shelf.io](https://shelf.io/blog/diffusion-models-for-machine-learning/#:~:text=Diffusion%20models%20are%20a%20type,learned%20patterns%20in%20existing%20data)

​

[shelf.io](https://shelf.io/blog/diffusion-models-for-machine-learning/#:~:text=Diffusion%20models%20for%20machine%20learning,and%20coherent%2C%20like%20an%20image)

. They’re also being applied to audio (e.g., generating music or speech), video, and more. Other models like **autoregressive transformers** (e.g., ImageGPT, VQ-VAE + transformer decoders) are also competitive for image generation but can have issues with high-res outputs. **GANs** are still used where speed is paramount or data is limited, but the trend is towards diffusion or hybrid approaches.

And of course, there’s crossover: hybrid models like **MaskGIT** (which use transformer with diffusion-like masking schedule), or **Score-based models** (diffusion interpreted via score matching). Research is very active here.

To illustrate a tiny bit of a diffusion model concept in code (not full training, just an idea), suppose we have a trained noise predictor `eps_model(x_noisy, t)` and we want to do a naive sampling:

```python
# Pseudocode for diffusion sampling (very simplified)
x_T = torch.randn(1, 3, 64, 64)  # start from pure noise image 3x64x64
for t in range(T, 0, -1):
    eps_pred = eps_model(x_T, torch.tensor([t]))
    # compute x_{t-1} from x_t and eps_pred (here assume variance fixed, simplified)
    x_T = 1/sqrt(alpha[t]) * (x_T - (1-alpha[t])/sqrt(1-alpha_cum[t]) * eps_pred)
    if t > 1:
        z = torch.randn_like(x_T)
        x_T = x_T + sigma[t] * z  # add some noise for stochasticity
# Now x_T is x_0 (approximately a sample from the model)

```

This is a rough sketch. In practice, you’d use libraries like Hugging Face diffusers which have implementations for stable diffusion, etc.

**Impact:** Diffusion models (and large transformers) have launched an era of generative AI in mainstream awareness. Tools like **Stable Diffusion** allow anyone to create art from text. **ChatGPT** (an instruction-tuned GPT model) can generate human-like conversations and code. It feels like science fiction becoming real – generative models can produce images, text, even videos (rudimentary but improving) and audio that were once exclusive to human creativity. As an engineer, it’s both exciting and a bit daunting to see models that can do so much.

---

## Epilogue

From the triumph of AlexNet in 2012 to the generative marvels of diffusion models in 2025, the field of neural networks has advanced at breakneck speed. We witnessed:

- The resurgence of deep **CNNs** and their optimization tricks (ReLU, dropout, GPU training) opening the floodgates of deep learning.
- The development of **RNNs** and improvements like **LSTMs/GRUs** that gave neural nets a form of memory, enabling sequence understanding.
- Creative unsupervised networks like **autoencoders** learning hidden factors of data.
- The advent of **GANs** introducing adversarial training – neural nets sparring to create realistic data.
- Hinton’s visionary **Capsule Networks** attempting to preserve entity pose and part-whole relationships.
- **ResNets** making ultradeep networks trainable via skip connections, a simple fix that had profound impact.
- The **Transformer** revolution – leveraging attention to handle sequences more effectively than ever, and scaling to unimaginable model sizes.
- Specialized transformers like **BERT** and **GPT** that ushered in pre-trained language models as general NLP solutions.
- The encroachment of transformers into vision with **ViT**, challenging CNN supremacy by treating images as patch sequences.
- And finally, the rise of **Diffusion Models** and other modern generative methods, pushing the boundaries of what AI can create, and hinting at a future where generating content is as core a capability of neural networks as perceiving content.

It’s been a decade of wonders in AI. Each breakthrough built on the last: skip connections helping transformer layers, CNN knowledge helping design better positional encodings or efficient attention, etc. As a machine learning engineer, staying abreast of these developments sometimes feels like watching a high-speed train – but also offers a toolkit of techniques to mix and match for new problems. And who knows what the next chapter will bring? Perhaps new architectures that combine the strengths of all we’ve seen, or completely new paradigms (spiking neural nets? neural ODEs? something beyond tensor-based learning?).

One thing is certain: the story is far from over. If the journey from 2012 to 2025 transformed the landscape of AI, the next decade may well bring innovations we can barely imagine – and we’ll have an even longer story to tell then, hopefully with the same excitement and awe we feel today looking back at these milestones.