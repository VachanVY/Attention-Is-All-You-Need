# Transformers: Attention Is All You Need

What do humans and artificial neural networks agree on?
That Attention Is All You Need ;)

![Alt text](images/image-1.png)
![Alt text](images/image.png)

## Introduction
* Was difficult to learn dependencies between distant positions. In Transformers, this is reduced to a constant number of operations
* Self-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence

## Embedding
### Input Embedding

* Use Learned Embeddings to convert input tokens and output tokens to vectors of dimention `d_model`
* We share the same weight matrix between the 2 Enbedding layers and pre-softmax linear transformation

### Positional Embedding
* The model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the embeddings
*
```python
   PE_(pos, i) = sin(pos/1e4**(i/d_model)) if i % 2 == 0 else cos(pos/1e4**(i-1/d_model))
```
* The authors chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset `k`, `PE_(pos+k)` can be represented as a linear function of
`PE_(pos)`
* They also experimented with using learned positional embeddings instead, and found that the two versions produced nearly identical results. We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training

* ![Alt text](image-2.png)
* However, you can also apply the normal embedding used in word embeddings for positional embeddings

## Scalled-Dot-Product Attention
* ![Alt text](image-3.png)
* The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values).

* Retrieves a value `v_i` for a query `q` based on a key `k_i` in database.

* The tensor `score` before masking, tells how much correlated each word is for other words in the sentence as shown above
<img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*D8GA7_DsjTfudI0-.png" width="300">
* Why masking? Eg: "am" shouldn't have access to the attention weight of "fine" as it is from the future. Similarly for others
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/0*0pqSkWgSPZYr_Sjx.png" width="300">
* So, we mask the future attention weights with `-inf`
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*QYFua-iIKp5jZLNT.png" width="600">

* Then the softmax function makes them 0
<img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/0*3ykVCJ9okbgB0uUR.png" width="600">
* If we mask the illegal connections as `0` after softmax, the sum won't be `1`, so we mask the illegal connections as `-inf` before applying softmax, so after applying softmax the sum will be `1`

## Multi-Head Attention

* Instead of performing a single Attention function with Queries, Key and Values, the authors found that it is beneficial to linearly tranform the Queries, Key and Values `h` times with different learned linear projections as shown in the above figure. Then they are concatenated and linear transformed again.

* Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

`MultiHead(Q, K, V) = Concat(h_1, ..., h_h)`
  where `h_i = Scalled_Dot_Product_Attention(W_Q @ Q, W_K @ K, W_V @ V)`

## Feed-Forward Layers
* 2 Feed-Forward Layers, 1st with a relu activation and the 2nd without any activation
* The dimentionality of input and output is `d_model`, the inner-layer has dimentionality `d_model*4`

## The Model
### For Machine Translation
* Encoder: Each layer has 2 sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network.There are residual connections also.
Output of each SubLayer is `LayerNorm(x + Sublayer(x))`.

* Decoder: Has 3 sub-layers unlike the Encoder. The first sublayer is modified to prevent positions from attending further positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position `i` can depend only on the known outputs at positions less than `i`.

* <img src="https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" width="500">

---

### For AutoRegressive Tasks

* **GPT-1 architecture**

  <img src="https://www.researchgate.net/publication/349521999/figure/fig1/AS:994338746146817@1614080239754/Decoder-Only-Architecture-used-by-GPT-2.ppm" width="400">
---

### How is Attention applied?

* **Encoder-Decoder Attention (Cross-Attention):**
 * *Queries from previous Decoder*
 * *Keys and Values (both are same) from output of Encoder*
 * *This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder RNN-attention-mechanisms in seq-to-seq models*

* **Self-Attention in Encoder:**
  * *Keys, Queries and Values come from same place, in this case the input or the previous layer of Encoder*

* **Self-Attention in Decoder:**
  * *We need to prevent the leftward flow of information i.e information from the future, otherwise it would be cheating as the model would see words which it should have to predict later (in the context of seq2seq-RNN model, we use Biderectional RNN in Encoder, but not in Decoder so that it doesn't cheat, RNN is a sequential model so there's no need to mask, but transformer is not a sequential model so we need to mask the inputs)*

  * *We implement this inside Scaled-Dot-Product-Attention, by masking out
   (setting to `float("-inf")`) all values in the input of the softmax which correspond to illegal connections

---

### Regularization: Dropout

* Dropout is applied to the output of each sub-layer, before it is added to the sub-layer input and normalized
 Dropout is also applied to the embeddings in both the Encoder and Decoder
---

### Optimizer:

* Adam Optimizer with `β1=0.9`, `β2=0.98` and `epsilon=1e-9`
* Learning Rate Schedule:
  * `lr = (d_model**-0.5) * min(step**-0.5, step * warmup_steps**-1.5)`
  * This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.
