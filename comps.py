import tensorflow as tf
tnp = tf.experimental.numpy
nn = tf.keras

# Positional Embedding
class PositionalEncoding(nn.layers.Layer):
    """Sine-Cosine Positional Embedding"""
    def __init__(self, max_length:int, d_model:int, **kwargs):
        super().__init__(**kwargs)
        p, i = tf.meshgrid(tf.range(float(max_length)), 2*tf.range(d_model/2))
        
        theta = p/10_000**(i/d_model)
        angle = tf.transpose(theta)

        self.pos_embed = tf.reshape(tf.stack([tf.sin(angle), tf.cos(angle)]), (maxlen, d_model))

    def call(self, x=None):
        if x is None:
            return self.pos_embed[tf.newaxis] # (1, max_length, d_model) 
        return self.pos_embed[:x.shape[1], :][tf.newaxis] # (1, T, d_model)

# Tok + Pos Embeddings
class Embed(nn.layers.Layer):
    """token_embedding + positional_embedding"""
    def __init__(
            self,
            max_length:int,
            vocab_size:int,
            d_model:int,
            learnt_pos_embed:bool,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.word_embed = nn.layers.Embedding(vocab_size, d_model) # (B, T, d_model) =(vocab_size, d_model)=> (B, T)
        self.position_embed = (
            PositionalEncoding(max_length, d_model)() if learnt_pos_embed 
            else nn.layers.Embedding(max_length, d_model)(tf.range(max_length))
        )

    def call(self, inputs):
        T = inputs.shape[1]
        tok_embed = self.word_embed(inputs) # (B, T, d_model)
        pos_embed = self.position_embed[:T, :] # (MAX_LENGTH, d_model) =[:T, :]=> (T, d_model)
        return tok_embed + pos_embed # (B, T, d_model) + (T, d_model) ==> (B, T, d_model)

# Attention
## Beginner? Start with this
class AttentionV1(nn.layers.Layer):
    def __init__(
                self, 
                causal: bool, 
                d_model: int, 
                n_heads: int, 
                dropout_rate: float, 
                **kwargs
        ):
        super().__init__(**kwargs)
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.dq = self.dk = self.dv = d_model//n_heads

        self.wq = nn.layers.Dense(self.dq, use_bias=False)
        self.wk = nn.layers.Dense(self.dk, use_bias=False)
        self.wv = nn.layers.Dense(self.dv, use_bias=False)
        self.w = nn.layers.Dense(d_model, use_bias=False)
        
        self.dropout = nn.layers.Dropout(rate=self.dropout_rate) 

    def attention(self, Q, K, V):
        mask_tensor = lambda x: tf.where(tnp.tril(tf.ones_like(x)) == 0, float('-inf'), x)
        scores = tf.matmul(Q, K, transpose_b=True)/self.dk**0.5 # (B, T, T) <= (B, T, dq) @ (B, T, dk).T
        scores = mask_tensor(scores) if self.causal else scores
        return tf.nn.softmax(scores, axis=-1) @ V # (B, T, d_v)

    def head(self, inp2q, inp2k, inp2v):
        Q, K, V = self.wq(inp2q), self.wk(inp2k), self.wv(inp2v)
        return self.attention(Q, K, V)

    def call(self, inp2q, inp2k, inp2v, training=False):
        heads = tf.concat([self.head(inp2q, inp2k, inp2v) for _ in range(self.n_heads)], axis=-1)
        output = self.linear(heads)
        return self.dropout(output, training=training) if training and self.dropout_rate!=0 else output

# Now comfortable with Multi-dimensional Tensors? Go with this
class AttentionV2(nn.layers.Layer):
    def __init__(
            self, 
            causal:bool, 
            d_model:int, 
            n_heads:int, 
            dropout_rate:float, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.causal = causal
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.dq = self.dk = self.dv = d_model//n_heads

        self.w = nn.layers.Dense(d_model, use_bias=False)
        self.wq = nn.layers.Dense(d_model, use_bias=False)
        self.wk = nn.layers.Dense(d_model, use_bias=False)
        self.wv = nn.layers.Dense(d_model, use_bias=False)
        self.dropout = nn.layers.Dropout(rate=dropout_rate)

    
    def _mask(self, x:tf.Tensor):
        
    
    def call(
            self, 
            inp2q:tf.Tensor, 
            inp2k:tf.Tensor, 
            inp2v:tf.Tensor,
            training=False
        ):
        B, N, d_model = inp2q.shape
        T = inp2k.shape[0]

        # compute q, k, v
        q = self.wq(inp2q) # (B, N, d_model)
        k = self.wk(inp2k) # (B, T, d_model)
        v = self.wv(inp2v) # (B, T, d_model)
        
        # seperate heads
        q = tf.reshape(q, (self.n_heads, B, N, self.dq)) # (h, B, N, dq)
        k = tf.reshape(k, (self.n_heads, B, T, self.dk)) # (h, B, T, dk)
        v = tf.reshape(v, (self.n_heads, B, T, self.dv)) # (h, B, T, dv)

        # compute attention weights
        att_wei = tf.matmul(q, k, transpose_b=True)/d_model**0.5 # (h, B, N, T)
            if self.causal:
                tril = tnp.tril(tf.ones_like(x))
                att_wei = tf.where(tril==0., -tnp.inf, x) # (h, B, N, T)
        att_wei = tf.nn.softmax(att_wei, axis=-1) # (h, B, N, N)

        # apply attention weights to v
        att_out = att_wei @ v # (h, B, N, dv)
        # combine heads
        att_out = tf.reshape(att_out, (B, N, d_model)) # (B, T, h*dv) == (B, T, d_model)

        # linear of att_out
        linear_att_out = self.w(att_out) # (B, T, d_model)
        return self.dropout(linear_att_out) if training and self.dropout_rate!=0 else linear_att_out
    
