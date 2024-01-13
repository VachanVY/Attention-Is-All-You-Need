import tensorflow as tf
tnp = tf.experimental.numpy
nn = tf.keras

from comps import (
    AttentionV2 as Attention,
    Embed
)

class Args:
    d_model:int
    n_heads:int
    assert d_model % n_heads == 0
    vocab_size:int
    max_length:list
    dropout_rate:float

    num_encoder_layers:int
    num_decoder_layers:int


def Transformer(Args:Args):
    FeedForward = lambda: nn.Sequential([
                nn.layers.Dense(Args.d_model*4),
                nn.layers.Activation(nn.activations.relu),         
                nn.layers.Dense(Args.d_model),
                nn.layers.Dropout(Args.dropout_rate)
            ])

    ## Transformer Network
    encoder_inputs = nn.Input(shape=(None,)) # so can take inputs of varied length
    x = Embed(
        learnt_pos_embed=False,
        d_model=Args.d_model,
        vocab_size=Args.vocab_size,
        max_length=Args.max_length[0],
    )(encoder_inputs)
    # Encoder Blocks
    for _ in range(Args.num_encoder_layers):
        z = Attention(
            causal=False,
            d_model=Args.d_model,
            n_heads=Args.n_heads,
            dropout_rate=Args.dropout_rate
            )(x, x, x)
        x = nn.layers.LayerNormalization()(nn.layers.Add()([z, x]))

        z = FeedForward()(x)
        x = nn.layers.LayerNormalization()(nn.layers.Add()([z, x]))
    encoder_output = x

    decoder_inputs = nn.Input(shape=(None,))
    x = Embed(
        learnt_pos_embed=False,
        d_model=Args.d_model,
        vocab_size=Args.vocab_size,
        max_length=Args.max_length[1],
    )(decoder_inputs) # (B, T, d_model)
    # Decoder Blocks
    for _ in range(Args.num_decoder_layers):
        z = Attention(
            causal=True,
            d_model=Args.d_model,
            n_heads=Args.n_heads,
            dropout_rate=Args.dropout_rate
            )(x, x, x) # (B, T, d_model)
        x = nn.layers.LayerNormalization()(nn.layers.Add()([z, x]))

        z = Attention(
            causal=True,
            d_model=Args.d_model,
            n_heads=Args.n_heads,
            dropout_rate=Args.dropout_rate
        )(x, encoder_output, encoder_output) # (B, T, d_model)
        x = nn.layers.LayerNormalization()(nn.layers.Add()([z, x]))

        z = FeedForward()(x)
        x = nn.layers.LayerNormalization()(nn.layers.Add()([z, x]))
            
    logits = nn.layers.Dense(Args.vocab_size)(x) # (B, T, vocab_size)

    model = nn.Model(inputs=[encoder_inputs, decoder_inputs], outputs=logits, name="transformer")
    print("Number of parameters in the model", model.count_params())
    return model
