import jax
from flax import nnx
import jax.numpy as jnp
import jax.random as random
from time import time

@nnx.dataclass
class CLIPTextConfig:
	vocab_size: int = 49408
	hidden_size: int = 512
	intermediate_size: int = 2048
	projection_dim: int = 512
	num_hidden_layers: int = 12
	num_attention_heads: int = 8
	max_position_embeddings: int = 77
	
class CLIPTextEmbeddings(nnx.Module):
	def __init__(self, config:CLIPTextConfig, rngs:nnx.Rngs):
		self.token_embedding = nnx.Embed(num_embeddings=config.vocab_size, features=config.hidden_size, rngs=rngs)
		self.position_embedding = nnx.Param(nnx.initializers.normal(stddev=0.02)
		(rngs.param(),(config.max_position_embeddings,config.hidden_size))) # (77 x 512)
	
	def __call__(self, input_ids):	
		seq_len = input_ids.shape[1]
		x = self.token_embedding(input_ids)
		x = x + self.position_embedding.get_value()[:seq_len,:] #(Seq_Len X 512)
		return x

class CLIPTextAttention(nnx.Module):
	def __init__(self, config:CLIPTextConfig, rngs:nnx.Rngs):
		self.mha = nnx.MultiHeadAttention(
			num_heads=config.num_attention_heads,
			in_features=config.hidden_size,
			qkv_features=config.hidden_size,
			out_features=config.hidden_size,
			rngs=rngs
			)

	def __call__(self, x, mask):
		return self.mha(inputs_q=x, mask=mask, decode=False)

class CLIPTextEncoderLayer(nnx.Module):
	def __init__(self, config:CLIPTextConfig, rngs:nnx.Rngs):
		self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
		self.attn = CLIPTextAttention(config, rngs=rngs)
		self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)

		self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
		self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, rngs=rngs)

	def __call__(self, x, mask):
		x = x + self.attn(self.layer_norm1(x), mask=mask)
		residual = x
		x = self.layer_norm2(x)
		x = self.fc1(x)
		x = nnx.gelu(x)
		x = self.fc2(x)
		x = residual + x

		return x

class CLIPTextEncoder(nnx.Module):
	def __init__(self, config:CLIPTextConfig, rngs:nnx.Rngs):
		self.embeddings = CLIPTextEmbeddings(config, rngs=rngs)
		self.layers = nnx.List([CLIPTextEncoderLayer(config, rngs=rngs) for _ in range(config.num_hidden_layers)])
		self.final_layer_norm = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
		self.text_projection = nnx.Linear(config.hidden_size, config.projection_dim, use_bias=False, rngs=rngs)

	def __call__(self, input_ids:jnp.ndarray,):
		seq_len = input_ids.shape[1]

		x = self.embeddings(input_ids)
		causal_mask = jnp.tril(jnp.ones((seq_len,seq_len),dtype=jnp.bool_))
		causal_mask = causal_mask[jnp.newaxis, jnp.newaxis, :, :]

		for layer in self.layers:
			x = layer(x, mask=causal_mask)

		eos_token_indices = jnp.argmax(input_ids, axis=-1)

		pooled_x = x[jnp.arange(input_ids.shape[0]), eos_token_indices, :]	
		pooled_x = self.final_layer_norm(pooled_x)
		text_features = self.text_projection(pooled_x)

		return text_features


if __name__ == '__main__':

	rngs = nnx.Rngs(42)

	config = CLIPTextConfig()
	text_embedding = CLIPTextEmbeddings(config, rngs)
	text_attention = CLIPTextAttention(config, rngs)
	text_encoder_layer = CLIPTextEncoderLayer(config, rngs)
	text_encoder_block = CLIPTextEncoder(config, rngs)

	mock_input_ids = jax.random.randint(jax.random.PRNGKey(0),shape=(2,10), minval=0, maxval = config.vocab_size)
	start_time_tec = time()
	output_features = text_embedding(mock_input_ids)
	end_time_tec = time()

	seq_len = output_features.shape[1]
	causal_mask = jnp.tril(jnp.ones((seq_len,seq_len),dtype=jnp.bool_))
	causal_mask = causal_mask[jnp.newaxis,jnp.newaxis,:,:]
	start_time_tac = time()
	output_attention = text_attention(output_features,mask=causal_mask)
	end_time_tac = time()

	start_time_elc = time()
	encoder_layer_output = text_encoder_layer(output_features, mask=causal_mask)
	end_time_elc = time()

	start_time_ebc = time()
	encoder_block_output = text_encoder_block(mock_input_ids)
	end_time_ebc = time()

	print()
	print("="*20,"Text Embedding Class","="*20)
	print("Input Shape (Batch, Seq_Len):", mock_input_ids.shape)
	print("Output Shape (Batch, Seq_Len, Hidden_dim):", output_features.shape)
	print("Type of text embedding:",type(text_embedding))
	print("Took",end_time_tec-start_time_tec,"seconds")
	print()
	print("="*20,"Text Attention Class","="*20)
	print("Input Features Shape (Batch, Seq_Len, Hidden_dim):", output_features.shape)
	print("Output Attention Shape (Batch, Seq_Len, Hidden_dim):", output_attention.shape)
	print("Type of text attention output ", type(output_attention))
	print("Took",end_time_tac-start_time_tac,"seconds")
	print()
	print("="*20,"Encoder Layer Class","="*20)
	print("Encoder Layer Output Shape:", encoder_layer_output.shape)
	print("Type of encoder layer output:",type(encoder_layer_output))
	print("Took",end_time_elc-start_time_elc,"seconds")
	print()
	print("="*20,"Encoder Class","="*20)
	print("Encoder Output Shape:", encoder_block_output.shape)
	print("Type of Encoder Output:",type(encoder_block_output))
	print("Took",end_time_ebc-start_time_ebc,"seconds")
	print()
	summary_table = nnx.tabulate(text_encoder_block, mock_input_ids)
	print(summary_table)

	'''Output:
	==================== Text Embedding Class ====================
	Input Shape (Batch, Seq_Len): (2, 10)
	Output Shape (Batch, Seq_Len, Hidden_dim): (2, 10, 512)
	Type of text embedding: <class '__main__.CLIPTextEmbeddings'>
	Took 0.08970499038696289 seconds

	==================== Text Attention Class ====================
	Input Features Shape (Batch, Seq_Len, Hidden_dim): (2, 10, 512)
	Output Attention Shape (Batch, Seq_Len, Hidden_dim): (2, 10, 512)
	Type of text attention output  <class 'jaxlib._jax.ArrayImpl'>
	Took 0.45462512969970703 seconds

	==================== Encoder Layer Class ====================
	Encoder Layer Output Shape: (2, 10, 512)
	Type of encoder layer output: <class 'jaxlib._jax.ArrayImpl'>
	Took 0.5069653987884521 seconds

	==================== Encoder Block Class ====================
	Encoder  Output Shape: (2, 512)
	Type of Encoder Output: <class 'jaxlib._jax.ArrayImpl'>
	Took 0.5094683170318604 seconds

	[Table like Keras model.summary()]
	Total Parameters: 63,428,096 (253.7 MB)
	'''