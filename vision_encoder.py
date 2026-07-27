import jax
from flax import nnx
import jax.numpy as jnp
import jax.random as random
from time import time

@nnx.dataclass
class CLIPVisionConfig:
	image_size: int = 224
	patch_size: int = 32
	hidden_size: int = 768
	intermediate_size: int = 3072
	projection_dim: int = 512
	num_hidden_layers: int = 12
	num_attention_heads: int = 12
	num_channels: int = 3

class CLIPVisionEmbeddings(nnx.Module):
	def __init__(self, config:CLIPVisionConfig, rngs=nnx.Rngs):
		self.config = config
		self.patch_embedding = nnx.Conv(
			in_features=config.num_channels, 
			out_features=config.hidden_size, 
			kernel_size=(config.patch_size, config.patch_size), 
			strides=(config.patch_size, config.patch_size), 
			padding="VALID",
			use_bias=False, 
			rngs=rngs)
		self.class_embedding = nnx.Param(nnx.initializers.normal(stddev=0.02)
			(rngs.params(),(1, 1, config.hidden_size)))
		num_patches = (config.image_size//config.patch_size)**2
		self.position_embedding = nnx.Param(nnx.initializers.normal(stddev=0.02)
			(rngs.params(),(num_patches + 1, config.hidden_size)))

	def __call__(self, pixel_values:jnp.ndarray):
		b = pixel_values.shape[0]
		x = self.patch_embedding(pixel_values) # (batch, 7, 7, 768)
		x = jnp.reshape(x,(b, -1, self.config.hidden_size))  # (batch, 49, 768)
		cls_tokens = jnp.broadcast_to(self.class_embedding.get_value(), (b, 1, self.config.hidden_size)) # (1, 1, 768) -> (batch, 1, 768)
		x = jnp.concatenate([cls_tokens, x], axis=1)
		x = x + self.position_embedding.get_value()

		return x

class CLIPVisionAttention(nnx.Module):
	def __init__(self, config:CLIPVisionConfig, rngs:nnx.Rngs):
		self.mha = nnx.MultiHeadAttention(
			num_heads=config.num_attention_heads, 
			in_features=config.hidden_size, 
			qkv_features=config.hidden_size, 
			out_features=config.hidden_size,
			rngs=rngs
			)

	def __call__(self, x:jnp.ndarray):
		return self.mha(inputs_q=x, decode=False)

class CLIPVisionEncoderLayer(nnx.Module):
	def __init__(self, config:CLIPVisionConfig, rngs:nnx.Rngs):
		self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
		self.attn = CLIPVisionAttention(config, rngs)
		self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
		self.fc1 = nnx.Linear(config.hidden_size, config.intermediate_size, rngs=rngs)
		self.fc2 = nnx.Linear(config.intermediate_size, config.hidden_size, rngs=rngs)

	def __call__(self, x:jnp.ndarray):
		x = x + self.attn(self.layer_norm1(x))
		residual = x
		x = self.layer_norm2(x)
		x = self.fc1(x)
		x = nnx.gelu(x)
		x = self.fc2(x)
		x = residual + x

		return x

class CLIPVisionEncoder(nnx.Module):
	def __init__(self, config:CLIPVisionConfig, rngs=nnx.Rngs):
		self.embeddings = CLIPVisionEmbeddings(config, rngs)
		self.layers = nnx.List([CLIPVisionEncoderLayer(config, rngs) for _ in range(config.num_hidden_layers)])
		self.visual_projection = nnx.Linear(config.hidden_size, config.projection_dim, use_bias=False, rngs=rngs)
		self.pre_layrnorm = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
		self.post_layernorm = nnx.LayerNorm(config.hidden_size, epsilon=1e-5, rngs=rngs)
	def __call__(self, pixel_values:jnp.ndarray):
		x = self.embeddings(pixel_values)
		x = self.pre_layrnorm(x)
		for layer in self.layers:
			x = layer(x)
		pooled_x = x[:, 0, :]
		pooled_x = self.post_layernorm(pooled_x)
		image_features = self.visual_projection(pooled_x)

		return image_features

if __name__=='__main__':

	rngs = nnx.Rngs(42)
	
	config = CLIPVisionConfig()
	vision_embeddings = CLIPVisionEmbeddings(config, rngs)
	vision_attention = CLIPVisionAttention(config, rngs)
	vision_encoder_layer = CLIPVisionEncoderLayer(config, rngs)
	vision_encoder_block = CLIPVisionEncoder(config, rngs)

	mock_pixel_values = random.uniform(random.PRNGKey(0), shape=(2,224,224,3), minval=0.0, maxval=1.0)
	
	start_time_vec = time()
	output_tokens = vision_embeddings(mock_pixel_values)
	end_time_vec = time()

	start_time_vac = time()
	vision_attention_output = vision_attention(output_tokens)
	end_time_vac = time()

	start_time_velc = time()
	vision_encoder_layer_output = vision_encoder_layer(output_tokens)
	end_time_velc = time()	

	start_time_vebc = time()
	vision_encoder_block_output = vision_encoder_block(mock_pixel_values)
	end_time_vebc = time()	

	print()
	print("="*20,"Vision Embedding Class","="*20)
	print("Input Images Shape (Batch, Height, Width, Channel):", mock_pixel_values.shape)
	print("Output Tokens Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim):", output_tokens.shape)
	print("Type of Image Embedding:",type(output_tokens))
	print("Took",end_time_vec-start_time_vec,"seconds")
	print()
	print("="*20,"Vision Attention Class","="*20)
	print("Input Tokens Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim):", output_tokens.shape)
	print("Output Attention Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim:", vision_attention_output.shape)
	print("Type of Vision Attention Output :",type(vision_attention_output))
	print("Took",end_time_vac-start_time_vac,"seconds")
	print()
	print("="*20,"Vision Encoder Layer Class","="*20)
	print("Encoder Layer Output Shape:", vision_encoder_layer_output.shape)
	print("Type of encoder layer output:",type(vision_encoder_layer_output))
	print("Took",end_time_velc-start_time_velc,"seconds")
	print()
	print("="*20,"Vision Encoder Class","="*20)
	print("Vision Encoder Output Shape:", vision_encoder_block_output.shape)
	print("Type of Vision Encoder Output:",type(vision_encoder_block_output))
	print("Took",end_time_vebc-start_time_vebc,"seconds")
	print()
	summary_table = nnx.tabulate(vision_encoder_block, mock_pixel_values)
	print(summary_table)

	'''Output:
	==================== Vision Embedding Class ====================
	Input Images Shape (Batch, Height, Width, Channel): (2, 224, 224, 3)
	Output Tokens Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim): (2, 50, 768)
	Type of Image Embedding: <class 'jaxlib._jax.ArrayImpl'>
	Took 0.09098315238952637 seconds

	==================== Vision Attention Class ====================
	Input Tokens Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim): (2, 50, 768)
	Output Attention Shape (Batch, Image Patches(49) + [CLS](1), Hidden_dim: (2, 50, 768)
	Type of Vision Attention Output : <class 'jaxlib._jax.ArrayImpl'>
	Took 0.4039616584777832 seconds

	==================== Vision Encoder Layer Class ====================
	Encoder Layer Output Shape: (2, 50, 768)
	Type of encoder layer output: <class 'jaxlib._jax.ArrayImpl'>
	Took 0.5051116943359375 seconds

	==================== Vision Encoder Class ====================
	Vision Encoder Output Shape: (2, 512)
	Type of Vision Encoder Output: <class 'jaxlib._jax.ArrayImpl'>
	Took 0.4315483570098877 seconds
		
	[Table like Keras model.summary()]
	Total Parameters: 87,847,680 (351.4 MB)
	'''