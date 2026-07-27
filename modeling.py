import jax
from flax import nnx
import jax.numpy as jnp
import jax.random as random
from time import time
from text_encoder import *
from vision_encoder import *

class CLIPModel(nnx.Module):
	def __init__(self, text_config:CLIPTextConfig, vision_config:CLIPVisionConfig, rngs:nnx.Rngs):
		self.text_model = CLIPTextEncoder(text_config, rngs)
		self.vision_model = CLIPVisionEncoder(vision_config, rngs)
		self.logit_scale = nnx.Param(jnp.array(2.6592))

	def __call__(self, input_ids:jnp.ndarray, pixel_values:jnp.ndarray):
		'''
		   input_ids : (Batch, 77)
		   pixel_values : (Batch, 224, 224, 3)
		'''
		text_features = self.text_model(input_ids)
		image_features = self.vision_model(pixel_values)

		text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
		image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)

		# Dot Product Similarity (Batch Vision, 512) x (512, Batch Text) -> (Batch Vision, Batch Text)
		logits_per_image = jnp.matmul(image_features, text_features.T)  
		scale = jnp.exp(self.logit_scale.get_value())
		logits_per_image = logits_per_image * scale
		logits_per_text = logits_per_image.T

		return logits_per_image, logits_per_text, image_features, text_features

if __name__ =="__main__":
	
	rngs = nnx.Rngs(42)
	text_config = CLIPTextConfig()
	vision_config = CLIPVisionConfig()

	clip_model = CLIPModel(text_config, vision_config, rngs)

	mock_input_ids = jax.random.randint(jax.random.PRNGKey(1), shape=(2,77), minval=0, maxval=text_config.vocab_size)
	mock_pixel_values = jax.random.uniform(jax.random.PRNGKey(2), shape=(3,224,224,3), minval=0.0, maxval=1.0)

	start_time = time()
	img_logits, txt_logits, img_feats, txt_feats = clip_model(mock_input_ids, mock_pixel_values)
	end_time = time()

	print("="*20,"Unified CLIP Model","="*20)
	print("Image Features Shape :", img_feats.shape)
	print("Text Features Shape :", txt_feats.shape)
	print("Logits Per Image Matrix Shape", img_logits.shape)
	print("Logits Per Text Matrix Shape", txt_logits.shape)
	print("Took",end_time-start_time,"seconds")
	print()
	print(nnx.tabulate(clip_model, mock_input_ids, mock_pixel_values))