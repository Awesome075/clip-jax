import jax
import jax.numpy as jnp
from flax import nnx
from modeling import CLIPModel, CLIPTextConfig, CLIPVisionConfig
from conversion import load_flax_weights
from transformers import CLIPImageProcessor, CLIPTokenizer
from PIL import Image
import requests

rngs = nnx.Rngs(42)
text_config = CLIPTextConfig()
vision_config = CLIPVisionConfig()
clip_model = CLIPModel(text_config, vision_config, rngs)
clip_model = load_flax_weights(clip_model, "flax_model.msgpack")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"  # Dog picture
image = Image.open(requests.get(url, stream=True).raw)
text_queries = ["a photo of a dog","a photo of a cat", "a photo of a car"]

text_inputs = tokenizer(text_queries, padding = "max_length",  max_length=77, return_tensors ="np")
input_ids = jnp.array(text_inputs["input_ids"])

image_inputs = image_processor(images=image, return_tensors="np")
pixel_values_hf = image_inputs["pixel_values"] # Shape: (1,3,224,224) (B, C, H, W)
pixel_values = jnp.array(jnp.transpose(pixel_values_hf, (0,2,3,1))) # Shape: (1, 224, 224, 3) (B, H, W, C)

image_logits , text_logits , image_features, text_features = clip_model(input_ids, pixel_values)
probabilities = jax.nn.softmax(image_logits, axis=-1)

print("\n" + "="*20 + " PREDICTIONS " + "="*20)
for label, prob in zip(text_queries, probabilities[0]):
    print(f"Candidate: {label:<20} | Probability: {prob * 100:.2f}%")


