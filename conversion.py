import jax.numpy as jnp
from flax import nnx
from modeling import *
from flax import serialization, traverse_util

rngs = nnx.Rngs(42)
text_config = CLIPTextConfig()
vision_config = CLIPVisionConfig()

clip_model = CLIPModel(text_config, vision_config, rngs)

mock_input_ids = jax.random.randint(jax.random.PRNGKey(1), shape=(2,77), minval=0, maxval=text_config.vocab_size)
mock_pixel_values = jax.random.uniform(jax.random.PRNGKey(2), shape=(3,224,224,3), minval=0.0, maxval=1.0)

img_logits, txt_logits, img_feats, txt_feats = clip_model(mock_input_ids, mock_pixel_values)

# Loading CLIP Weights from official hf repo
with open("flax_model.msgpack","rb") as f:
	raw_bytes = f.read()

# Extracting names of layers in weights file
msgpack_params = serialization.msgpack_restore(raw_bytes)
flat_msgpack_params = traverse_util.flatten_dict(msgpack_params, keep_empty_nodes=False)

# Extracting names of layers from the Model Architecture
model_state = nnx.state(clip_model)
flat_model_state = nnx.to_flat_state(model_state)

nnx_dict = {".".join(str(x) for x in path): var for path, var in flat_model_state}
aligned_params={}
matched_keys = set()

for hf_tuple_path, array in  flat_msgpack_params.items():
	hf_path = ".".join(str(x) for x in hf_tuple_path)
	my_path = hf_path
	
	if "encoder." in my_path:
		my_path = my_path.replace("encoder.","")
	if "self_attn." in my_path:
		my_path = my_path.replace("self_attn.","attn.mha.")
	if "_proj." in my_path:
		my_path = my_path.replace("_proj","")
	if "mha.q" in my_path:
		my_path = my_path.replace("mha.q","mha.query")
	if "mha.k" in my_path:
		my_path = my_path.replace("mha.k","mha.key")
	if "mha.v" in my_path:
		my_path = my_path.replace("mha.v","mha.value")
	if "mlp.fc" in my_path:
		my_path = my_path.replace("mlp.fc","fc")
	if "text_projection" in my_path:
		my_path = my_path.replace("text_projection","text_model.text_projection")
	if "visual_projection" in my_path:
		my_path = my_path.replace("visual_projection","vision_model.visual_projection")
	if "position_embedding" in my_path:
		my_path = "".join(my_path.rsplit(".embedding",1))
		
# Reshaping 2D kernels (512, 512) to 3D kernels (512, 8, 64) compatible with multiple
# heads initialized in nnx.MultiHeadAttention(in_features, num_heads, head_features)
	if "text_model." in my_path:
		if any(x in my_path for x in["query.kernel","key.kernel","value.kernel"]):
			array = jnp.reshape(array, (512, 8, 64))

# Out projection shape is (8, 64, 512)
		elif "out.kernel" in my_path:
			array = jnp.reshape(array, (8, 64, 512))		

# Reshaping 1D biases (512,) to 2D biases (8, 64) compatible with multiple
# heads initialized in nnx.MultiHeadAttention(in_features, num_heads, head_features)
		elif any(x in my_path for x in["query.bias","key.bias","value.bias"]):
			array = jnp.reshape(array, (8, 64))

	if "vision_model." in my_path:
		if any(x in my_path for x in["query.kernel","key.kernel","value.kernel"]):
			array = jnp.reshape(array, (768, 12, 64))

		elif "out.kernel" in my_path:
			array = jnp.reshape(array, (12, 64, 768))	

		elif any(x in my_path for x in["query.bias","key.bias","value.bias"]):
			array = jnp.reshape(array, (12, 64))

		if "embeddings.class_embedding" in my_path:
			array = jnp.reshape(array, (1, 1, 768))
	

	if my_path in nnx_dict:
		expected_shape = nnx_dict[my_path].get_value().shape
		if array.shape == expected_shape:
			tuple_key = tuple(int(x) if x.isdigit() else x for x in my_path.split("."))
			aligned_params[tuple_key] = nnx.Variable(array)
			matched_keys.add(my_path)
		else:
			print(f"Shape Mismatch on {my_path}: Got {array.shape}, expected {expected_shape}")
	else:
		print(f"Skipped / Unmapped HF track: {hf_path} -> Attempted as: {my_path}")

all_model_keys = set(nnx_dict.keys())
uninitialized_keys = all_model_keys-matched_keys

if uninitialized_keys:
	print(f"WARNING: {len(uninitialized_keys)} parameters in model were not loaded." )

new_state = nnx.State.from_flat_path(aligned_params)
nnx.update(clip_model, new_state)

print("\n" + "="*45)
print("SUCCESS: Weights Loaded!")
print("="*45)







