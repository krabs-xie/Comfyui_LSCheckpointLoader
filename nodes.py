import comfy.sd
import comfy.utils
import folder_paths
import logging
from comfy import model_management
from comfy import model_detection
from comfy import clip_vision
import torch
from comfy.sd import VAE
from comfy.sd import CLIP
import yaml
import safetensors.torch
from typing import Optional, Dict
import hashlib
from Comfyui_LSCheckpointLoader.sd_cache import model_cache
import comfy.model_patcher

class LSCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                       "The CLIP model used for encoding text prompts.", 
                       "The VAE model used for encoding and decoding images to and from latent space.")
    
    FUNCTION = "load_checkpoint_dict"

    CATEGORY = "RubberSheet"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint_dict(self, ckpt_name):

        def load_checkpoint_guess_config(ckpt_path, ckpt_name,output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
            sd = load_torch_file(ckpt_path,ckpt_name)

            try:
                out = load_state_dict_guess_config(sd, output_vae, output_clip, output_clipvision, embedding_directory, output_model, model_options, te_model_options=te_model_options)
            except Exception as e:
                logging.error(f"Error load_state_dict_guess_config model {ckpt_name}: {e}")

            if out is None:
                raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))
            return out
        
        def get_string_hash(input_string):
            # 使用 SHA-256 生成哈希值
            hash_object = hashlib.sha256(input_string.encode())
            return hash_object.hexdigest()  # 返回十六进制的哈希值

        def load_torch_file(ckpt, model_name, safe_load=False, device=None):
            cache_sd = model_cache.get_item(ckpt, 'sd')
            # 如果模型已经加载，直接返回已加载的模型
            if cache_sd:
                return cache_sd

            print("-------------------------------------------------------")
            if device is None:
                device = torch.device("cpu")

            try:
                # 判断文件格式，加载模型
                if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
                    sd = safetensors.torch.load_file(ckpt, device=device.type)
                else:
                    if safe_load:
                        if not 'weights_only' in torch.load.__code__.co_varnames:
                            logging.warning("Warning: torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                            safe_load = False
                    if safe_load:
                        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
                    else:
                        pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
                    if "global_step" in pl_sd:
                        logging.debug(f"Global Step: {pl_sd['global_step']}")
                    if "state_dict" in pl_sd:
                        sd = pl_sd["state_dict"]
                    else:
                        sd = pl_sd
                # save the references of Tensor to cache
                model_cache.cache_sd(ckpt, sd)
                logging.info(f"Model {model_name} loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading model {ckpt}: {e}")
                sd = None
            return sd

        def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None, output_model=True, model_options={}, te_model_options={}):
            clip = None
            clipvision = None
            vae = None
            model = None
            model_patcher = None
            # sd 是一个字典类型
            print(type(sd))  # <class 'dict'>
            diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
            logging.warning(f"diffusion_model_prefix: {diffusion_model_prefix}")
            parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
            logging.warning(f"parameters: {parameters}")
            weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)
            logging.warning(f"weight_dtype: {weight_dtype}")
            load_device = model_management.get_torch_device()
            logging.warning(f"load_device: {load_device}")

            model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
            logging.warning(f"model_config: {model_config}")

            if model_config is None:
                return None

            unet_weight_dtype = list(model_config.supported_inference_dtypes)
            if weight_dtype is not None and model_config.scaled_fp8 is None:
                unet_weight_dtype.append(weight_dtype)

            model_config.custom_operations = model_options.get("custom_operations", None)
            unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))

            if unet_dtype is None:
                unet_dtype = model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype)

            manual_cast_dtype = model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
            model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

            if model_config.clip_vision_prefix is not None:
                if output_clipvision:
                    clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

            if output_model:
                inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
                model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
                model.load_model_weights(sd, diffusion_model_prefix)

            if output_vae:
                vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
                vae_sd = model_config.process_vae_state_dict(vae_sd)
                vae = VAE(sd=vae_sd)

            if output_clip:
                clip_target = model_config.clip_target(state_dict=sd)
                if clip_target is not None:
                    clip_sd = model_config.process_clip_state_dict(sd)
                    if len(clip_sd) > 0:
                        parameters = comfy.utils.calculate_parameters(clip_sd)
                        clip = CLIP(clip_target, embedding_directory=embedding_directory, tokenizer_data=clip_sd, parameters=parameters, model_options=te_model_options)
                        m, u = clip.load_sd(clip_sd, full_model=True)
                        if len(m) > 0:
                            m_filter = list(filter(lambda a: ".logit_scale" not in a and ".transformer.text_projection.weight" not in a, m))
                            if len(m_filter) > 0:
                                logging.warning("clip missing: {}".format(m))
                            else:
                                logging.debug("clip missing: {}".format(m))

                        if len(u) > 0:
                            logging.debug("clip unexpected {}:".format(u))
                    else:
                        logging.warning("no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.")

            left_over = sd.keys()
            if len(left_over) > 0:
                logging.debug("left over keys: {}".format(left_over))

            if output_model:
                model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=model_management.unet_offload_device())
                if inital_load_device != torch.device("cpu"):
                    logging.info("loaded straight to GPU")
                    model_management.load_models_gpu([model_patcher], force_full_load=True)

            return (model_patcher, clip, vae, clipvision)

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path,ckpt_name, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        print(out)
        return out[:3]


