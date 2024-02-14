import logging
from rich import print
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import LlamaCpp, HuggingFacePipeline
from langchain.llms.base import LLM as LLMBase
from huggingface_hub import hf_hub_download
from auto_gptq import AutoGPTQForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    pipeline,
    # TextGenerationPipeline,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)


class LLMFactory:
    config: Dict[str, Any] = {}
    callback: Optional[Callable[[str], None]] = None

    def __init__(self, config: Dict[str, Any], callback: Optional[Callable[[str], None]] = None):
        self.config = config
        self.callback = callback
    
    def get_llm(self) -> LLMBase:
        # class CallbackHandler(BaseCallbackHandler):
        #     def on_llm_new_token(self, token: str, **kwargs) -> None:
        #         callback(token)

        # callbacks = [CallbackHandler()] if callback else None
        device_type = self.config["device_type"] or "cpu"
        selected_llm = self.config["llm"]

        if selected_llm not in self.config:
            raise ValueError(f"LLM config '{selected_llm}' not found in config file.")
        
        selected_llm_config = self.config[selected_llm]
        
        llm = self.load_llm_from_config(device_type=device_type, **selected_llm_config)

        return llm

    @classmethod
    def load_llm_from_config(cls, model_id, model_basename=None, device_type=None, **kwargs) -> LLMBase:
        logging.info(f"Loading Model: '{model_id}', device_type: '{device_type}'")
        logging.info("This action can take a few minutes!")
        # print("kwargs: ")
        # print(kwargs)

        if model_basename is not None:
            model_basename_lowered = model_basename.lower()
    
            # Use LamaCpp / HuggingFacePipeline for GGUF/GGML quantized models
            if ".gguf" in model_basename_lowered:
                llm = cls._load_quantized_model_gguf_ggml(model_id, model_basename, device_type, **kwargs)
                return llm
            elif ".ggml" in model_basename_lowered:
                model, tokenizer = cls._load_quantized_model_gguf_ggml(model_id, model_basename, device_type, **kwargs)
            # Use AutoGPTQForCausalLM for GPTQ quantized models
            else:
                model, tokenizer = cls._load_quantized_model_qptq(model_id, model_basename, device_type, **kwargs)
        else:
            # Use LlamaForCausalLM (cpu / mps devices) or AutoModelForCausalLM (cuda devices) 
            model, tokenizer = cls._load_full_model(model_id, model_basename, device_type, **kwargs)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # print("generation_config:")
        # print(generation_config)
        # print("model:")
        # print(model)

        # Create a pipeline for text generation
        # pipe = TextGenerationPipeline(
        #     task="text-generation",
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=torch.device("cpu"),

            # https://artificialcorner.com/run-mistral7b-quantized-for-free-on-any-computer-2cadc18b45a2
            # max_new_tokens=512,
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.95,
            # top_k=40,
            # repetition_penalty=1.1

            # max_length=kwargs["max_tokens"],     # MAX_NEW_TOKENS,
            # temperature=0.2,
            # top_p=0.95,
            # repetition_penalty=1.15,
            # max_new_tokens=512,
            generation_config=generation_config,
        )
        # pipe.model.to("mps")

        llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("Local LLM Loaded")

        return llm


    @staticmethod
    def _load_quantized_model_gguf_ggml(model_id, model_basename, device_type, **kwargs):
        """
        Load a GGUF/GGML quantized model using LlamaCpp.
        Langchain docs: https://python.langchain.com/docs/integrations/llms/llamacpp#installation-with-metal

        Installation for Metal:
        CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python==0.1.83 --no-cache-dir

        This function attempts to load a GGUF/GGML quantized model using the LlamaCpp library.
        If the model is of type GGML, and newer version of LLAMA-CPP is used which does not support GGML,
        it logs a message indicating that LLAMA-CPP has dropped support for GGML.

        Parameters:
        - model_id (str): The identifier for the model on HuggingFace Hub.
        - model_basename (str): The base name of the model file.
        - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.

        Returns:
        - LlamaCpp: An instance of the LlamaCpp model if successful, otherwise None.

        Notes:
        - The function uses the `hf_hub_download` function to download the model from the HuggingFace Hub.
        - The number of GPU layers is set based on the device type.
        """

        logging.info("Using Llamacpp for GGUF/GGML quantized models")

        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename,
                resume_download=True,
                # cache_dir=MODELS_PATH,
            )

            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1
                kwargs["f16_kv"] = True  # MUST set to True, otherwise you will run into problem after a couple of calls
                logging.info("Using MPS for GGUF/GGML quantized models")

            llm = LlamaCpp(model_path=model_path, **kwargs)

            return llm
        except Exception as err:
            logging.error(f"Error loading GGUF/GGML model: {err}")
            if "ggml" in model_basename:
                logging.info("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
            return None


    @staticmethod
    def _load_quantized_model_gguf_ggml_huggingface(model_id, model_basename, device_type, **kwargs):
        """
        Load a GGUF/GGML quantized model using HuggingFacePipeline.

        pip uninstall llama-cpp-python -y
        CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pipenv install -U llama-cpp-python==0.1.83

        This function attempts to load a GGUF/GGML quantized model using the HuggingFacePipeline library.

        Parameters:
        - model_id (str): The identifier for the model on HuggingFace Hub.
        - model_basename (str): The base name of the model file.
        - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.

        Returns:
        - HuggingFace model: An instance of the HuggingFace model if successful, otherwise None.
        """

        logging.info("Using HuggingFacePipeline.from_model_id")
        # print("kwargs: ")
        # print(kwargs)

        if model_id is None:
            raise ValueError("model_id is required for HuggingFace models")
        
        llm = HuggingFacePipeline.from_model_id(
            task="text-generation",
            model_id=model_id,
            **kwargs,
        )

        return llm


    @staticmethod
    def _load_quantized_model_qptq(model_id, model_basename, device_type, **kwargs):
        """
        Load a GPTQ quantized model using AutoGPTQForCausalLM.

        This function loads a quantized model that ends with GPTQ and may have variations
        of .no-act.order or .safetensors in their HuggingFace repo.

        Parameters:
        - model_id (str): The identifier for the model on HuggingFace Hub.
        - model_basename (str): The base name of the model file.
        - device_type (str): The type of device where the model will run.

        Returns:
        - model (AutoGPTQForCausalLM): The loaded quantized model.
        - tokenizer (AutoTokenizer): The tokenizer associated with the model.

        Notes:
        - The function checks for the ".safetensors" ending in the model_basename and removes it if present.
        """

        # The code supports all huggingface models that ends with GPTQ and have some variation
        # of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")
        # print("kwargs: ")
        # print(kwargs)

        use_safetensors = False
        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            use_safetensors = True
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            local_files_only=True,
            # trust_remote_code=True,
            # device_map="auto",
            device="cpu",
            # use_triton=False,
            # quantize_config=None,
            # offload_folder="offload",
            **kwargs,
        )
        # MODEL_FOR_CAUSAL_LM_MAPPING.register("chatnerds-gptq", model.__class__)

        return model, tokenizer


    @staticmethod
    def _load_full_model(model_id, model_basename, device_type, **kwargs):
        """
        Load a full model using either LlamaTokenizer or AutoModelForCausalLM.

        This function loads a full model based on the specified device type.
        If the device type is 'mps' or 'cpu', it uses LlamaTokenizer and LlamaForCausalLM.
        Otherwise, it uses AutoModelForCausalLM.

        Parameters:
        - model_id (str): The identifier for the model on HuggingFace Hub.
        - model_basename (str): The base name of the model file.
        - device_type (str): The type of device where the model will run.

        Returns:
        - model (Union[LlamaForCausalLM, AutoModelForCausalLM]): The loaded model.
        - tokenizer (Union[LlamaTokenizer, AutoTokenizer]): The tokenizer associated with the model.

        Notes:
        - The function uses the `from_pretrained` method to load both the model and the tokenizer.
        - Additional settings are provided for NVIDIA GPUs, such as loading in 4-bit and setting the compute dtype.
        """

        if device_type.lower() in ["mps", "cpu"]:
            logging.info("Using LlamaTokenizer")
            tokenizer = LlamaTokenizer.from_pretrained(model_id)  # cache_dir="./models/"
            model = LlamaForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                **kwargs
            )  # cache_dir="./models/"
        else:
            logging.info("Using AutoModelForCausalLM for full models")
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)  # cache_dir="./models/"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
                # torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                # offload_folder="offload",
                # cache_dir=MODELS_PATH,
                # trust_remote_code=True, # set these if you are using NVIDIA GPU
                # load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.float16,
                # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
                **kwargs,
            )
            model.tie_weights()
        
        return model, tokenizer
