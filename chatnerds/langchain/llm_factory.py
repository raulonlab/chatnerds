import io
from contextlib import redirect_stdout
import logging
from typing import Any, Callable, Dict, Optional, Tuple
from langchain.llms.base import LLM as LLMBase
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
)


class LLMFactory:
    config: Dict[str, Any] = {}
    callback: Optional[Callable[[str], None]] = None

    def __init__(
        self, config: Dict[str, Any], callback: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.callback = callback

    def get_embedding_function(self) -> Embeddings:
        embeddings_config = {**self.config["embeddings"]}
        if embeddings_config["model_name"].startswith("hkunlp/") or embeddings_config[
            "model_name"
        ].startswith("BAAI/"):
            provider_class = HuggingFaceInstructEmbeddings
        else:
            provider_class = HuggingFaceEmbeddings

        # capture module stdout and log them as debug level
        trap_stdout = io.StringIO()
        with redirect_stdout(trap_stdout):
            provider_instance = provider_class(**embeddings_config)

        logging.debug(trap_stdout.getvalue())

        return provider_instance

    def get_summarize_model(self) -> Tuple[LLMBase, str]:
        selected_summarize_model = self.config.get("summarize", {}).get("model", None)

        return self.get_model(selected_model=selected_summarize_model, is_chat=False)

    def get_model(
        self, selected_model: Optional[str] = None, is_chat: Optional[bool] = True
    ) -> Tuple[LLMBase, str]:
        # class CallbackHandler(BaseCallbackHandler):
        #     def on_llm_new_token(self, token: str, **kwargs) -> None:
        #         callback(token)

        # callbacks = [CallbackHandler()] if callback else None

        device_type = self.config.get("device_type", "cpu")

        selected_model, selected_model_config = self.get_selected_model_and_config(
            selected_model
        )

        llm_provider = selected_model_config.pop("provider", "llamacpp")
        prompt_type = selected_model_config.pop("prompt_type", None)

        match llm_provider:
            case "ollama":
                # standarize model, model_id, model_name
                if selected_model_config["model"] is None:
                    selected_model_config["model"] = selected_model_config.pop(
                        "model_id", None
                    ) or selected_model_config.pop("model_name", None)

                if is_chat:
                    from langchain_community.chat_models.ollama import ChatOllama

                    llm = ChatOllama(**selected_model_config)
                else:
                    from langchain_community.llms.ollama import Ollama

                    llm = Ollama(**selected_model_config)
            case "openai":
                # standarize model, model_id, model_name
                if selected_model_config["model_name"] is None:
                    selected_model_config["model_name"] = selected_model_config.pop(
                        "model_id", None
                    ) or selected_model_config.pop("model", None)

                if is_chat:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(**selected_model_config)
                else:
                    from langchain_openai import OpenAI

                    llm = OpenAI(**selected_model_config)
            case "llamacpp":
                llm = self.load_llm_from_config(
                    device_type=device_type, **selected_model_config
                )
            case _:
                raise ValueError(
                    f"Uknown LLM provider '{llm_provider}'. Please use 'ollama', 'openai' or 'llamacpp'."
                )

        return llm, prompt_type

    def get_selected_model_and_config(
        self, selected_model: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        if not selected_model:
            selected_model = self.config.get("default_model", None)

        if not selected_model:
            raise ValueError("Key 'default_model' not found in config file")

        if isinstance(selected_model, list) and len(selected_model) > 0:
            selected_model = str(selected_model[0]).strip()
        elif isinstance(selected_model, str) and len(selected_model.strip()) > 0:
            selected_model = selected_model.strip()
        else:
            raise ValueError(
                f"Invalid value '{selected_model}' in config's key 'default_model'"
            )

        # Load selected LLM config from the list of models in config file
        selected_model_config = self.config.get("models", {}).get(selected_model, None)

        # Or load selected LLM config from the root of config file
        if not selected_model_config:
            selected_model_config = self.config.get(selected_model, None)

        if not selected_model_config:
            raise ValueError(
                f"Model preset '{selected_model}' not found in config file"
            )
        elif not isinstance(selected_model_config, dict):
            raise ValueError(
                f"Model preset '{selected_model}' is not a valid dictionary in config file"
            )

        # Return a cloned dictionary of the selected config
        return selected_model, dict(selected_model_config)

    @classmethod
    def load_llm_from_config(
        cls, model_id, model_basename=None, device_type=None, **kwargs
    ) -> LLMBase:
        if model_basename is not None:
            model_basename_lowered = model_basename.lower()

            # Use LamaCpp / HuggingFacePipeline for GGUF/GGML quantized models
            if ".gguf" in model_basename_lowered:
                llm = cls._load_quantized_model_gguf_ggml(
                    model_id, model_basename, device_type, **kwargs
                )
                return llm
            elif ".ggml" in model_basename_lowered:
                model, tokenizer = cls._load_quantized_model_gguf_ggml(
                    model_id, model_basename, device_type, **kwargs
                )
            # Use AutoGPTQForCausalLM for GPTQ quantized models
            else:
                model, tokenizer = cls._load_quantized_model_qptq(
                    model_id, model_basename, device_type, **kwargs
                )
        else:
            # Use LlamaForCausalLM (cpu / mps devices) or AutoModelForCausalLM (cuda devices)
            model, tokenizer = cls._load_full_model(
                model_id, model_basename, device_type, **kwargs
            )

        from transformers import GenerationConfig, pipeline

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
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

        from langchain_community.llms import HuggingFacePipeline

        llm = HuggingFacePipeline(pipeline=pipe)
        logging.debug("Local LLM Loaded")

        return llm

    @staticmethod
    def _load_quantized_model_gguf_ggml(
        model_id, model_basename, device_type, **kwargs
    ):
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
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=model_id,
                filename=model_basename,
                resume_download=True,
                # cache_dir=MODELS_PATH,
            )

            if device_type.lower() == "mps":
                # kwargs["n_gpu_layers"] = 1
                kwargs["f16_kv"] = (
                    True  # MUST set to True, otherwise you will run into problem after a couple of calls
                )
                logging.info("Using MPS for GGUF/GGML quantized models")

            from langchain_community.llms.llamacpp import LlamaCpp

            llm = LlamaCpp(model_path=model_path, **kwargs)

            return llm
        except Exception as err:
            logging.error(f"Error loading GGUF/GGML model: {err}")
            if "ggml" in model_basename:
                logging.info(
                    "If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead"
                )
            raise err

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

        use_safetensors = False
        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            use_safetensors = True
            model_basename = model_basename.replace(".safetensors", "")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        from auto_gptq import AutoGPTQForCausalLM

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

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaForCausalLM,
            LlamaTokenizer,
        )

        if device_type.lower() in ["mps", "cpu"]:
            logging.info("Using LlamaTokenizer")
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id
            )  # cache_dir="./models/"
            model = LlamaForCausalLM.from_pretrained(
                model_id, device_map="auto", **kwargs
            )  # cache_dir="./models/"
        else:
            logging.info("Using AutoModelForCausalLM for full models")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True
            )  # cache_dir="./models/"
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
