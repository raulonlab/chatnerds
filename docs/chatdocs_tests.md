# chatdocs tests

All tests made in a Macbook Pro M1 2020 with 16GB RAM.

## Test 1: TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF

Model: [TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF · Hugging Face](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF)

```yaml
llm: ctransformers

ctransformers:
  model: TheBloke/Wizard-Vicuna-13B-Uncensored-GGUF
  model_file: Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf
  model_type: llama
  config:
    context_length: 2048  # Default 1024
    max_new_tokens: 2048  # Default 256
    temperature: 0.1      # 0.1 means more strict with the sources
    # gpu_layers: 0
    # threads: 10  # Use CPU. Number of physical cores your CPU has
    gpu_layers: 100    # 50  # Use GPU
    threads: 1         # 16  # GPU

download: true
```

## Test 2: JohanAR/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGUF (better results)

Model: [JohanAR/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGUF · Hugging Face](https://huggingface.co/JohanAR/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGUF)

```yaml
llm: ctransformers

ctransformers:
  model: JohanAR/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GGUF
  model_file: wizard-vicuna-13b-uncensored-superhot-8k.q4_K_M.gguf
  model_type: llama
  config:
    context_length: 2048  # Default 1024
    max_new_tokens: 2048  # Default 256
    temperature: 0.1
    # gpu_layers: 0
    # threads: 10  # Use CPU. Number of physical cores your CPU has
    gpu_layers: 100    # 50  # Use GPU
    threads: 1         # 16  # GPU

download: true
```
