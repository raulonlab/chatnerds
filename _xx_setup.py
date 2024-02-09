from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = [line.strip() for line in f if ('selenium' not in line and 'webdriver' not in line)]

name = "chatnerds"

setup(
    name=name,
    version="0.0.0",
    description="Chat with youtube videos, podcasts and your private documents using AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raul On Lab",
    author_email="raulonlab@gmail.com",
    url="https://github.com/raulonlab/{}".format(name),
    license="MIT",
    packages=[name],
    package_data={name: ["config.yml"]},
    entry_points={
        "console_scripts": [
            f"{name} = {name}.cli:app",
        ],
    },
    install_requires=[
        "python-dotenv",
        "typing-extensions",
        "typer",
        "tqdm",
        "rich",
        "pytubefix",
        "requests",
        "feedparser",
        "optimum",
        "deepmerge",
        "pyyaml",

        # LLM dependencies
        "langchain",
        "langchain-community",
        "chromadb",
        "InstructorEmbedding",
        "sentence-transformers",
        "huggingface_hub",
        "transformers",
        "auto-gptq",
        "accelerate",
        "llama-cpp-python",
        "openai",
        "openai-whisper",

        # Document Loaders
        "pdfminer.six",
        "extract-msg",
        # "pandoc",
        # "pypandoc",
        "unstructured",
    ],
    extras_require={
        "gptq": [
            "auto-gptq>=0.2.2,<0.3.0",
            # "auto-gptq>=0.4.2,<0.5.0",
            # "optimum>=1.12.0",
        ],
        "tests": [
            "pytest",
        ],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="{}langchain chroma ai llm".format(name),
)
