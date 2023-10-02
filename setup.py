from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

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
    entry_points={
        "console_scripts": [
            f"{name} = {name}.cli:app",
        ],
    },
    install_requires=[
        "chatdocs==0.2.6",
        "pytube==15.0.0",
        "openai-whisper==20230918",
        "typer>=0.9.0",
        "typing-extensions>=4.4.0,<5.0.0",
        "requests==2.29.0",
        "feedparser==6.0.10",
        "tqdm>=4.64.1,<5.0.0",
        # Temporary fix for version conflicts (using python 3.11?)
        # "numba==0.58.0",
    ],
    extras_require={
        # "gptq": [
        #     "auto-gptq>=0.2.2,<0.3.0",
        # ],
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
    keywords="{} ctransformers transformers langchain chroma ai llm".format(name),
)
