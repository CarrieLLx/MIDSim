from setuptools import setup, find_packages

# Core dependencies
required_packages = [
    "aiofiles>=24.1.0",
    "asyncpg>=0.30.0",
    "dataclasses_json>=0.6.7",
    "grpcio>=1.70.0",
    "httpx>=0.28.1",
    "loguru>=0.6.0",
    "matplotlib>=3.10.3",
    "networkx>=3.4.2",
    "numpy>=2.2.5",
    "pandas>=2.2.3",
    "protobuf>=6.30.2",
    "pydantic>=2.11.4",
    "pyvis>=0.3.2",
    "seaborn>=0.13.2",
    "setuptools>=75.6.0",
    "tqdm>=4.67.1",
    "faiss-cpu>=1.9.0",
    "pyyaml",
    "openai",
    "volcengine-python-sdk",
    "requests",
    "tomli>=2.0; python_version < '3.11'",
]

# Optional dependencies for tuning
tune_extras = [
    "mlflow",
    "torch",
    "peft>=0.15.1",
    "transformers",
    "trl>=0.16.0",
    "vllm",
]

setup(
    name='YuLan-OneSim',
    version='1.0.0',
    author='Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen',
    author_email='wanglei154@ruc.edu.cn',
    description='MIDSim: Multi-channel Information Diffusion Simulator, built on YuLan-OneSim',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RUC-GSAI/YuLan-OneSim',
    project_urls={
        'Paper': 'https://arxiv.org/abs/2505.07581',
    },
    packages=find_packages(where="src", include=["onesim", "onesim.*"]),
    package_dir={'': 'src'},
    py_modules=['main', 'app'],
    include_package_data=True,
    package_data={"onesim.agent": ["prompts/*.txt"]},
    install_requires=required_packages,
    extras_require={'tune': tune_extras},
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Sociology',
    ],
    entry_points={
        'console_scripts': [
            'midsim=main:cli_entry_point',
            'yulan-onesim-cli=main:cli_entry_point',
        ],
    },
)
