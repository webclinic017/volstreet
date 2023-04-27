from setuptools import setup, find_packages
setup(
    name="autotrading",
    version="0.4.3",
    packages=find_packages(),
    install_requires=[
        "asttokens==2.2.1",
        "backcall==0.2.0",
        "certifi==2022.12.7",
        "colorama==0.4.6",
        "comm==0.1.2",
        "decorator==5.1.1",
        "discord.py>=2.2.2",
        "entrypoints==0.4",
        "executing==1.2.0",
        "idna==3.4",
        "jedi==0.18.2",
        "lxml==4.9.2",
        "matplotlib-inline==0.1.6",
        "nest-asyncio==1.5.6",
        "numpy==1.24.2",
        "packaging==23.0",
        "pandas==1.5.3",
        "parso==0.8.3",
        "pickleshare==0.7.5",
        "platformdirs==2.6.2",
        "prompt-toolkit==3.0.36",
        "psutil==5.9.4",
        "pure-eval==0.2.2",
        "Pygments==2.14.0",
        "pyotp==2.8.0",
        "python-dateutil==2.8.2",
        "pytz==2022.7.1",
        "pywin32==305",
        "pyzmq==25.0.0",
        "requests==2.28.2",
        "scipy==1.10.1",
        "six==1.16.0",
        "smartapi-python==1.3.0",
        "spyder-kernels>=2.2.1",
        "stack-data==0.6.2",
        "tornado==6.2",
        "traitlets==5.8.1",
        "urllib3==1.26.14",
        "wcwidth==0.2.6",
        "websocket-client==1.5.1",
        "xlrd==2.0.1",
        "yfinance==0.2.14",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
