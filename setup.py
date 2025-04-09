from setuptools import setup, find_packages

setup(
    name="keployrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "python-multipart",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.3",
        "langchain-community>=0.0.10",
        "faiss-cpu",
        "streamlit>=1.30.0",
        "watchdog",
        "pandas>=2.0.0",
        "pydantic"
    ],
)