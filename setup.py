from setuptools import setup, find_packages

setup(
    name="dspy-advanced-prompting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dspy-ai>=2.4.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "pydantic>=2.0.0",
        "jinja2>=3.1.0",
        "pytest>=7.0.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "jsonschema>=4.0.0",
    ],
    python_requires=">=3.8",
)