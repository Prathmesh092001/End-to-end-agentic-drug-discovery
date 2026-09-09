from setuptools import setup, find_packages

setup(
    name="partex_agent",
    version="0.1.0",
    author="Partex AI Research & Platform",
    description="Agentic GenAI pipeline for drug asset intelligence and multi-modal drug discovery",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
)
