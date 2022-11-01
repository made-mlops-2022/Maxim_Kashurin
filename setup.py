from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="hw1",
    packages=find_packages(),
    version="0.1.0",
    description="Первая дз по MLOps",
    author="Максим Кашурин",
    entry_points={
        "console_scripts": [
            "ml_example_train = ml_project.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,

)
