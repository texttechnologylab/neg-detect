from setuptools import setup, find_packages

setup(
    name='neg-detect',  # Replace with your desired package name
    version='0.1.1',  # Initial version of your package
    description='A BERT-based inference module for negation detection (cue, scope) -> planning to add focus and event in the near future',  # Short description
    long_description=open('README.md').read(),  # Make sure README.md exists
    long_description_content_type='text/markdown',  # Format of your README
    author='Leon Hammerla',  # Replace with your name
    author_email='8i40irjqx@mozmail.com',  # Replace with your email
    url='https://github.com/LeonHammerla/neg-detect',  # Replace with your repo URL (if available)
    packages=find_packages(),
    install_requires=[
        'torch>=1.0.0',
        'transformers>=4.0.0'
    ],  # Add other dependencies if required
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace license if different
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)