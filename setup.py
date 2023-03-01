from setuptools import setup, find_packages

setup()

'''
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='concept-gradients',
    version='0.0.1',
    author='Andrew Bai',
    author_email='andrewbai@cs.ucla.edu',
    description='Implementation of Concept Gradients for feature interpretation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jybai/concept-gradients',
    project_urls = {
        "Bug Tracker": "https://github.com/jybai/concept-gradients/issues"
    },
    license='MIT',
    packages=find_packages(where='src', exclude=['test']),
    install_requires=['argparse', 'numpy', 'pandas', 'matplotlib', 'pillow', 'tqdm', 
                      'torch', 'torchvision', 'torchmetrics'],
)
'''
