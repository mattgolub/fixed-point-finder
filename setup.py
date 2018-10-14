from setuptools import setup, find_packages

setup(
    name = 'fixed-point-finder',
    version = '1.0.0',
    url = 'https://github.com/mattgolub/fixed-point-finder.git',
    author = 'Matt Golub',
    author_email = 'mgolub@stanford.edu',
    description = 'A Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks',
    long_description=open('README.md').read()
    packages = find_packages(),
    install_requires = [
        'recurrent-whisperer',
        'numpy >= 1.15.2', 
        'scipy >= 1.1.0', 
        'scikit-learn >=  0.20.0',
        'matplotlib >= 2.2.3',
        'pyyaml >= 3.13'],
    dependency_links=['http://github.com/mattgolub/recurrent-whisperer/tarball/master']
    extras_require={
        'tf': ['tensorflow>=1.10.0'],
        'tf_gpu': ['tensorflow-gpu>=1.10.0'],},
    license='Apache 2.0',
)
