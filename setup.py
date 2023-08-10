from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='simple_genetic_algorithm',
    version='0.1',
    description='An intuitive, flexible and efficient implementation of a genetic algorithm in Python',
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='genetic algorithm optimization',
    url='http://github.com/yourname/simple_genetic_algorithm',
    author='Eliel Camargo-Molina, Jonas Wessén',
    author_email='eliel@camargo-molina.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)