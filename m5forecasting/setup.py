from setuptools import setup

setup(name='m5forecasting',
    version='0.1',
    description='Kaggle M5 Forecasting Competition',
    url='',
    author='Rajat Mittal',
    author_email='rajat0502@gmail.com',
    license='MIT',
    packages=['m5forecasting'],
    zip_safe=False,
    include_package_data=True,
    package_data={'': ['data/*.csv']}
    )
