from setuptools import setup, find_packages


setup(
    name='alphai_delphi',
    description='Delphi: Alpha-I Prototype Environment',
    author='Gabriele Alese, Richard Mason, Daniele Murroni',
    author_email='gabriele.alese@alpha-i.co, richard.mason@alpha-i.co, daniele.murroni@alpha-i.co',
    version='0.5.0',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'numpy',
        'pandas',
        'tables',
        'pandas-market-calendars==0.11',
        'alphai_finance==2.0.0',
        'performance_analysis==2.1.7',
        'dateutils==0.6.6',
        'marshmallow==3.0.0b4',
        'marshmallow-enum==1.4.1',
    ],
    dependency_links=[
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_finance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/performance_analysis/',
        ]
)
