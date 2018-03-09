Delphi – The runner of oracles
=============
A tool to help the development and testing of new machine learning models

Setup Development Environment
-----------------------------

###Create conda environment
```bash
$ conda create -n delphi-env python=3.5
$ source activate delphi-env
```
###Install dependencies
```bash
$ pip install -r requirements.txt --src $CONDA_PREFIX
$ pip install -r dev-requirements.txt
```

###Running the test suite
```bash
$ pytest tests/
```

Documentation
-------------
* [Todo List](./doc/TODO.md)

Install
=======

To install run
```text
$ python setup.py install
```


