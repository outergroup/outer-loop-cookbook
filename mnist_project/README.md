# MNIST and GPs and BO

This is an example project using outerloop.

Things you can do:

- Run an individual MNIST experiment using the best known config

```bash
python run_mnist.py
```

- Fit a GP to results, run cross-validation

```bash
python run_cross_validation.py --model-name [todo] --sweep-name [todo]
```

- Run a distributed hyperparameter sweep on an MNIST experiment:

```bash
python run_sweep.py --sweep-name mnist1
```

<!--
- Run a single run of Bayesian Optimization

```bash
python run_bo.py --model-name [todo] --sweep-name [todo]
```

- Run performance test on GP

```python
python run_gp_performance_test.py --model-name [todo] --sweep-name [todo]
```

-->
