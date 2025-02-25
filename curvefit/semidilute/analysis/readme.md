# Semidilute Systems Analysis

For semidilute systems, we first extract the structure factor by fitting the $\gamma=100$ curve. Here the $S(Q)$ was modeled using KAN trained on the hard core Yukawa colloid dataset, and the form factor was arbitrarily selected from a fuzzy ball model. 

Once the structure factor is extracted, the next step is to calculate the form factor using $P(Q) = \frac{I(Q)}{S(Q)}$. According to [this paper](https://arxiv.org/abs/2406.00311), the partial correlation function can be extracted by solving a linear least squares equation. 

## Usage
To use the provided code, follow these steps:

1. Ensure you have the required package installed:
    ```bash
    pip install git+https://github.com/KindXiaoming/pykan
    ```

2. Open the `fit_sq_kan_gamma_100.ipynb` notebook to run the curve fitting code for the $\gamma=100$ curve.

3. Open the `cvsans.ipynb` notebook to run the contrast variation data analysis.

4. Open the `figure_summary_fit.ipynb` notebook to view the summarized result figures.