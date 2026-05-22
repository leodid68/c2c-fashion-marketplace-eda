# C2C Fashion Marketplace — Exploratory Data Analysis

A story-driven EDA of **98,913 users** from a French C2C fashion marketplace
([Kaggle dataset](https://www.kaggle.com/datasets/jmmvutu/ecommerce-users-of-a-french-c2c-fashion-store)).
The analysis builds an activation funnel, defines six behavioural user segments,
explores geographic and gender dynamics, quantifies the engagement multiplier of
the mobile app, and exports a Tableau-ready dataset.

**Headline finding**: 98% of registered users have never sold anything and 95%
have never bought anything. The marketplace is held up by a 0.4% sliver of
"Power Users" — figuring out who they are and where they live is the whole point.

## Portfolio notebooks

Two Quarto notebooks tell the story end-to-end:

1. **[The Ghost Marketplace](https://leodid68.github.io/01-the-ghost-marketplace.html)**
   — Data quality, distributions, activation funnel, six segments, power-user deep dive.
2. **[Geography, Gender & the App Effect](https://leodid68.github.io/02-geography-gender-app.html)**
   — Choropleth of activation by country, gender behavioural splits, the 8× app
   multiplier, social-vs-transactional decoupling, and the Tableau-ready export.

Notebooks live in `portfolio/` as `.qmd` files and render to self-contained HTML.

## Repository layout

```
c2c-fashion-eda/
├── data/                                       # Raw Kaggle CSVs
│   └── 6M-0K-99K.users.dataset.public.csv      # 98,913 users × 27 features
├── eda_c2c_fashion.py                          # Original end-to-end EDA script (matplotlib)
├── outputs/                                    # PNG charts + Tableau-ready CSV
│   ├── 01_distributions.png ... 08_social_network.png
│   └── c2c_fashion_tableau.csv
├── portfolio/                                  # Portfolio notebooks (Quarto + Plotly)
│   ├── _quarto.yml
│   ├── requirements.txt
│   ├── portfolio_theme.py
│   ├── 01-the-ghost-marketplace.qmd
│   └── 02-geography-gender-app.qmd
└── requirements.txt                            # Script dependencies
```

## Reproducing

### Run the original analysis script

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eda_c2c_fashion.py
```

Outputs (8 PNGs + Tableau CSV) land in `outputs/`.

### Re-render the portfolio notebooks

```bash
cd portfolio
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PATH=".venv/bin:$PATH" VIRTUAL_ENV="$(pwd)/.venv" quarto render
```

Requires [Quarto](https://quarto.org/) ≥ 1.4.

## Stack

- **Analysis**: Python 3.12, pandas, numpy, scipy
- **Visualisation**: matplotlib + seaborn (script), Plotly (notebooks)
- **Publishing**: Quarto → self-contained HTML
- **Dashboard target**: Tableau (CSV in `outputs/c2c_fashion_tableau.csv`)
