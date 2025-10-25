[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17442082.svg)](https://doi.org/10.5281/zenodo.17442082)
# Launch-Day Diffusion: Tracking Hacker News Impact on GitHub Stars for AI Tools

**Preprint:** https://doi.org/10.5281/zenodo.17442082  
© 2025 Obada Kraishan. Text/figures: CC BY 4.0. Code: MIT.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author**: Obada Kraishan, Texas Tech University

## 🎯 Key Findings

Our analysis of 138 repository launches (2024-2025) reveals:
- **Immediate Impact**: Repositories gain an average of 121 stars within 24 hours, 189 stars within 48 hours, and 289 stars within a week of HN exposure
- **Timing Matters**: The difference between optimal and suboptimal posting hours is ~200 stars
- **Best Window**: 12-17 UTC consistently outperforms other time slots
- **Show HN Paradox**: The "Show HN" tag shows no statistical advantage after controlling for other factors

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/obadaKraishan/icwsm-hn-github.git
cd icwsm-hn-github
pip install -r requirements.txt
```

### Environment Setup
```bash
cp .env.example .env
# Add your GitHub token (optional but recommended for higher rate limits)
echo "GH_TOKEN=your_github_token_here" >> .env
```

### Run Complete Pipeline
```bash
# Run all scripts in sequence (takes ~5 minutes)
make demo

# Or run individual components
python src/01_collect_hn_posts.py --start 2024-01-01 --end 2025-01-01
python src/02_extract_github_repos.py
# ... etc
```

## 📊 Pipeline Overview

The system consists of 10 modular scripts that can be run independently or as a complete pipeline:

| Script | Purpose | Output |
|--------|---------|--------|
| `01_collect_hn_posts.py` | Fetch HN posts linking to GitHub | `hn_posts.csv` |
| `02_extract_github_repos.py` | Parse GitHub owner/repo from URLs | `github_repos_from_hn.csv` |
| `03_github_repo_metadata.py` | Fetch repository metadata | `github_repos_metadata.csv` |
| `04_github_stars_timeseries.py` | Get time-stamped star events | `stars_timeseries.csv` |
| `05_build_event_windows.py` | Align timeseries to HN post time | `event_windows.csv` |
| `06_feature_engineering.py` | Create modeling features/labels | `features_labels.csv` |
| `07_event_study_plots.py` | Generate event study curves | Event study figures |
| `08_model_star_growth.py` | Train predictive models | Model predictions & metrics |
| `09_ablation_checks.py` | Run robustness checks | Ablation estimates |
| `10_make_report_txt.py` | Generate summary report | `REPORT.txt` |

## 📁 Project Structure

```
Launch-Day-Diffusion/
├── src/                    # Pipeline scripts
│   ├── figures/           # Generated plots
│   ├── raw/               # Raw API responses (JSONL)
│   ├── processed/         # Clean CSV files
│   └── summaries/         # Text summaries
├── requirements.txt       # Python dependencies
├── config.yaml           # Optional configuration
└── Makefile              # Automation commands
```

## 🔬 Methodology

### Event Study Design
- **Window**: ±7 days around HN post time (t=0)
- **Alignment**: Hourly star counts aggregated to daily totals
- **Labels**: Δ24h, Δ48h, Δ168h star gains

### Models
- **Elastic Net**: Interpretable linear relationships with L1/L2 regularization
- **Gradient Boosting**: Captures non-linear patterns and interactions
- **Validation**: 80/20 train-test split with 5-fold cross-validation

### Statistical Tests
- OLS regression with heteroscedasticity-robust standard errors (HC1)
- Controls for baseline repository characteristics
- Multiple model specifications for robustness

## 📈 Sample Results

### Event Study Curves
<img width="1170" height="750" alt="event_curve_posthour_bins" src="https://github.com/user-attachments/assets/77fe31d9-5f61-4cf3-98ba-c01e6f295d32" />


### Model Performance
| Model | Horizon | MAE | RMSE | R² |
|-------|---------|-----|------|-----|
| Gradient Boosting | 48h | 30.5 | 60.1 | 0.77* |
| Gradient Boosting | 7d | 92.5 | 182.0 | 0.48 |

*Includes day-0 momentum features

## 🔧 Configuration Options

Edit `config.yaml` or use command-line arguments:

```yaml
# config.yaml
start_date: "2024-01-01"
end_date: "2025-01-01"
min_score: 10
query: "llm,gpt,rag,transformers,langchain"
```

## 📚 Cite

If you use this work, please cite the preprint:

**APA**
Kraishan, O. (2025). *Launch-Day Diffusion: Tracking Hacker News Impact on GitHub Stars for AI Tools*. Zenodo. https://doi.org/10.5281/zenodo.17442082

**BibTeX**
@misc{kraishan2025launchday,
  title        = {Launch-Day Diffusion: Tracking Hacker News Impact on GitHub Stars for AI Tools},
  author       = {Kraishan, Obada},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17442082},
  url          = {https://doi.org/10.5281/zenodo.17442082}
}

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for extension:
- Additional social platforms (Reddit, Twitter/X)
- Different software ecosystems (npm, PyPI)
- Enhanced feature engineering
- Real-time monitoring capabilities


```markdown
## 📄 License
- **Code:** MIT (see [LICENSE](LICENSE)).
- **Text, figures, and the preprint PDF:** Creative Commons **CC BY 4.0**.


## 🙏 Acknowledgments

- Hacker News Algolia API for search functionality
- GitHub REST API for repository data
- Texas Tech University College of Media and Communication

## ⚠️ Ethical Considerations

- All data collected via public APIs in compliance with terms of service
- No private or personal information collected
- Rate limiting implemented to respect API quotas
- Results represent associations, not causal effects

## 📧 Contact

Obada Kraishan - [omareikr@ttu.edu](mailto:omareikr@ttu.edu)  
ORCID: [0009-0007-7180-8620](https://orcid.org/0009-0007-7180-8620)  
Website: https://okraishan.com/

---

**Note**: This is a demonstration system submitted to IEEE ICDM 2025 Demo Track. The pipeline is designed to be immediately reproducible (runs in under 5 minutes on standard hardware) and extensible for both academic research and practical applications. We encourage you to build upon this foundation for your own research or tools!
