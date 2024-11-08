# Task 2 Signal Evaluation

This document provides instructions for running the signal evaluation using `task2_eval.py`.

## Prerequisites

1. Ensure Python and required packages are installed (as specified in `requirements.txt`).
    * Try `pip install -r requirements.txt` in your env
2. Confirm that all dependencies are correctly set up in the environment.
3. Configure HuggingFace API token and login to Huggingface_Hub to access model `SesameStreet/FinRLlama-3.2-3B-Instruct` 

## Configuring Date Range

**Important**: Set the start and end dates directly within the `task2_eval.py` script. Open the script and locate the date variables section to specify the desired evaluation period.

```python
# Set your date range here
START_DATE = "YYYY-MM-DD"
END_DATE = "YYYY-MM-DD"
```

## Configuring Data References

**Important**: Default script references in `task2_eval.py` are `task2_stocks.csv` & `task2_news.csv`, edit to evaluation datasets accordingly