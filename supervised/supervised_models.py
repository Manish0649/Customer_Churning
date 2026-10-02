"""Compatibility entry point for the reorganized supervised workflow.

Run: python -m supervised.supervised_models --models random_forest
"""
from .supervised_learning import main, run_supervised_analysis, train_model

__all__ = ["run_supervised_analysis", "train_model"]

if __name__ == "__main__":
    main()
