"""Abstract base class for all faithfulness evaluators."""

from __future__ import annotations

import abc
from typing import Any


class BaseEvaluator(abc.ABC):
    """Abstract base for a faithfulness metric evaluator.

    Each subclass wraps one metric (FactCC, FENICE, MiniCheck, …).
    All evaluators follow the same contract:

    * ``__init__`` accepts at least ``device`` (str) and ``model_path``
      (optional local path override, for offline environments).
    * ``load()`` loads the model/tokenizer into memory (called lazily).
    * ``evaluate(records, source_col, summary_col)`` returns a list of
      dicts, one per record, containing the original record fields plus
      a ``<metric_name>_score`` float and optional extra fields.

    Notes
    -----
    ``model_path`` should always be respected over any default HuggingFace
    hub identifier so that the pipeline works fully offline (e.g. on a
    company cluster with no internet access).
    """

    #: Short name used for the output column, e.g. "factcc"
    metric_name: str = ""

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.batch_size = batch_size
        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load models into memory. Called once before ``evaluate``."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str = "input",
        summary_col: str = "abstract_sum",
    ) -> list[dict[str, Any]]:
        """Score a list of records and return them with a score column appended.

        Parameters
        ----------
        records:
            Raw dataset records (dicts).
        source_col:
            Key in each record that holds the source document.
        summary_col:
            Key in each record that holds the summary to evaluate.

        Returns
        -------
        list[dict]
            Each dict is the original record + ``{metric_name}_score`` (float,
            0–1 range where higher = more faithful) and any metric-specific
            extra fields.
        """
        if not self._loaded:
            self.load()
        return self._evaluate(records, source_col, summary_col)

    # ------------------------------------------------------------------
    # Abstract internals
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _load(self) -> None:
        """Load model weights / tokenizer."""

    @abc.abstractmethod
    def _evaluate(
        self,
        records: list[dict[str, Any]],
        source_col: str,
        summary_col: str,
    ) -> list[dict[str, Any]]:
        """Implement the actual scoring logic."""
