"""
Shared ModuleResult dataclass used by all 7 modules.

This single definition ensures consistent attribute access across the pipeline.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModuleResult:
    """
    Standardised result object returned by every module.

    Attributes
    ----------
    ticker : str
        The asset ticker this result is for.
    module_name : str
        Human-readable module identifier.
    normalised_score : float
        The module's output score on a 0-100 scale (higher = better quality).
    confidence : float
        Data completeness / reliability factor in [0, 1].
    components : dict
        All sub-calculations (fully transparent).
    warnings : list[str]
        Any data quality or methodology warnings.
    metadata : dict
        Optional extra information.
    """
    ticker: str = ""
    module_name: str = ""
    normalised_score: float = 50.0
    confidence: float = 1.0
    components: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Legacy aliases so that modules written with different field names
    # still work transparently.
    @property
    def score(self) -> float:
        return self.normalised_score

    @score.setter
    def score(self, v: float):
        self.normalised_score = v

    @property
    def metrics(self) -> dict:
        return self.components

    @metrics.setter
    def metrics(self, v: dict):
        self.components = v

    @property
    def details(self) -> dict:
        return self.metadata

    @details.setter
    def details(self, v: dict):
        self.metadata = v
