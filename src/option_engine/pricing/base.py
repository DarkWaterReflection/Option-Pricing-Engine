"""Common pricing abstractions.

``PricingModel`` is the strategy interface every model implements; ``PricingResult``
is the structured return type that carries the price plus model-specific
diagnostics (standard error for Monte Carlo, step count for the lattice, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from option_engine.instruments import ExerciseStyle, MarketData, OptionContract


@dataclass(frozen=True, slots=True)
class PricingResult:
    """Outcome of a pricing call.

    Attributes
    ----------
    price:
        The model's present value of the option.
    model:
        Human-readable model name.
    std_error:
        Monte Carlo standard error of the estimate (``None`` for deterministic
        models).
    confidence_interval:
        Optional ``(low, high)`` 95% CI for stochastic estimators.
    diagnostics:
        Free-form model metadata (steps, paths, variance-reduction flags, ...).
    """

    price: float
    model: str
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.price)


class PricingModel(ABC):
    """Strategy interface for option pricing.

    Subclasses implement :meth:`price`. The contract is deliberately narrow so
    that sensitivity analysis and benchmarking can treat every model identically.
    """

    #: Display name; overridden by subclasses.
    name: str = "abstract"

    #: Exercise styles the model can value.
    supported_styles: frozenset[ExerciseStyle] = frozenset({ExerciseStyle.EUROPEAN})

    def supports(self, contract: OptionContract) -> bool:
        """Whether this model can price ``contract``'s exercise style."""
        return contract.exercise in self.supported_styles

    def _check_supported(self, contract: OptionContract) -> None:
        if not self.supports(contract):
            raise NotImplementedError(
                f"{self.name} does not support {contract.exercise.value} exercise"
            )

    @abstractmethod
    def price(self, contract: OptionContract, market: MarketData) -> PricingResult:
        """Return the present value of ``contract`` under ``market``."""
        raise NotImplementedError

    def value(self, contract: OptionContract, market: MarketData) -> float:
        """Convenience wrapper returning the scalar price."""
        return self.price(contract, market).price
