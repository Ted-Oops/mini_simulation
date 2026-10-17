from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MarketConfig:
    agents: int = 5000
    steps: int = 200000
    seed: int = 0
    fundamental_weight: float = 10.0
    chartist_weight: float = 1.2
    noise_weight: float = 1.0
    reference_horizon: int = 200
    fundamental_horizon: float = 200.0
    fundamental_horizon_mode: str = "fixed"
    trait_weights: str = "raw"
    risk_aversion: float = 0.1
    initial_price: float = 300.0
    max_initial_shares: float = 50.0
    max_initial_cash: float = 15000.0
    tick: float = 0.0005
    noise_parameter: float = 1e-4
    noise_interpretation: str = "std"
    fundamental_sigma: float = 1e-3
    fundamental_clock: str = "event"
    order_policy: str = "multiple"
    warmup: str = "noise"
    variance_floor: float = 1e-16
    quantity_mode: str = "continuous"
    grid_rule: str = "sample_then_round"
    burn_in: int = 0
    profile_stride: int = 100
    initialization_steps: int = 0

    def __post_init__(self):
        for name in ("agents", "steps", "reference_horizon", "profile_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if not 0 <= self.burn_in < self.steps - 1:
            raise ValueError("剔除期须非负且至少保留两个事件")
        if isinstance(self.initialization_steps, bool) or not isinstance(self.initialization_steps, int) or self.initialization_steps < 0:
            raise ValueError("初始化事件数必须为非负整数")
        for name in ("fundamental_horizon", "risk_aversion", "initial_price", "tick", "variance_floor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        for name in ("fundamental_weight", "chartist_weight", "noise_weight", "noise_parameter",
                     "fundamental_sigma", "max_initial_shares", "max_initial_cash"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} 必须为有限非负数")
        if self.fundamental_weight + self.chartist_weight + self.noise_weight == 0:
            raise ValueError("至少一种预期成分必须非零")
        for name, choices in (("noise_interpretation", ("std", "variance")),
                              ("fundamental_clock", ("event", "population")),
                              ("order_policy", ("multiple", "replace")),
                              ("fundamental_horizon_mode", ("fixed", "individual")),
                              ("trait_weights", ("raw", "normalized")),
                              ("grid_rule", ("sample_then_round", "discrete_interval")),
                              ("warmup", ("noise", "fundamental", "flat")),
                              ("quantity_mode", ("continuous", "integer"))):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} 不支持当前取值")

    @property
    def noise_std(self):
        return self.noise_parameter if self.noise_interpretation == "std" else math.sqrt(self.noise_parameter)
