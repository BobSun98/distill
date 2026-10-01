"""验证指标的平台期计数；只保存少量状态，不引入 trainer/callback 框架。"""


class PlateauMonitor:
    def __init__(self, settings):
        self.settings = settings
        self.best_value = None
        self.best_step = None
        self.reference_value = None
        self.bad_checks = 0

    def update(self, value, step):
        # 最佳 checkpoint 保留真实最低值，不因 min_delta 忽略小幅改善。
        new_best = self.best_value is None or value < self.best_value
        if new_best:
            self.best_value, self.best_step = value, step

        # 平台期与最近一次有效改善比较，小幅改善可以累积到 min_delta。
        meaningful = (self.reference_value is None
                      or (value < self.reference_value
                          and value <= self.reference_value - self.settings["min_delta"]))
        if meaningful:
            self.reference_value = value
            self.bad_checks = 0
        elif step >= self.settings["min_steps"]:
            self.bad_checks += 1
        if step < self.settings["min_steps"]:
            self.bad_checks = 0

        return {"metric": self.settings["metric"], "value": value,
                "new_best": new_best, "best_value": self.best_value, "best_step": self.best_step,
                "bad_checks": self.bad_checks,
                "should_stop": self.bad_checks >= self.settings["patience"]}
