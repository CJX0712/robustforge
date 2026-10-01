"""防御基类契约。"""

from __future__ import annotations


class Defense:
    name = "defense"
    model = None

    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def get_model(self):
        return self.model
