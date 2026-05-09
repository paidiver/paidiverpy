class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))

    def error(self, *args, **kwargs):
        self.messages.append(("error", args, kwargs))

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args, kwargs))


class DummyStep:
    def __init__(self, name: str, step_name: str):
        self.name = name
        self.step_name = step_name

    def to_dict(self):
        return {"name": self.name, "step_name": self.step_name, "mode": "fixed", "params": {"value": 1}}

class FakeCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaled = None

    def scale(self, value):
        self.scaled = value


class FakeClient:
    def __init__(self, cluster):
        self.cluster = cluster
        self.dashboard_link = "http://fake-dashboard"
