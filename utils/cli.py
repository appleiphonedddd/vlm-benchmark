import argparse


class CaseInsensitiveChoice:
    """argparse type that matches registry keys regardless of the casing typed"""

    def __init__(self, registry):
        self.choices = sorted(registry)

    def __call__(self, value):
        if value.lower() not in self.choices:
            raise argparse.ArgumentTypeError(
                f"invalid choice: {value!r} (choose from {', '.join(self.choices)})")
        return value.lower()
