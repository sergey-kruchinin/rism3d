import os, sys


class SuppressOutput:
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self.devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = self.devnull
        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = self.devnull

    def __exit__(self, *args):
        self.devnull.close()
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr

