from collections import defaultdict

import matplotlib.pyplot as plt

class plot_context:
    def __init__(self, **kwargs):
        self.params = defaultdict(lambda: None)
        self.params.update(kwargs)

    def __getattr__(self, item):
        return self.params.get(item, None)

    def __enter__(self):
        if self.standalone or self.figsize: plt.figure(figsize=self.figsize)
        if self.subplot: plt.subplot(*self.subplot) if isinstance(self.subplot, tuple) else plt.subplot(self.subplot)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.title is not None: plt.title(self.title)
        if self.suptitle is not None: plt.suptitle(self.suptitle)
        if self.legend is not None: plt.legend(self.legend, loc="best")
        if self.xlabel is not None: plt.xlabel(self.xlabel)
        if self.ylabel is not None: plt.ylabel(self.ylabel)
        if self.xscale is not None: plt.xscale(self.xscale)
        if self.yscale is not None: plt.yscale(self.yscale)
        if self.xticks is not None: plt.xticks(self.xticks, self.xlabels, rotation=self.xticksrotation)
        if self.yticks is not None: plt.yticks(self.yticks, self.ylabels, rotation=self.yticksrotation)
        if self.xlim is not None: plt.xlim(*self.xlim)
        if self.ylim is not None: plt.ylim(*self.ylim)
        if self.colorbar: plt.colorbar()
        if self.grid: plt.grid()
        if self.tight: plt.tight_layout()
        if self.export: self._save_creating_directory(self.export, dpi=self.dpi)
        if self.show: plt.show()
        if any([self.standalone, self.show, self.close, self.export]): plt.close()

    @staticmethod
    def _save_creating_directory(path, dpi=None):
        from pathlib import Path
        import os.path

        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
