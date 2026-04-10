# pyright: strict
import os
from contextlib import contextmanager
from pathlib import Path

from IPython.core.getipython import get_ipython
from IPython.display import display  # type: ignore
from matplotlib.figure import Figure


def project_path() -> Path:
    return Path(__file__).parent.parent.parent


def disp[T](value: T) -> T:
    """Displays a value in IPython and passes it along unmodified."""
    display(value)
    return value


@contextmanager
def _with_source_date_epoch():
    """Temporarily sets `SOURCE_DATA_EPOCH=0`.

    Normally, matplotlib embeds a timestamp into saved PDFs, producing pointless diffs every time somebody reruns a notebook.
    Running `fig.savefig` inside this context manager prevents that.
    """
    original = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        yield
    finally:
        if original is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = original


def savefig(fig: Figure, path: Path | str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _with_source_date_epoch():
        fig.savefig(p, bbox_inches="tight")  # type: ignore


def init():
    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "3")
    os.chdir(project_path())
