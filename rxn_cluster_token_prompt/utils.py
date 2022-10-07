from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve


def download_url(url: str, dest_dir: Path, name: Optional[str] = None):
    """Download a file to disk.
    from https://github.com/deepchem/deepchem/blob/master/deepchem/utils/data_utils.py
    Parameters
    ----------
    url: str
      The URL to download from
    dest_dir: str
      The directory to save the file in
    name: str
      The file name to save it as.  If omitted, it will try to extract a file name from the URL
    """
    if name is None:
        name = url
        if "?" in name:
            name = name[: name.find("?")]
        if "/" in name:
            name = name[name.rfind("/") + 1 :]
    dest_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest_dir / name)
