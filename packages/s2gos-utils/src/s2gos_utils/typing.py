import os

from upath import UPath

#: Path-like type annotation for all path parameters
#: Supports local paths, remote URLs, and UPath objects
PathLike = UPath | os.PathLike
