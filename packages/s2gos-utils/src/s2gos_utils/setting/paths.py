from dynaconf.utils.boxing import Box, DynaBox

from ..io import PathRef


def to_pathref(path_setting: DynaBox | dict | str) -> PathRef:
    if isinstance(path_setting, str):
        # Simple string path
        return PathRef(str(path_setting))

    ps = path_setting.copy()
    if isinstance(path_setting, (DynaBox, Box)):
        ps = ps.to_dict()

    return PathRef(**ps)
