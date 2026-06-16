from dynaconf.utils.boxing import Box, DynaBox

from ..io import PathRef


def to_pathref(path_setting: DynaBox | dict | str) -> PathRef:
    if isinstance(path_setting, (DynaBox, Box)):
        path_setting = path_setting.to_dict()
    return PathRef.model_validate(path_setting)
