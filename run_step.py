# run_step.py
# (this is run inside the image that is used by the KPO in Airflow)

import importlib
import json
import os
import typing
from typing import Any

from pydantic import BaseModel

INPUT_PREFIX = "STEP_INPUT_"


class _XComEncoder(json.JSONEncoder):
    """Serialize types that json.dump can't handle natively."""

    def default(self, obj):
        if hasattr(obj, "model_dump"):  # Pydantic models (e.g. PathRef)
            return obj.model_dump()
        return str(obj)


def resolve_function(module_name: str, qualname: str):
    module = importlib.import_module(module_name)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


_SCALAR_TYPES = (int, float, bool)


def _pydantic_type(hint) -> type[BaseModel] | None:
    """Return the Pydantic model class from a hint, including Optional[Model]."""
    if isinstance(hint, type) and issubclass(hint, BaseModel):
        return hint
    args = typing.get_args(hint)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if (
            len(non_none) == 1
            and isinstance(non_none[0], type)
            and issubclass(non_none[0], BaseModel)
        ):
            return non_none[0]
    return None


def coerce_inputs(func, inputs: dict[str, Any]) -> dict[str, Any]:
    """Cast inputs to the types declared in func's signature.

    Airflow renders all Jinja {{ params.* }} as strings, so numeric params
    arrive as str even when declared as float/int in the process function.
    Pydantic models (e.g. PathRef) arrive as dicts after XCom round-trip.
    We use the function's type hints (with Annotated stripped) to coerce them.
    """
    try:
        # include_extras=False strips Annotated[X, ...] → X
        hints = typing.get_type_hints(func, include_extras=False)
    except Exception:
        return inputs

    coerced = {}
    for key, value in inputs.items():
        hint = hints.get(key)
        if value is None:
            coerced[key] = value
        elif hint in _SCALAR_TYPES and isinstance(value, str):
            coerced[key] = hint(value)
        elif (model_cls := _pydantic_type(hint)) and not isinstance(value, model_cls):
            coerced[key] = model_cls(value)
        else:
            coerced[key] = value
    return coerced


def main(
    *,
    func_module: str,
    func_qualname: str,
    inputs: dict[str, Any],
    output_keys: list[str] | None = None,
):
    func = resolve_function(func_module, func_qualname)
    inputs = coerce_inputs(func, inputs)

    result = func(**inputs)

    if output_keys:
        if isinstance(result, tuple):
            output = dict(zip(output_keys, result))
        else:
            output = {output_keys[0]: result}
    else:
        output = {"return_value": result}

    # using env variables to allow for easy testing.
    XCOM_DIR = os.environ.get("AIRFLOW_XCOM_DIR", "/airflow/xcom")
    XCOM_FILE = os.path.join(XCOM_DIR, "return.json")

    os.makedirs(XCOM_DIR, exist_ok=True)
    with open(XCOM_FILE, "w") as f:
        json.dump(output, f, cls=_XComEncoder)


if __name__ == "__main__":  # pragma: no cover
    import sys

    payload = json.loads(sys.argv[1])
    main(**payload)