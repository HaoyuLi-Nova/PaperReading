# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from collections.abc import Mapping, Sequence
from scripts.attention.checks import check_valid_tuple_or_element
from scripts.utils import log


def _validate_single_dim_params(params: Mapping, layer_idx: int, num_dims: int | None) -> dict:
    """
    Helper function to validate NATTEN parameters for a dimensionality profile.

    Args:
        params (Mapping): parameter dict with window_size/window_size_float and other params
        layer_idx (int): layer index for error messages
        num_dims (int | None): 1, 2, 3, or None (for single-profile format)

    Returns:
        dict: validated parameter dict with proper types
    """
    if not isinstance(params, Mapping):
        dim_str = f" ({num_dims}-D)" if num_dims else ""
        raise ValueError(f"Parameters for layer {layer_idx}{dim_str} must be a dict or None, got params={params!r}.")
    is_causal = False if "is_causal" not in params else params["is_causal"]
    if "window_size_float" in params:
        window_size_float = params["window_size_float"]
        if (
            not isinstance(window_size_float, Sequence)
            or len(window_size_float) not in [1, 2, 3]
            or any((not isinstance(x, float) for x in window_size_float))
        ):
            raise ValueError(
                f"'window_size_float' must be a float tuple of size 1, 2, or 3, got window_size_float={window_size_float!r}"
            )
        window_size_float = tuple((k for k in window_size_float))
        num_dims = len(window_size_float)

        def check_stride_dilation(x):
            if isinstance(x, float):
                if 0.0 <= x <= 1.0:
                    return tuple((x for _ in range(num_dims)))
            elif (
                isinstance(x, Sequence)
                and len(x) == num_dims
                and all((isinstance(y, float) and 0.0 <= y <= 1.0 for y in x))
            ):
                return tuple((y for y in x))
            else:
                raise ValueError(f"Invalid natten float parameter: x={x!r}")

        stride_float = 0.0 if "stride_float" not in params else params["stride_float"]
        dilation_float = 0.0 if "dilation_float" not in params else params["dilation_float"]
        stride_float = check_stride_dilation(stride_float)
        dilation_float = check_stride_dilation(dilation_float)
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )
        if any((x in params for x in ["window_size", "stride", "dilation"])):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got params={params!r}."
            )
        return {
            "window_size_float": window_size_float,
            "stride_float": stride_float,
            "dilation_float": dilation_float,
            "is_causal": is_causal,
        }
    elif "window_size" in params:
        window_size = params["window_size"]
        num_dims = len(window_size)
        stride = 1 if "stride" not in params else params["stride"]
        dilation = 1 if "dilation" not in params else params["dilation"]
        if any(("_float" in x for x in params.keys())):
            raise ValueError(
                f"Please either use _float parameters, or integer ones, and not mix the two. Got params={params!r}."
            )
        window_size = check_valid_tuple_or_element(
            window_size, num_dims=num_dims, typename=int, raise_error=True, param_name="window_size"
        )
        stride = check_valid_tuple_or_element(
            stride, num_dims=num_dims, typename=int, raise_error=True, param_name="stride"
        )
        dilation = check_valid_tuple_or_element(
            dilation, num_dims=num_dims, typename=int, raise_error=True, param_name="dilation"
        )
        is_causal = check_valid_tuple_or_element(
            is_causal, num_dims=num_dims, typename=bool, raise_error=True, param_name="is_causal"
        )
        return {"window_size": window_size, "stride": stride, "dilation": dilation, "is_causal": is_causal}
    else:
        raise ValueError(
            f"Sparse parameters for a layer must have key 'window_size' or 'window_size_float', got params={params!r} in layer index {layer_idx}."
        )


def verify_natten_parameter_list(natten_parameter_list: list | None, num_layers: int) -> list | None:
    """
    Converts list of NATTEN parameters into expected types, and assigns defaults to unset
    parameters.
    This needs to be done separately during model initialization, and not forward pass.
    There are no torch operations in this function.

    Args:
        natten_parameter_list (list | None): list of NATTEN parameters. Must be either None, or a
            list of mappings, one for each layer. Each list element must be either None,
            representing no sparsity / masking (full dense attention), or a mapping of NATTEN
            parameters.

            Parameters can be specified directly with integer or float format:
                - 'window_size_float' (required), 'stride_float', 'dilation_float'
                - 'window_size' (required), 'stride', 'dilation'

            Or, parameters can be specified for multiple dimensionality profiles in case of
            mixed-training (i.e. image and video training) using keys "1d", "2d", "3d":
                - Each key maps to either None (dense attention) or a parameter dict

            Integer and float parameters cannot be used together in the same layer!
            Additionally, you can specify 'is_causal'.

            Examples:
            ```
            # 50 percent sparsity along each dimension in a 2-D token layout
            {'window_size_float': (0.5, 0.5)}  # valid

            # 50 percent sparsity along each dimension in a 2-D token layout
            # Maximum dilation along first dimension, no dilation along second dimension
            {'window_size_float': (0.5, 0.5), 'dilation_float': (1.0, 0.0)}  # valid

            # Fixed window size of 8x8, dilation of 2x1.
            # NOTE: requires ALL inputs to be at least 16x8
            {'window_size': (8, 8), 'dilation': (2, 1)}  # valid

            # Multi-profile: different parameters for 2D (images) and 3D (videos)
            {
                "2d": {"window_size_float": (0.5, 0.5)},
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Multi-profile: 2D uses dense attention, 3D uses sparse
            {
                "2d": None,
                "3d": {"window_size_float": (1.0, 0.5, 0.5)}
            }  # valid

            # Invalid:
            {'window_size_float': (0.5, 0.5), 'dilation': (2, 1)}
            ```

        num_layers (int): number of layers in the model. Just used to verify list length.

    Returns:
        output_parameter_list (list | None): verified and type-checked NATTEN parameters, or None if
            no parameters passed.
    """
    if natten_parameter_list is not None:
        parameter_list_out = []
        if not isinstance(natten_parameter_list, Sequence):
            raise ValueError(
                f"Argument 'natten_parameter_list' must be a list or None, got natten_parameter_list={natten_parameter_list!r}."
            )
        if len(natten_parameter_list) != num_layers:
            raise ValueError(
                f"Number of elements in 'natten_parameter_list' must match number of layers in the model, got num_layers={num_layers!r}, len(natten_parameter_list)={len(natten_parameter_list)!r}."
            )
        for i, layer_parameters in enumerate(natten_parameter_list):
            if layer_parameters is None:
                log.debug(f"Layer {i} will use DENSE attention.")
                parameter_list_out.append(None)
                continue
            if not isinstance(layer_parameters, Mapping):
                raise ValueError(
                    f"Sparse parameters for a layer must be a dict or None, got layer_parameters={layer_parameters!r} in layer index {i}."
                )
            dim_keys = {"1d", "2d", "3d"}
            has_dim_keys = any((k in layer_parameters for k in dim_keys))
            if has_dim_keys:
                validated_multi_profile = {}
                for dim_str, dim_int in [("1d", 1), ("2d", 2), ("3d", 3)]:
                    if dim_str in layer_parameters:
                        dim_params = layer_parameters[dim_str]
                        if dim_params is None:
                            validated_multi_profile[dim_int] = None
                        else:
                            validated_multi_profile[dim_int] = _validate_single_dim_params(dim_params, i, dim_int)
            else:
                validated_params = _validate_single_dim_params(layer_parameters, i, None)
                if "window_size_float" in validated_params:
                    num_dims = len(validated_params["window_size_float"])
                else:
                    num_dims = len(validated_params["window_size"])
                validated_multi_profile = {num_dims: validated_params}
            if all((v is None for v in validated_multi_profile.values())):
                log.debug(f"Layer {i} will use DENSE attention (all profiles None).")
                parameter_list_out.append(None)
            else:
                parameter_list_out.append(validated_multi_profile)
                log.info(f"Layer {i} NATTEN parameters: {validated_multi_profile}")
        return parameter_list_out
    return None
