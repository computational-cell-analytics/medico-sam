def get_medico_sam_model(
    model_type, device=None, checkpoint_path=None, use_sam_med2d=False, use_sam3d=False, encoder_adapter=False, **kwargs
):
    model_kwargs = {
        "model_type": model_type,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }

    assert (use_sam_med2d + use_sam3d) < 2, "Please use either of 'use_sam_med2d' or 'use_sam3d'."

    if use_sam_med2d:
        from medico_sam.models.sam_med2d.util import get_sam_med2d_model
        _fetch_model = get_sam_med2d_model
        model_kwargs["encoder_adapter"] = encoder_adapter

    else:
        if use_sam3d:
            from micro_sam.sam_3d_wrapper import get_3d_sam_model
            _fetch_model = get_3d_sam_model

        else:
            from micro_sam.util import get_sam_model
            _fetch_model = get_sam_model

        for k, v in kwargs.items():
            model_kwargs[k] = v

    return _fetch_model(**model_kwargs)
