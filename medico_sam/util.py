def get_medico_sam_model(
    model_type, device=None, checkpoint_path=None, use_sam_med2d=False, encoder_adapter=False, **kwargs
):
    model_kwargs = {
        "model_type": model_type,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }

    if use_sam_med2d:
        from medico_sam.models.sam_med2d.util import get_sam_med2d_model
        _fetch_model = get_sam_med2d_model
        model_kwargs["encoder_adapter"] = encoder_adapter

    else:
        from micro_sam.util import get_sam_model
        _fetch_model = get_sam_model

        for k, v in kwargs.items():
            model_kwargs[k] = v

    return _fetch_model(**model_kwargs)
