from lightning.pytorch.profilers import PyTorchProfiler


class DefaultPyTorchProfiler(PyTorchProfiler):
    def __init__(
        self,
        profile_memory=True,
        with_flops=True,
        use_cuda=True,
        **kwargs,
    ):
        super().__init__(
            profile_memory=profile_memory,
            with_flops=with_flops,
            use_cuda=use_cuda,
            **kwargs,
        )
