from torch import nn


class ProjectionWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        in_features: int,
        embed_dim: int,
        out_features: int | None = None,
        project_kwargs: list[str] | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.module = module
        self.embedding = nn.Linear(in_features, embed_dim)
        self.projection = nn.Linear(embed_dim, out_features)

        self.project_kwargs = project_kwargs or []

    def forward(self, *args, **kwargs):
        args = [self.embedding(arg) for arg in args]

        for kwarg in self.project_kwargs:
            kwargs[kwarg] = self.embedding(kwargs[kwarg])

        x = self.module(*args, **kwargs)

        x = (self.projection(x[0]),) + x[1:]
        return x
