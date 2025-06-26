from jsonargparse import CLI

from asymdsd.data import create_zarr_ds


def create_zarr_ds_cli():
    CLI(create_zarr_ds)


if __name__ == "__main__":
    create_zarr_ds_cli()
