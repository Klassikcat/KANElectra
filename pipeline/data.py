from flytekit import task, Resources
import polars as pl


@task(cache=True, cache_version="1.0", limits=Resources(cpu="4", mem="16Gi"))
def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
    return


@task(cache=True, cache_version="1.0", limits=Resources(cpu="1", mem="16Gi"))
def download_data(url: str) -> pl.DataFrame:
    return pl.read_csv(url)
