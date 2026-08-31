from pathlib import Path
import pandas as pd

BASE_PATH = Path(__file__).parents[2].resolve()
DEFAULT_DATA_PATH = BASE_PATH / "data" / "datos_credito.parquet"


def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")
