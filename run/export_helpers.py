from pathlib import Path
from typing import Optional, Union
import pandas as pd
from pandas import ExcelWriter


def export_excel(
    df: pd.DataFrame,
    *,
    path: Union[str, Path, None] = None,
    writer: Optional[ExcelWriter] = None,
    sheet_name: Optional[str] = None,
    label: str = "",
) -> None:
    """
    Skriver en DataFrame til Excel og logger hva som ble skrevet.
    - Enten til én fil via ``path``, eller til et ark i en eksisterende ``ExcelWriter`` via
      ``writer`` + ``sheet_name``.
    - Sørger for at mappen til filen eksisterer når ``path`` brukes.
    - Logger til konsoll: label (hvis angitt), filsti, df.shape, kolonner og head(5).
    """
    if writer is None and path is None:
        raise ValueError("Må enten gi path, eller writer + sheet_name")
    if writer is not None and sheet_name is None:
        raise ValueError("Må oppgi sheet_name når writer brukes")

    file_path: Optional[Path] = None

    if writer is None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(file_path, index=False)
    else:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        file_path = Path(getattr(writer, "path", "")) if getattr(writer, "path", None) else None

    if not label:
        label = sheet_name or (file_path.stem if file_path is not None else "")

    print("[export_excel]" + (f" {label}" if label else ""))
    if file_path is not None:
        print(f"  Fil: {file_path}")
    elif writer is not None:
        maybe = getattr(writer, "path", None)
        if maybe:
            print(f"  Fil: {maybe}")
    print(f"  Shape: {df.shape}")
    print(f"  Kolonner: {list(df.columns)}")
    print("  head(5):")
    print(df.head(5).to_string(index=False))
