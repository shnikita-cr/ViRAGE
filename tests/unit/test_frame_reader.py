from __future__ import annotations

from src.services.data import read_dataframe


def test_read_dataframe_falls_back_to_cp1251_for_csv(tmp_path):
    data_path = tmp_path / "cp1251.csv"
    data_path.write_bytes("Метод,Значение\nА,1\nБ,2\n".encode("cp1251"))

    df = read_dataframe(data_path)

    assert list(df.columns) == ["Метод", "Значение"]
    assert df["Метод"].tolist() == ["А", "Б"]
    assert df.attrs["source_encoding"] == "cp1251"
    assert df.attrs["source_format"] == "csv"


def test_read_dataframe_records_utf8_sig_encoding_for_csv(tmp_path):
    data_path = tmp_path / "utf8_sig.csv"
    data_path.write_bytes("Name,Value\nA,1\n".encode("utf-8-sig"))

    df = read_dataframe(data_path)

    assert list(df.columns) == ["Name", "Value"]
    # pandas can read UTF-8 BOM with plain utf-8 as well. The important part is
    # that a successful encoding is recorded for downstream diagnostics.
    assert df.attrs["source_encoding"] in {"utf-8", "utf-8-sig"}
