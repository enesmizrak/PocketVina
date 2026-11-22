"""Complete-record equality, ownership and normalization for stored PoseBusters checks."""
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pocketvina.io import pb, tables


def record(**changes):
    return dict(protein_id="protein", ligand_idx=0, ligand_id="ligand", pocket=1, mode=1,
                pb_config="dock", pb_version="test", pb_status="timeout", pb_valid=None, pb_mask=0,
                pb_na_mask=0, **changes)


def arrow(rows):
    return tables.to_arrow(rows, tables.POSEBUSTERS.with_metadata({b"custom": b"unchanged"}))


def test_roundtrip_and_identical_nullable_records(tmp_path):
    import duckdb

    table = arrow([{**record(), "pb_valid": False, "runtime_s": 1.25, "error": "diagnostic\n'",
                    "uniprot": "P12345", "n_heavy_docked": 7, "pb_mask": 3, "failed_checks": "z,a"},
                   {**record(), "ligand_idx": 1, "pocket": -1, "mode": -1},
                   {**record(), "ligand_idx": 2, "mode": -1}])
    duplicate = pa.concat_tables([table.slice(0, 1), table, table.slice(2)])
    normalized = pb.normalize(duplicate, "protein", "candidate.parquet")
    assert normalized.equals(table, check_metadata=True)
    path = tmp_path / "quoted's run=1" / "protein.parquet"
    path.parent.mkdir()
    pq.write_table(duplicate, path, row_group_size=1)
    data = path.read_bytes()
    with duckdb.connect() as con:
        sql = pb.source(con, [("protein", path)])
        result = con.sql(sql)
        got = (result.to_arrow_table() if hasattr(result, "to_arrow_table") else result.arrow())
        if isinstance(got, pa.RecordBatchReader):
            got = got.read_all()
        got = got.cast(table.schema)
        assert got.equals(table, check_metadata=True)
    assert path.read_bytes() == data
    assert pb.read([("protein", path)], []).shape == (3, 0)
    assert pb.read([("protein", path)], ["ligand_idx"]).ligand_idx.tolist() == [0, 1, 2]


@pytest.mark.parametrize("column,value", [("pb_config", "different"), ("pb_version", "different"),
    ("pb_mask", 1), ("pb_na_mask", 1), ("error", "private diagnostic"), ("runtime_s", 1.0),
    ("pb_valid", False), ("failed_checks", "check"), ("ligand_id", "another"), ("uniprot", "P12345"),
    ("n_heavy_docked", 4), ("pb_status", "error")])
@pytest.mark.parametrize("reverse", [False, True])
def test_complete_column_conflicts(tmp_path, column, value, reverse):
    import duckdb

    rows = [record(), {**record(), column: value}]
    table = arrow(rows[::-1] if reverse else rows)
    path = tmp_path / "protein.parquet"
    pq.write_table(table, path, row_group_size=1)
    before = path.read_bytes()
    for call in (lambda: pb.normalize(table, "protein", str(path)),
                 lambda: pb.read([("protein", path)], ["pb_status"])):
        with pytest.raises(pb.PoseBustersIntegrityError) as exc:
            call()
        assert exc.value.category == "conflicting-records"
        assert exc.value.key == ("protein", 0, 1, 1)
        assert str(path) in str(exc.value) and "--overwrite" in str(exc.value)
        assert "private diagnostic" not in str(exc.value)
    with duckdb.connect() as con, pytest.raises(pb.PoseBustersIntegrityError):
        pb.source(con, [("protein", path)])
    assert path.read_bytes() == before


@pytest.mark.parametrize("other", [None, float("nan"), 1.0])
def test_nan_and_null_are_distinct(tmp_path, other):
    # Construct Arrow directly: nullable Python/pandas conversion intentionally treats Python NaN as null.
    table = arrow([record(), record()])
    field = table.schema.field("runtime_s")
    table = table.set_column(table.schema.get_field_index("runtime_s"), field,
                             pa.array([float("nan"), other], type=pa.float32(), from_pandas=False))
    path = tmp_path / "protein.parquet"
    pq.write_table(table, path)
    if other is not None and math.isnan(other):
        assert len(pb.normalize(table, "protein", str(path))) == 1
        assert len(pb.read([("protein", path)])) == 1
    else:
        with pytest.raises(pb.PoseBustersIntegrityError):
            pb.normalize(table, "protein", str(path))
        with pytest.raises(pb.PoseBustersIntegrityError):
            pb.read([("protein", path)])


def test_empty_and_wrong_owner(tmp_path):
    import duckdb

    empty = tables.POSEBUSTERS.with_metadata({b"custom": b"empty"}).empty_table()
    assert pb.normalize(empty, "protein", "empty.parquet").equals(empty, check_metadata=True)
    path = tmp_path / "protein.parquet"
    pq.write_table(empty, path)
    assert pb.read([("protein", path)]).empty
    table = arrow([record()])
    pq.write_table(table, path)
    for call in (lambda: pb.normalize(table, "other", str(path)),
                 lambda: pb.read([("other", path)])):
        with pytest.raises(pb.PoseBustersIntegrityError) as exc:
            call()
        assert exc.value.category == "wrong-owner" and exc.value.protein == "other"
    # An identical correctly owned copy cannot excuse a misplaced row.
    with duckdb.connect() as con, pytest.raises(pb.PoseBustersIntegrityError, match="wrong-owner"):
        pb.source(con, [("protein", path), ("other", path)])


def test_declared_type_conversion_and_part_order(tmp_path):
    # These doubles become the same exact float32 value at the declared-type boundary.
    rows = [record(), record()]
    table = arrow(rows)
    table = table.set_column(table.schema.get_field_index("runtime_s"), "runtime_s",
                             pa.array([1.0, 1.0 + 1e-9], type=pa.float64()))
    normalized = pb.normalize(table, "protein", "candidate.parquet")
    assert len(normalized) == 1 and normalized.schema.field("runtime_s").type == pa.float32()
    first, second = tmp_path / "protein.parquet", tmp_path / "other.parquet"
    pq.write_table(normalized, first)
    tables.write_table([{**record(), "protein_id": "other"}], tables.POSEBUSTERS, second)
    assert pb.read([("other", second), ("protein", first)]).protein_id.tolist() == ["other", "protein"]
