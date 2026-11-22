"""Private integrity boundary for the current PoseBusters check set, without chemistry dependencies.

Every persisted column participates in equality. DuckDB groups nulls together and NaNs together, but keeps
null distinct from NaN. Physical ordinals preserve order only after complete-record conflicts are ruled out.
"""
import os

import pyarrow as pa

from . import tables

PB_KEY = ("protein_id", "ligand_idx", "pocket", "mode")
_READ_PARQUET = "union_by_name = true, hive_partitioning = false"


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(strings) -> str:
    return "[" + ", ".join(_sql_str(s) for s in strings) + "]"


class PoseBustersIntegrityError(ValueError):
    """An ambiguous check set or a record in another protein's part."""

    def __init__(self, category, protein, path, key):
        self.category, self.protein, self.path, self.key = category, protein, path, tuple(key)
        super().__init__(f"PoseBusters integrity [{category}]: protein {protein!r}, part {path!r}, key {self.key!r}. "
                         "Recompute explicitly with pocketvina bust --results RESULTS --protein PROTEIN "
                         "--overwrite, selecting the intended config/select options.")


def _validated(con, raw, parts):
    """Validate the whole source; return SQL for its ordered, unique logical records."""
    keys = ", ".join(PB_KEY)
    columns = ", ".join(tables.POSEBUSTERS.names)
    wrong = con.execute(f"SELECT _owner, _path, {keys} FROM ({raw}) WHERE protein_id IS DISTINCT FROM _owner "
                        "LIMIT 1").fetchall()
    if wrong:
        owner, path, *key = wrong[0]
        raise PoseBustersIntegrityError("wrong-owner", owner, path, key)
    conflict = con.execute(f"SELECT {keys} FROM (SELECT DISTINCT {columns} FROM ({raw})) "
                           f"GROUP BY {keys} HAVING count(*) > 1 LIMIT 1").fetchall()
    if conflict:
        key = conflict[0]
        path = next(path for owner, path in parts if owner == key[0])
        raise PoseBustersIntegrityError("conflicting-records", key[0], path, key)
    return (f"SELECT {columns} FROM ({raw}) QUALIFY row_number() OVER "
            f"(PARTITION BY {keys} ORDER BY _file_pos, _row_pos) = 1 ORDER BY _file_pos, _row_pos")


def source(con, parts):
    """Validate explicit ``(owner_protein_id, path)`` parts in DuckDB and return normalized source SQL.

    Callers discover current, committed parts first. No full records enter Python during validation; each
    diagnostic query fetches at most one offending key. The returned source remains a live Parquet query.
    """
    parts = [(owner, os.fspath(path)) for owner, path in parts]
    files = [path for _, path in parts]
    owners = ", ".join(f"({_sql_str(path)}, {_sql_str(owner)}, {n})" for n, (owner, path) in enumerate(parts))
    sql_types = {pa.string(): "VARCHAR", pa.int8(): "TINYINT", pa.int16(): "SMALLINT", pa.int32(): "INTEGER",
                 pa.bool_(): "BOOLEAN", pa.float32(): "FLOAT"}
    columns = ", ".join(f"CAST(r.{f.name} AS {sql_types[f.type]}) AS {f.name}" for f in tables.POSEBUSTERS)
    raw = (f"SELECT {columns}, f._owner, f._path, f._file_pos, r.file_row_number AS _row_pos "
           f"FROM read_parquet({_sql_list(files)}, {_READ_PARQUET}, filename = true, file_row_number = true) r "
           f"JOIN (VALUES {owners}) f(_path, _owner, _file_pos) ON r.filename = f._path")
    return _validated(con, raw, parts)


def normalize(table: pa.Table, owner: str, path: str) -> pa.Table:
    """Convert a publication candidate to declared types and normalize it, retaining its schema metadata."""
    import duckdb

    schema = tables.POSEBUSTERS.with_metadata(table.schema.metadata)
    table = tables.to_arrow(table, schema)
    con = duckdb.connect()
    try:
        con.register("pv_pb_candidate", table)
        raw = (f"SELECT *, {_sql_str(owner)} AS _owner, {_sql_str(os.fspath(path))} AS _path, "
               "0 AS _file_pos, row_number() OVER () - 1 AS _row_pos FROM pv_pb_candidate")
        _validated(con, raw, [(owner, path)])
        keys = ", ".join(PB_KEY)
        indices = con.execute(f"SELECT min(_row_pos) FROM ({raw}) GROUP BY {keys} ORDER BY min(_row_pos)").fetchall()
        return table.take(pa.array([row[0] for row in indices], type=pa.int64()))
    finally:
        con.close()


def read(parts, columns=None):
    """Read normalized projected records with Arrow's declared pandas dtypes and physical row order."""
    import duckdb

    schema = tables.POSEBUSTERS
    columns = schema.names if columns is None else columns
    selected = pa.schema([schema.field(c) for c in columns])
    con = duckdb.connect()
    try:
        sql = source(con, parts)
        result = con.sql(f"SELECT {', '.join(columns) if columns else '1 AS _empty'} FROM ({sql})")
        # DuckDB 2 renamed the Arrow result method; retain support for existing installations.
        table = result.to_arrow_table() if hasattr(result, "to_arrow_table") else result.arrow()
        if isinstance(table, pa.RecordBatchReader):
            table = table.read_all()
        return (table.cast(selected) if columns else table.select([])).to_pandas()
    finally:
        con.close()
