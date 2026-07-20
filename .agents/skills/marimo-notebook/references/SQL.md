There are multiple ways to use SQL in marimo. Under the hood, a SQL cell is just a function call to `marimo.sql`. A cell looks like this: 

```python
@app.cell(hide_code=True)
def _(df, mo):
    grouped = mo.sql(
        f"""
        SELECT category, AVG(value) as mean FROM df GROUP BY category ORDER BY mean;
        """,
        output=False
    )
    return (grouped,)
```

`grouped` is a polars dataframe. By defauly marimo uses DuckDB in memory and can refer to dataframe variables that are in scope. 

This is what the signature is of `mo.sql`: 

```python
def sql(query: str, *, output: bool=True, engine: Optional[DBAPIConnection]=None) -> Any
```

Typically a `sql` call returns a polars dataframe, but the user can configure pandas as an alternative. 

Notice how a query string goes in with SQL and how you can pass a specific database engine. Be aware that different SQL engines may have different SQL dialects. 

## SQLAlchemy

One possible engine is SQLAlchemy. 

```python
import sqlalchemy

# Create an in-memory SQLite database with SQLAlchemy
sqlite_engine = sqlalchemy.create_engine("sqlite:///:memory:")
```

You can also use `SQLModel` with a similar connection string.

## DuckDB 

You can also use DuckDB with a connection string.

```python
import duckdb

# Create a DuckDB connection
duckdb_conn = duckdb.connect("file.db", read_only=True)
```

## PyIceberg 

marimo supports data catalogs as well.

```python
from pyiceberg.catalog.rest import RestCatalog

catalog = RestCatalog(
    name="catalog",
    warehouse="1234567890",
    uri="https://example.com",
    token="my-token",
)
```

