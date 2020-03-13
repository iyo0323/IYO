[PostgreSQLにてテーブルやカラムの各種情報を取得するSQL](http://devlights.hatenablog.com/entry/20080226/p1)


[PostgreSQL - Retrieve primary key columns](https://wiki.postgresql.org/wiki/Retrieve_primary_key_columns)


[List all index names, column names and its table name of a PostgreSQL database](https://stackoverflow.com/questions/6777456/list-all-index-names-column-names-and-its-table-name-of-a-postgresql-database )





```sql
SELECT relname AS table_name 
FROM pg_stat_user_tables 
--WHERE relname LIKE 'xxx%' 
ORDER BY relname ASC
```

```sql
SELECT * FROM pg_attribute limit 100
SELECT * FROM pg_stat_user_tables limit 100
SELECT * FROM pg_constraint limit 100
```






```sql
SELECT NEXTVAL('log_seq')
SELECT NEXTVAL FOR db2inst1.log_seq from sysibm.SYSDUMMY1
```


Alter Table Column
```sql
ALTER TABLE t_b_complete_details CHANGE t_b_complate_id t_b_complete_id int(10) UNSIGNED NOT NULL;
```


Select Table Name
```sql
SELECT relname AS table_name FROM pg_stat_user_tables WHERE relname LIKE 'xxx%' ORDER BY relname ASC
```

Select Column Name
```sql
SELECT column_name, data_type, * FROM information_schema.columns WHERE table_name = 'mastercustomerloan';
```

Select Table Comment
```sql
SELECT
	psut.relname AS TABLE_NAME
	,pd.description AS TABLE_COMMENT
FROM
	pg_stat_user_tables psut
	,pg_description pd
WHERE 1 = 1
	--AND psut.relname='テーブル名'
	AND psut.relid=pd.objoid
	AND pd.objsubid=0
```

Select Field Comment
```sql
SELECT
	--pd.objsubid,
	psat.relname AS TABLE_NAME
	,pa.attname AS COLUMN_NAME
	,pd.description AS COLUMN_COMMENT
FROM
	pg_stat_all_tables psat
	,pg_description pd
	,pg_attribute pa
WHERE 1 = 1
	--AND psat.schemaname=(select schemaname from pg_stat_user_tables where relname = 'テーブル名')
	--AND psat.relname='テーブル名'
	AND psat.relid = pd.objoid
	--AND pd.objsubid=0
	AND pd.objoid = pa.attrelid
	AND pd.objsubid = pa.attnum
ORDER BY
	TABLE_NAME ASC,
	pd.objsubid ASC
```

Select Index
```sql
SELECT
  U.usename                AS user_name,
  ns.nspname               AS schema_name,
  idx.indrelid :: REGCLASS AS table_name,
  i.relname                AS index_name,
  idx.indisunique          AS is_unique,
  idx.indisprimary         AS is_primary,
  am.amname                AS index_type,
  idx.indkey,
       ARRAY(
           SELECT pg_get_indexdef(idx.indexrelid, k + 1, TRUE)
           FROM
             generate_subscripts(idx.indkey, 1) AS k
           ORDER BY k
       ) AS index_keys,
  (idx.indexprs IS NOT NULL) OR (idx.indkey::int[] @> array[0]) AS is_functional,
  idx.indpred IS NOT NULL AS is_partial
FROM pg_index AS idx
  JOIN pg_class AS i
    ON i.oid = idx.indexrelid
  JOIN pg_am AS am
    ON i.relam = am.oid
  JOIN pg_namespace AS NS ON i.relnamespace = NS.OID
  JOIN pg_user AS U ON i.relowner = U.usesysid
WHERE NOT nspname LIKE 'pg%'; -- Excluding system tables
```
