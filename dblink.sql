


-- w bazie docelowej logujemy się jako postgres
-- create extenstion dblink

drop table if exists log_data;
create table log_data as
    select * from dblink(
        'hostaddr=192.168.0.101 port=5432 dbname=logs user=bartek password=Aga',
        'select * from akuratne_25k') as dl (
            lo_ip character varying,
            entered timestamp without time zone,
            lo_id integer,
            rank_w_ip bigint,
            avg_id numeric,
            avg_id_rows_current numeric,
            id_parity integer,
            rank_w_ip_parity integer,
            the_same_parity integer,
            target_1 integer,
            target_2 integer
        );


drop table if exists dl_models;
create table dl_models as
    select * from dblink(
        'hostaddr=192.168.0.101 port=5432 dbname=logs user=bartek password=Aga',
        'select * from dl_models') as dl (
            id integer,
            python_model_id integer,
            lr double precision,
            batch_size integer,
            epochs integer,
            test_loss double precision,
            test_accuracy double precision,
            entered timestamp without time zone,
            patience integer
        );


drop table if exists dl_models_performance;
create table dl_models_performance as
    select * from dblink(
        'hostaddr=192.168.0.101 port=5432 dbname=logs user=bartek password=Aga',
        'select * from v_dl_models_performance') as dl (
            id integer,
            python_model_id integer,
            lr double precision,
            batch_size integer,
            epochs integer,
            test_loss double precision,
            test_accuracy double precision,
            entered timestamp without time zone,
            patience integer,
            run_id integer,
            time_diff interval,
            model_rank bigint
        );




-- kolumny i typy z potrzebnych tabel:
select "column_name", data_type from  INFORMATION_SCHEMA.columns where table_schema = 'public' and table_name = 'akuratne_25k';
select "column_name", data_type from  INFORMATION_SCHEMA.columns where table_schema = 'public' and table_name = 'dl_models';
select "column_name", data_type from  INFORMATION_SCHEMA.columns where table_schema = 'public' and table_name = 'v_dl_models_performance';

/*
"""
lo_ip	character varying
entered	timestamp without time zone
lo_id	integer
rank_w_ip	bigint
avg_id	numeric
avg_id_rows_current	numeric
id_parity	integer
rank_w_ip_parity	integer
the_same_parity	integer
target_1	integer
target_2	integer
"""

"""
id	integer
python_model_id	integer
lr	double precision
batch_size	integer
epochs	integer
test_loss	double precision
test_accuracy	double precision
entered	timestamp without time zone
patience	integer
"""

id integer,
python_model_id integer,
lr double precision,
batch_size integer,
epochs integer,
test_loss double precision,
test_accuracy double precision,
entered timestamp without time zone,
patience integer,
run_id integer,
time_diff interval,
model_rank bigint


*/