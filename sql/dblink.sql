


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

alter table dl_models alter column id type SERIAL not null primary key;


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



-- uwaga, potrzebuję mieć id jako serial a nie int (bo inserty z pytjona !!)



-- naprawa dl_models
-- lepsza metoda

begin transaction;

create table dl_models_backup as select * from dl_models;

drop table dl_models cascade;

create table dl_models (
    id SERIAL not null primary key,
    python_model_id INT,
    lr double precision,
    batch_size INT,
    epochs INT,
    train_loss double precision,
    train_accuracy double precision,
    valid_loss double precision,
    valid_accuracy double precision,
    test_loss double precision,
    test_accuracy double precision,
    machine_id varchar(90),
    architecture varchar(250),
    optimizer varchar(90),
    patience int,
    entered timestamp not null default now()
);

insert into dl_models (python_model_id, lr, batch_size, epochs, test_loss, test_accuracy, patience)
    select python_model_id, lr, batch_size, epochs, test_loss, test_accuracy, patience from dl_models_backup;

commit;

select * from dl_models limit 10;

rollback;

*/