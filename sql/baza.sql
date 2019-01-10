
-- baza dla modeli

-- tabela z danymi modeli
-- drop table if exists dl_models_results;
create table dl_models_results (
    id SERIAL not null primary key,
    python_model_id INT,
    lr double precision,
    batch_size INT,
    epochs INT,
    optimizer varchar(90),
    training_time interval,
    train_loss double precision,
    train_accuracy double precision,
    valid_loss double precision,
    valid_accuracy double precision,
    test_loss double precision,
    test_accuracy double precision,
    machine_id varchar(90),
    architecture varchar(250),
    patience int,
    entered timestamp not null default now()
);


-- view v_dl_models_runs
drop view if exists v_dl_models_runs cascade;
create or replace view v_dl_models_runs as
    select
        m.*,
        case
            when python_model_id < 73 then 1
            when python_model_id >= 73 and python_model_id < 101 then 2
            when python_model_id >= 101 and python_model_id < 250 then 3
            when python_model_id >= 250 and python_model_id < 310 then 4
            when python_model_id >= 250 and python_model_id < 460 then 5
            when python_model_id >= 460 and python_model_id < 668 then 6
            when python_model_id >= 668 and python_model_id < 818 then 7
            when python_model_id >= 818 and python_model_id < 1494 then 700
            when python_model_id >= 1494 and python_model_id < 1731 then 8
            when python_model_id >= 1731 and python_model_id < 1886 then 9
            when python_model_id >= 1886 and python_model_id < 1948 then 10
            when python_model_id >= 1948 and python_model_id < 2430 then 11
            else 12
        end run_id,
        entered -  lag(entered, 1) over (partition by
            case
                when python_model_id < 73 then 1
                when python_model_id >= 73 and python_model_id < 101 then 2
                when python_model_id >= 101 and python_model_id < 250 then 3
                when python_model_id >= 250 and python_model_id < 310 then 4
                when python_model_id >= 250 and python_model_id < 460 then 5
                when python_model_id >= 460 and python_model_id < 668 then 6
                when python_model_id >= 668 and python_model_id < 818 then 7
                when python_model_id >= 818 and python_model_id < 1494 then 700
                when python_model_id >= 1494 and python_model_id < 1731 then 8
                when python_model_id >= 1731 and python_model_id < 1886 then 9
                when python_model_id >= 1886 and python_model_id < 1948 then 10
                when python_model_id >= 1948 and python_model_id < 2430 then 11
                else 12
            end order by id
        ) time_diff
    from
        dl_models m
    where 1 = 1
    order by
        case
            when python_model_id < 73 then 1
            when python_model_id >= 73 and python_model_id < 101 then 2
            when python_model_id >= 101 and python_model_id < 250 then 3
            when python_model_id >= 250 and python_model_id < 310 then 4
            when python_model_id >= 250 and python_model_id < 460 then 5
            when python_model_id >= 460 and python_model_id < 668 then 6
            when python_model_id >= 668 and python_model_id < 818 then 7
            when python_model_id >= 818 and python_model_id < 1494 then 700
            when python_model_id >= 1494 and python_model_id < 1731 then 8
            when python_model_id >= 1731 and python_model_id < 1886 then 9
            when python_model_id >= 1886 and python_model_id < 1948 then 10
            when python_model_id >= 1948 and python_model_id < 2430 then 11
            else 12
        end,
        entered;

select * from v_dl_models_runs;

-- view v_dl_models_performance
create or replace view v_dl_models_performance as
  select
    r.*,
    rank() over (order by test_accuracy desc) model_rank
from v_dl_models_runs r
order by
    test_accuracy desc;

select * from v_dl_models_performance limit 25;



-- best model per run_id
drop view if exists v_dl_models_best_per_run;
create or replace view v_dl_models_best_per_run as
    select
       id,
       python_model_id,
       lr,
       batch_size,
       epochs,
       test_loss,
       test_accuracy,
       entered,
       run_id,
       time_diff training_time,
       model_rank,
       patience
    from (
        select
          p.*,
          max(test_accuracy) over (partition by run_id) ma
        from v_dl_models_performance p
        ) s
    where
      ma = test_accuracy
    order by
      test_accuracy desc;

select * from v_dl_models_best_per_run;


-- benchmark Linux CPU i7 3rdGen vs Win GPU + i7 8thGen
-- ----------------------------------------------------

-- python_model_id: od 2237 (nieparzyste, batchsize 8162) <--- LINUX
-- python_model_id: od 2237 (nieparzyste, batchsize 16384) <--- WINDOWS

-- benchmark GPU vs CPU
drop view if exists v_benchmark;
create view v_benchmark as
    select
        case
            when batch_size = 8192 then 'Linux CPU'
            else 'Windows GPU'
        end machine,
        count(*) ile,
        avg(training_time) avg_training_time
    from v_dl_models_runs where id >= 1520
    group by 1
    order by 3;

select * from v_benchmark;

commit;




-- XGBOOST Models
-- ---------------

-- best xgboost model

-- tabela z danymi modeli
-- drop table if exists xgb_models_results;
create table xgb_models_results (
    id SERIAL not null primary key,
    python_model_id INT,
    learning_rate double precision,
    max_depth int,
    n_estimators int,
    base_score double precision,
    reg_alpha double precision,
    reg_lambda double precision,
    min_child_weight int,
    max_leaf_nodes int,
    gamma double precision,
    max_delta_step double precision,
    subsample double precision,
    colsample_bytree double precision,
    scale_pos_weight double precision,
    eval_metric varchar(250),
    training_time interval,
    train_loss double precision,
    train_accuracy double precision,
    valid_loss double precision,
    valid_accuracy double precision,
    test_loss double precision,
    test_accuracy double precision,
    machine_id varchar(90),
    entered timestamp not null default now()
);


