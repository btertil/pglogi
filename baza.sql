
-- baza dla modeli

-- tabela z danymi modeli
-- drop table if exists dl_models_results;
create table dl_models_results (
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
            else 11
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
                else 11
            end order by id
        ) time_diff
    from
        dl_models_results m
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
            else 11
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


-- models per run_id
select run_id, count(*) ile, avg(test_accuracy) from v_dl_models_runs group by 1 order by avg(test_accuracy) desc;


-- speed & updates
select
  run_id,
  count(*) ile,
  avg(time_diff) avg_time_diff,
  avg((20090 * epochs) / batch_size) avg_updates,
  avg(((20090 * epochs) / batch_size) / (extract(epoch from time_diff)))  avg_speed,
  avg(test_accuracy) avg_test_accuracy
from v_dl_models_runs
where id not in (34, 35)
group by 1
order by 5 desc;


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

commit;


-- reporting
-- ----------

-- basic selects
select * from v_dl_models_runs;
select * from v_dl_models_runs where run_id >= 7 and patience is null and id > 1016;
select * from v_dl_models_performance limit 50;

-- best models per run
select * from v_dl_models_best_per_run;


-- current progress
select count(*) ile, max(id) max_id, max(test_accuracy) best from dl_models;
