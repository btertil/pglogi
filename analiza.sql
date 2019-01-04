

-- najlepsze
select * from dl_models order by test_accuracy desc;

-- analiza
select batch_size, count(*) ile, avg(test_accuracy) from dl_models group by batch_size order by avg(test_accuracy) desc;
select lr, count(*) ile, avg(test_accuracy) from dl_models where lr <> 0.1 group by lr order by avg(test_accuracy) desc;

select epochs, count(*) ile, avg(test_accuracy) from dl_models group by epochs order by avg(test_accuracy) desc;

-- to warto tylko jeśłi inne hyperparameters, beta, beta2 itd
select
  lr,
  batch_size,
  epochs,
  count(*) ile,
  avg(test_accuracy)
from dl_models
group by
  lr,
  batch_size,
  epochs
order by 5 desc;


-- czas szkolenia modelu
-- czas wykonywania na podstawie entered w bazie (uwaga, wirtualka, czas był rozjechany w service ntpd restart spowodował przesunięcie w 1 miejscu)

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
            when python_model_id >= 668 and python_model_id < 1494 then 7
            when python_model_id >= 1494 and python_model_id < 1731 then 8
            when python_model_id >= 1731 and python_model_id < 1886 then 9
            else 10
        end run_id,
        entered -  lag(entered, 1) over (partition by
            case
                when python_model_id < 73 then 1
                when python_model_id >= 73 and python_model_id < 101 then 2
                when python_model_id >= 101 and python_model_id < 250 then 3
                when python_model_id >= 250 and python_model_id < 310 then 4
                when python_model_id >= 250 and python_model_id < 460 then 5
                when python_model_id >= 460 and python_model_id < 668 then 6
                when python_model_id >= 668 and python_model_id < 1494 then 7
                when python_model_id >= 1494 and python_model_id < 1731 then 8
                when python_model_id >= 1731 and python_model_id < 1886 then 9
            else 10
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
            when python_model_id >= 668 and python_model_id < 1494 then 7
            when python_model_id >= 1494 and python_model_id < 1731 then 8
            when python_model_id >= 1731 and python_model_id < 1886 then 9
            else 10
        end,
        entered;

select * from v_dl_models_runs;

-- view v_dl_models_performance
create or replace view v_dl_models_performance as
  select * from v_dl_models_runs order by test_accuracy desc;

select * from v_dl_models_performance;


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
   patience
from (
    select
      r.*,
      max(test_accuracy) over (partition by run_id) ma
    from v_dl_models_runs r
    ) s
where
  ma = test_accuracy
order by
  test_accuracy desc;

-- updates

update dl_models set patience = 10 where patience is null and python_model_id >= 1494;

delete from dl_models where python_model_id = 640;

select '2 days 2 seconds' :: interval;

select '2 days 2 seconds' :: interval interval_time, extract(epoch from '2 days 2 seconds' :: interval) epoch_time;


select count(*) ile from dl_models;

commit;
