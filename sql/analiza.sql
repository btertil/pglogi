

-- najlepsze
select * from dl_models order by test_accuracy desc;

-- analiza
select batch_size, count(*) ile, avg(test_accuracy) from dl_models group by batch_size order by avg(test_accuracy) desc;
select lr, count(*) ile, avg(test_accuracy) from dl_models where lr <> 0.1 group by lr order by avg(test_accuracy) desc;

-- zbyt wiele epochs
-- select epochs, count(*) ile, avg(test_accuracy) from dl_models group by 1 order by avg(test_accuracy) desc;

-- statystyki dla run_id
select
    run_id,
    count(*) ile,
    min(test_accuracy),
    max(test_accuracy),
    avg(test_accuracy)
from v_dl_models_runs
group by 1
order by
    avg(test_accuracy) desc;

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


-- updates


select '2 days 2 seconds' :: interval;

select '2 days 2 seconds' :: interval interval_time, extract(epoch from '2 days 2 seconds' :: interval) epoch_time;


select count(*) ile, max(id) max_id, max(test_accuracy) best from dl_models;



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

-- last models
select * from v_dl_models_performance order by id desc;

commit;

select * from v_dl_models_performance where id=1;


-- xgboost models from ubuntulaptop (sql_alchemy via pandas):

-- best xgboost model
select
    index model_id,
    base_score,
    lr learning_rate,
    reg_alpha,
    reg_lambda,
    test_accuracy
from
    (select
        x.*,
        max(test_accuracy) over () from xgb_models x
    ) s
where test_accuracy = max;

-- delete from dl_models where id = 1518;