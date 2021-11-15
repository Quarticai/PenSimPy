import pandas as pd
from pensimpy.examples.helper import get_num_batches, set_labels, DATE_TIME, GRANULARITY, env


def run(
    start_datetime: str = '2021-01-01 00:00:00',
    end_datetime: str = '2021-01-31 00:00:00'
) -> None:

    num_batches = get_num_batches(start_datetime, end_datetime)
    dfs = []
    for i in range(num_batches):
        df, peni_yield = env.get_batches_with(labels=set_labels(), random_seed=1)
        print(f"=== peni_yield @ batch {i}: {peni_yield}")
        dfs.append(df)

    df_total = pd.concat(dfs)
    df_total[DATE_TIME] = pd.date_range(start=start_datetime, periods=len(df_total), freq=GRANULARITY)
    df_total = df_total[(df_total[DATE_TIME] > start_datetime) & (df_total[DATE_TIME] <= end_datetime)]
    print(df_total)


run()
