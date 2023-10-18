import sqlite3
from collections import defaultdict

import numpy as np

def fetch_events_during_benchmark(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch textId for "benchmark"
    cursor.execute("SELECT rowid FROM StringIds WHERE value = 'benchmark'")
    benchmark_text_id = cursor.fetchone()
    if benchmark_text_id is None:
        print("No 'benchmark' range found.")
        return
    benchmark_text_id = benchmark_text_id[0]

    # Fetch start and end times for "benchmark"
    cursor.execute(f"SELECT MIN(start), MAX(end) FROM NVTX_EVENTS WHERE textId = {benchmark_text_id}")
    start_time, end_time = cursor.fetchone()
    if start_time is None or end_time is None:
        print("No 'benchmark' range found.")
        return

    runtime_events = {}
    for name in ["cudaLaunchKernel", "cudaMemcpyAsync", "cudaDeviceSynchronize", "cuLaunchKernel"]:
        # Fetch textId for name
        cursor.execute(f"SELECT rowid FROM StringIds WHERE value LIKE '%{name}%'")
        text_id = cursor.fetchone()
        if text_id is None:
            print(f"No '{name}' events found.")
            continue
        text_id = text_id[0]

        # Fetch all events within the "benchmark" time range
        cursor.execute(f"SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId = {text_id} AND start >= {start_time} AND end <= {end_time}")
        runtime_events[name] = cursor.fetchall()

    all_events = defaultdict(list)
    unique_names = defaultdict(set)

    table = 'CUPTI_ACTIVITY_KIND_KERNEL'
    # Fetch all events within the "benchmark" time range
    cursor.execute(f"SELECT start, end, shortName FROM {table} WHERE start >= {start_time} AND end <= {end_time}")
    events = cursor.fetchall()

    for start, end, name_id in events:
        # Append to the list of events
        all_events[table].append((start, end))
        unique_names[table].add(name_id)

    # Other tables to fetch events from
    tables = [
        'CUPTI_ACTIVITY_KIND_MEMCPY',
    ]

    for table in tables:
        # Fetch all events within the "benchmark" time range
        cursor.execute(f"SELECT start, end FROM {table} WHERE start >= {start_time} AND end <= {end_time}")
        events = cursor.fetchall()

        for start, end in events:
            # Append to the list of events
            all_events[table].append((start, end))

    all_unique_names = set()
    for table, table_unique_names in unique_names.items():
        for name_id in table_unique_names:
            # Fetch the name for this event
            cursor.execute(f"SELECT value FROM StringIds WHERE rowid = {name_id}")
            name = cursor.fetchone()
            all_unique_names.add(name[0])

    # Close the database connection
    conn.close()

    return start_time, end_time, runtime_events, all_events, all_unique_names


def durations(events):
    events = np.array(events)
    return events[..., 1] - events[..., 0]


def main(db_path):
     (start_time,
      end_time,
      runtime_events,
      device_events,
      unique_names) = fetch_events_during_benchmark(db_path)

     estimated_cpu_wait_time = 0
     for name in ["cudaLaunchKernel", "cudaMemcpyAsync", "cudaDeviceSynchronize", "cuLaunchKernel"]:
         events = runtime_events.get(name, [])
         if len(events) < 5:
             continue

         # We assume the first few events didn't block.
         event_active_time = durations(events[1:4]).mean()
         total_event_active_time = int(event_active_time * len(events))
         estimated_cpu_wait_time += max(
             0,
             durations(events).sum() - total_event_active_time)

     for name in ["cudaDeviceSynchronize"]:
         events = runtime_events.get(name, [])
         if len(events) == 0:
             continue
         estimated_cpu_wait_time += durations(events).sum()

     benchmark_duration = end_time - start_time
     cpu_wait_pct = 100. * (1 - (estimated_cpu_wait_time / benchmark_duration))

     estimated_gpu_active_time = np.sum([durations(events).sum()
                                         for events in list(device_events.values())])
     gpu_wait_pct = 100. * estimated_gpu_active_time / benchmark_duration

     print(f'Duration: {benchmark_duration}')
     print(f'Estimated CPU wait time: {estimated_cpu_wait_time}')
     print(f'Estimated GPU active time: {estimated_gpu_active_time}')
     print(f'Estimated CPU usage: {cpu_wait_pct:>.2f}%')
     print(f'Estimated GPU usage: {gpu_wait_pct:>.2f}%')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', help='Path to the database file.')
    main(parser.parse_args().db_path)
