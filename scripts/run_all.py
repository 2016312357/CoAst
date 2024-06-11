from concurrent.futures import ThreadPoolExecutor, as_completed
import click
import os
from loguru import logger
import subprocess


def run_make(config_file):
    logger.info(f"Processing {config_file}")
    result = subprocess.run([config_file], shell=True, capture_output=True)
    last_result = str(result).split('\\n')[-2]
    if "such" in str(result):
        last_result = "None"
    short_config_name = config_file.split("analyze")[-1]
    return f"{short_config_name}: {last_result}"


COMMAND = "make param_contri_partition_softmax"
THREADS = 25
# This is our shell command, executed in subprocess.

@click.command()
@click.option("--list_file")
def run_all(list_file):
    if not os.path.exists(list_file):
        logger.error("File {} not found".format(list_file))
        raise FileNotFoundError(f"File {list_file} not found")

    memo = {}
    executor = ThreadPoolExecutor(max_workers=THREADS)
    with open(list_file, "r") as f:
        all_task = [executor.submit(run_make, (config_file)) for config_file in f.readlines()]

        for future in as_completed(all_task):
            data = future.result()
            logger.success(f"{data}")
            memo[data.split(":")[0]] = data.split(":")[-1]
    logger.info("------------------------------------------------")
    final_record = list(memo.items())
    final_record = sorted(final_record, key=lambda x:x[0])
    for k, v in final_record:
        _, method, ratio, tmp_name = k.split('/')
        if tmp_name.count("-") == 3:
            count, dataset, model, mode = tmp_name.split("-")
            mode = mode.split(".")[0]
            print(f"{method},{ratio},{count},{dataset},{model},{mode},{v}")
        else:
            dataset, model, mode = tmp_name.split("-")
            mode = mode.split(".")[0]
            print(f"{method},{ratio},5,{dataset},{model},{mode},{v}")

if __name__ == "__main__":
    run_all()
