import json
import os
import pathlib
import shutil


def clean():
    # remove logs with no stats from logs dir
    logs = list(pathlib.Path("logs").glob("*"))
    removed = 0
    total = len(logs)
    for log in logs:
        stats = os.path.join(log, "stats.json")
        if not os.path.exists(stats):
            removed += 1
            shutil.rmtree(log)
    print(f"Removed {removed} of {total} files")


def find():
    match_fields = {
        "dataset_path": "openwebtext",
    }
    output_fields = []
    res = {}
    logs = list(pathlib.Path("logs").glob("*"))
    for log in logs:
        config = os.path.join(log, "args.json")
        stats = os.path.join(log, "stats.json")
        try:
            with open(config, "r") as f:
                config = json.load(f)
            with open(stats, "r") as f:
                stats = json.load(f)
        except Exception:
            continue
        flag = True
        for k, v in match_fields.items():
            if config.get(k) is None or v not in config[k]:
                flag = False
                break
        if flag:
            output = dict()
            if len(output_fields) == 0:
                output["config"] = config
                output["stats"] = stats
            else:
                output = {k: config.get(k, None) for k in output_fields}
                for k, v in output.items():
                    if v is None:
                        output[k] = stats.get(v, None)
            res[str(log)] = output
                    
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    clean()
    find()