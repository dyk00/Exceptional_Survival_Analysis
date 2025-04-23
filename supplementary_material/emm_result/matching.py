from collections import defaultdict
from pathlib import Path

res_dir = Path("supplementary_material/emm_result")
models = ["cox", "coxnet", "ersf", "rsf", "weibull"]
suffixes = ["1", "10", "42"]

files = []
for model in models:
    for suffix in suffixes:
        path = res_dir / f"{model}{suffix}.txt"
        files.append(str(path))

data = defaultdict(lambda: defaultdict(set))

for file_path in files:
    p = Path(file_path)

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith("###") and line.endswith("###"):
            header = line

        elif header and line.startswith("Description:"):
            # get after description string
            desc = line[len("Description:") :].strip()

            # removes after (
            if "(" in desc:
                # [0] the first part before the first (
                desc = desc.split("(", 1)[0].strip()

            # split descriptions at &, sort and then combine
            equi_desc = " & ".join(sorted(s.strip() for s in desc.split("&")))

            # header, description, file paths (k,k,v)
            data[header][equi_desc].add(file_path)

for header, descs in data.items():
    # print header once
    header_printed = False
    sorted_descs = sorted(descs.items(), key=lambda item: len(item[1]), reverse=True)
    for desc, file in sorted_descs:
        if len(file) >= 7:
            if not header_printed:
                print(header)
                header_printed = True
            print(f"{desc}: found {len(file)} times")
    print()
