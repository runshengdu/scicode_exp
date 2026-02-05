import pandas as pd
from pathlib import Path
from collections import defaultdict

log_dir = Path("/Users/miao/Desktop/GBAI/Resources/Test/SciCode-main/eval/inspect_ai/tmp/openai-api-ark-doubao-seed-1-8-251228-3/evaluation_logs/with_background")

data = []
problem_stats = defaultdict(lambda: {"pass": 0, "fail": 0, "timeout": 0, "total": 0})

for log_file in sorted(log_dir.glob("*.log")):
    filename = log_file.name
    content = log_file.read_text(encoding="utf-8").strip()
    
    data.append({
        "文件名": filename,
        "文件内容": content
    })
    
    problem_id = filename.split(".")[0]
    problem_stats[problem_id]["total"] += 1
    
    if content == "pass":
        problem_stats[problem_id]["pass"] += 1
    elif content == "fail":
        problem_stats[problem_id]["fail"] += 1
    elif content == "time out":
        problem_stats[problem_id]["timeout"] += 1

df = pd.DataFrame(data)

total_pass = sum(stats["pass"] for stats in problem_stats.values())
total_fail = sum(stats["fail"] for stats in problem_stats.values())
total_timeout = sum(stats["timeout"] for stats in problem_stats.values())
total_steps = sum(stats["total"] for stats in problem_stats.values())

accuracy = (total_pass / total_steps * 100) if total_steps > 0 else 0

summary_data = []
for problem_id in sorted(problem_stats.keys()):
    stats = problem_stats[problem_id]
    problem_accuracy = (stats["pass"] / stats["total"] * 100) if stats["total"] > 0 else 0
    summary_data.append({
        "问题ID": problem_id,
        "通过数": stats["pass"],
        "失败数": stats["fail"],
        "超时数": stats["timeout"],
        "总步骤数": stats["total"],
        "正确率 (%)": f"{problem_accuracy:.2f}"
    })

summary_df = pd.DataFrame(summary_data)

output_file = Path("/Users/miao/Desktop/GBAI/Resources/Test/SciCode-main/eval/inspect_ai/tmp/evaluation_results-3.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="日志详情", index=False)
    summary_df.to_excel(writer, sheet_name="统计汇总", index=False)
    
    total_summary = pd.DataFrame([{
        "总通过数": total_pass,
        "总失败数": total_fail,
        "总超时数": total_timeout,
        "总步骤数": total_steps,
        "总体正确率 (%)": f"{accuracy:.2f}"
    }])
    total_summary.to_excel(writer, sheet_name="总体统计", index=False)

print(f"Excel 文件已生成: {output_file}")
print(f"总体正确率: {accuracy:.2f}%")
print(f"总通过数: {total_pass}")
print(f"总失败数: {total_fail}")
print(f"总超时数: {total_timeout}")
print(f"总步骤数: {total_steps}")
