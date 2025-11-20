import os
import pandas as pd
import gensim.downloader as api
import sys
import re
sys.stdout.reconfigure(encoding="utf-8")
HF_TOKEN = os.environ.get('HF_TOKEN')

# Login using e.g. `huggingface-cli login` to access this dataset
#Task 1
def get_reguest_data():
    all_pull_request = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
    all_pull_request = all_pull_request.rename(columns={
        "title": "TITLE",
        "id": "ID",
        "agent": "AGENTNAME",
        "body": "BODYSTRING",
        "repo_id": "REPOID",
        "repo_url": "REPOURL",
    })[["TITLE", "ID", "AGENTNAME", "BODYSTRING", "REPOID", "REPOURL"]]
    return all_pull_request

def get_repository_data():
    all_repository = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
    all_repository = all_repository.rename(columns={
        "id": "REPOID",
        "language": "LANG",
        "stars": "STARS",
        "url": "REPOURL"
    })[["REPOID", "LANG", "STARS", "REPOURL"]]
    return all_repository

def get_user_data():
    pr_task_type = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet")
    pr_task_type = pr_task_type.rename(columns={
        "id": "PRID",
        "title": "PRTITLE",
        "reason": "PRREASON",
        "type": "PRTYPE",
        "confidence": "CONFIDENCE"
    })[["PRID", "PRTITLE", "PRREASON", "PRTYPE", "CONFIDENCE"]]
    return pr_task_type

def get_pre_commit_data():
    pr_commit_details = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
    pr_commit_details = pr_commit_details.rename(columns={
        "pr_id": "PRID",
        "sha": "PRSHA",
        "message": "PRCOMMITMESSAGE",
        "filename": "PRFILE",
        "status": "PRSTATUS",
        "additions": "PRADDS",
        "deletions": "PRDELSS",
        "changes": "PRCHANGECOUNT",
        "patch": "PRDIFF"
    })[["PRID", "PRSHA", "PRCOMMITMESSAGE", "PRFILE", "PRSTATUS", "PRADDS", "PRDELSS", "PRCHANGECOUNT", "PRDIFF"]]
    pr_commit_details["PRDIFF"] = (
        pr_commit_details["PRDIFF"]
        .fillna("")
        .str.encode("ascii", "ignore")
        .str.decode("ascii"))
    return pr_commit_details

KEYWORDS = [
    "race","racy","buffer","overflow","stack","integer","signedness","underflow",
    "improper","unauthenticated","gain access","permission","cross site","css","xss",
    "denial service","dos","crash","deadlock","injection","request forgery","csrf",
    "xsrf","forged","security","vulnerability","vulnerable","exploit","attack",
    "bypass","backdoor","threat","expose","breach","violate","fatal","blacklist",
    "overrun","insecure"
]

def has_security(text):
    pattern = re.compile("|".join(re.escape(w) for w in KEYWORDS), re.IGNORECASE)
    return int(bool(pattern.search(text or "")))

if __name__ == "__main__":
    # Task 1
    all_pull_request = get_reguest_data()
    all_pull_request.to_csv("task1.csv", index=False, encoding="utf-8-sig")
    # Task 2
    all_repository = get_repository_data()
    all_repository.to_csv("task2.csv", index=False, encoding="utf-8-sig")
    # Task 3
    pr_task_type = get_user_data()
    pr_task_type.to_csv("task3.csv", index=False, encoding="utf-8-sig")
    # Task 4
    pr_commit_details = get_pre_commit_data()
    pr_commit_details.to_csv("task4.csv", index=False, encoding="utf-8-sig")
    # Task 5
    task5 = all_pull_request.merge(pr_task_type, left_on="ID", right_on="PRID", how="left")
    task5["SECURITY"] = task5.apply(
        lambda r: has_security((r["TITLE"] or "") + " " + (r["BODYSTRING"] or "")), axis=1)
    task5 = task5.rename(columns={"AGENTNAME": "AGENT", "PRTYPE": "TYPE"})[
        ["ID", "AGENT", "TYPE", "CONFIDENCE", "SECURITY"]]
    task5.to_csv("task5.csv", index=False, encoding="utf-8-sig")

