# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import json
import os
import sqlite3

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import openai
import scipy.stats as st

# OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# 数据库文件列表
db_files = [
    'counterfactual_1000_up.db', 'counterfactual_1000_control.db',
    'counterfactual_1000_down.db'
]

# 存储每个数据库的结果
all_scores_by_time_step = {}


async def fetch_gpt_score(session, prompt, time_step, post_content,
                          comment_content):
    try:
        response = await session.post(
            "https://yunwu.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}"},
            json={
                "model":
                "gpt-4o",
                "messages": [{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens":
                50,  # 增加max_tokens以确保完整响应
                "temperature":
                0.5
            })
        
        if response.status != 200:
            print(f"HTTP Error: {response.status}")
            return None
            
        response_json = await response.json()
        
        if 'choices' not in response_json or not response_json['choices']:
            print(f"No choices in response: {response_json}")
            return None
            
        gpt_response = response_json['choices'][0]['message']['content'].strip()
        print(f"GPT Response: {gpt_response}")  # 调试信息
        
        # 尝试解析JSON
        try:
            gpt_output = json.loads(gpt_response)
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试提取数字
            import re
            score_match = re.search(r'"score":\s*(\d+)', gpt_response)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 10:
                    print(f"Extracted score from text: {score}")
                    return score
            print(f"Failed to parse JSON or extract score from: {gpt_response}")
            return None
            
        score = gpt_output.get("score")
        if isinstance(score, int) and 1 <= score <= 10:
            print(f"time_stamp: {time_step}, post: {post_content}, comment: "
                  f"{comment_content}, score: {score}")
            return score
        else:
            print(f"Invalid score: {score}")
            return None
    except Exception as e:
        print(f"Error fetching GPT score: {e}")
    return None


async def process_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, created_at, info FROM trace WHERE action = 'create_comment'")
    trace_records = cursor.fetchall()

    # 先收集所有记录并按时间排序
    time_records = []
    for record in trace_records:
        print(f"Processing record: {record}")
        info = json.loads(record[2])
        created_at = record[1]  # 从created_at字段获取时间
        
        try:
            # 解析时间戳
            from datetime import datetime
            timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00') if created_at.endswith('Z') else created_at)
            time_records.append((timestamp, info))
        except Exception as e:
            print(f"Error processing timestamp for record {record}: {e}")
            continue
    
    # 按时间排序
    time_records.sort(key=lambda x: x[0])
    
    # 将评论分组到30个时间步中
    time_step_dict = {}
    total_records = len(time_records)
    num_timesteps = 30  # 根据YAML配置文件设置
    
    if total_records > 0:
        records_per_step = max(1, total_records // num_timesteps)
        
        for i, (timestamp, info) in enumerate(time_records):
            # 将记录分配到对应的时间步
            time_step = min(i // records_per_step, num_timesteps - 1)
            if time_step not in time_step_dict:
                time_step_dict[time_step] = []
            time_step_dict[time_step].append(info)

    if not time_step_dict:
        print(f"Warning: No valid records found in {db_file}")
        return {}

    scores_by_time_step = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for time_step, traces in time_step_dict.items():
            for trace in traces:
                comment_id = trace.get('comment_id')
                if comment_id:
                    cursor.execute(
                        ("SELECT post_id, content FROM comment WHERE "
                         "comment_id = ?"), (comment_id, ))
                    comment_record = cursor.fetchone()
                    if comment_record:
                        post_id, comment_content = comment_record

                        cursor.execute(
                            "SELECT content FROM post WHERE post_id = ?",
                            (post_id, ))
                        post_record = cursor.fetchone()
                        if post_record:
                            post_content = post_record[0]

                            prompt = f"""
                            Someone posted a counterfactual statement: "{post_content}", and then another person commented: "{comment_content}".
                            Please provide a rating from 1 to 10 to indicate how much this person agrees with the counterfactual content, where:
                            1 means fully believing and agreeing with it, and 10 means completely disbelieving and opposing it.

                            Respond only with a JSON object like this:
                            {{
                              "score": 7
                            }}

                            Ensure the "score" is a single integer between 1 and 10.
                            """  # noqa: E501

                            tasks.append(
                                (time_step,
                                 fetch_gpt_score(session, prompt, time_step,
                                                 post_content,
                                                 comment_content)))

        if not tasks:
            print(f"Warning: No tasks generated for {db_file}")
            return {}

        results = await asyncio.gather(*[task[1] for task in tasks])

        for (time_step, _), result in zip(tasks, results):
            if result is not None:
                if time_step not in scores_by_time_step:
                    scores_by_time_step[time_step] = []
                scores_by_time_step[time_step].append(result)

    conn.close()
    return scores_by_time_step


async def main():
    for db_file in db_files:
        scores = await process_database(db_file)
        if scores:
            all_scores_by_time_step[db_file] = scores
        else:
            print(f"Warning: No valid data for {db_file}")


# 运行异步主函数
asyncio.run(main())

# 绘制折线图
plt.figure(figsize=(12, 7))
colors = ['b', 'g', 'r']

for i, (db_file,
        scores_by_time_step) in enumerate(all_scores_by_time_step.items()):
    if not scores_by_time_step:
        print(f"Skipping empty data for {db_file}")
        continue

    time_steps = sorted(scores_by_time_step.keys())
    means = []
    conf_intervals = []

    for time_step in time_steps:
        scores = scores_by_time_step[time_step]
        if scores:
            mean = np.mean(scores)
            confidence_interval = st.t.interval(0.95,
                                                len(scores) - 1,
                                                loc=mean,
                                                scale=st.sem(scores))
            means.append(mean)
            conf_intervals.append(confidence_interval)

    if means:
        lower_bounds = [ci[0] for ci in conf_intervals]
        upper_bounds = [ci[1] for ci in conf_intervals]

        plt.plot(time_steps,
                 means,
                 marker='o',
                 color=colors[i],
                 label=db_file.split('/')[-1])
        plt.fill_between(time_steps,
                         lower_bounds,
                         upper_bounds,
                         color=colors[i],
                         alpha=0.2)
    else:
        print(f"No valid data to plot for {db_file}")

if not all_scores_by_time_step:
    print("Error: No valid data to plot")
else:
    plt.xlabel('Time Step')
    plt.ylabel('Average Score')
    plt.title('Average Scores with Confidence Intervals over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_scores_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
