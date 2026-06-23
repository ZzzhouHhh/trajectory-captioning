"""
Multi-stage reasoning prompt templates.

三阶段独立推理（CoT-inspired multi-stage reasoning）：
1. Starting semantic reasoning: 从出发地推断旅行者身份和出发原因
2. Arrival semantic reasoning: 从目的地推断出行意图
3. Trajectory caption generation: 综合前两步+原始属性生成最终描述
"""


# ========== 系统提示词与列说明 ==========

SYSTEM_PROMPT = (
    "You are an expert in trajectory interpretation and can generate "
    "concise textual descriptions based on the trajectory information I provided."
)

COLUMN_DESCRIPTION = (
    "The columns of trajectory data that I will provide include "
    "(trajectory_id, pickup_venueCategory, pickup_name, pickup_category, "
    "dropoff_venueCategory, dropoff_name, dropoff_category, "
    "passenger_count, pickup_hour, dropoff_hour, pickup_week). "
    "The last three columns indicate the hour when the user picked up and "
    "dropped off as well as which day of the week it was. "
    "Note: [For cases where pickup_name, pickup_category, dropoff_name, or "
    "dropoff_category are empty, you can choose to skip these entries or "
    "generate text descriptions using only pickup_venueCategory and "
    "dropoff_venueCategory.]"
)


# ========== Stage 1: Starting Semantic Reasoning ==========

def build_starting_reasoning_prompt(pickup_poi_info, pickup_hour, pickup_week):
    """
    Stage 1: 从起点POI和时间推断旅行者身份和出发原因。
    独立LLM调用，与Stage 2可并行。
    """
    prompt = (
        f"The traveler departed from the following location:\n"
        f"Origin POI: {pickup_poi_info}\n"
        f"Time: {pickup_hour}:00 on {pickup_week}.\n\n"
        f"Based on the origin POI category, name, and departure time, "
        f"infer the possible identity of the traveler (e.g., hotel guest, "
        f"office worker, resident, tourist) and the likely reason for "
        f"departing from this location. Provide a concise one-sentence reasoning."
    )
    return prompt


# ========== Stage 2: Arrival Semantic Reasoning ==========

def build_arrival_reasoning_prompt(dropoff_poi_info, dropoff_hour, dropoff_week):
    """
    Stage 2: 从终点POI和时间推断出行意图。
    独立LLM调用，与Stage 1可并行。
    """
    prompt = (
        f"The traveler arrived at the following location:\n"
        f"Destination POI: {dropoff_poi_info}\n"
        f"Time: {dropoff_hour}:00 on {dropoff_week}.\n\n"
        f"Based on the destination POI category, name, and arrival time, "
        f"infer the traveler's purpose or intention for arriving at this "
        f"location (e.g., having a meal, attending a meeting, shopping, "
        f"returning home). Provide a concise one-sentence reasoning."
    )
    return prompt


# ========== Stage 3: Trajectory Caption Generation ==========

def build_caption_generation_prompt(trajectory_info_text, pickup_poi_info,
                                     dropoff_poi_info, starting_reasoning,
                                     arrival_reasoning):
    """
    Stage 3: 综合原始轨迹属性 + Stage 1/2 的推理结果，生成最终轨迹描述。
    """
    prompt = (
        f"Synthesize a natural language description of the following trip:\n\n"
        f"Trip attributes:\n{trajectory_info_text}\n\n"
        f"Origin: {pickup_poi_info}\n"
        f"Destination: {dropoff_poi_info}\n\n"
        f"Context from origin analysis: {starting_reasoning}\n"
        f"Context from destination analysis: {arrival_reasoning}\n\n"
        f"Generate a single concise sentence describing this trip, "
        f"including the traveler's inferred identity, the trip purpose, "
        f"and relevant temporal context."
    )
    return prompt


# ========== 消融实验：单阶段直接生成 (w/o CoT / w/o MSR) ==========

def build_simple_prompt(trajectory_info_text, semantic_info_text):
    """
    不使用多阶段推理链，直接要求LLM生成轨迹描述。
    用于消融实验中的 w/o CoT (w/o multi-stage reasoning) 对比。
    """
    prompt = (
        f"Based on the following trajectory information, generate a natural "
        f"language description of the trip.\n\n"
        f"Trajectory Info:\n{trajectory_info_text}\n\n"
        f"Geographical Semantics:\n{semantic_info_text}\n\n"
        f"Generated Description:"
    )
    return prompt


# ========== 格式化函数 ==========

def format_trajectory_info_from_row(row):
    """
    从轨迹DataFrame的一行提取轨迹属性文本。
    包含：起止时间、星期几、乘客数量。
    """
    weekday = row.get('pickup_week', row.get('pickup_weekday', ''))
    if isinstance(weekday, (int, float)):
        weekday_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                       3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        weekday = weekday_map.get(int(weekday), str(weekday))

    pickup_hour = row.get('pickup_hour', '')
    dropoff_hour = row.get('dropoff_hour', '')
    passenger_count = row.get('passenger_count', 1)

    return (
        f"pickup_hour: {pickup_hour}, dropoff_hour: {dropoff_hour}, "
        f"pickup_week: {weekday}, passenger_count: {int(passenger_count)}"
    )


def format_poi_info(name, category, venue_category):
    """格式化单个POI信息为可读字符串"""
    parts = []
    if name:
        parts.append(name)
    if category:
        parts.append(f"({category})")
    if venue_category:
        parts.append(f"[{venue_category}]")
    return " ".join(parts) if parts else "Unknown"


def format_semantic_info_from_row(row):
    """
    从轨迹DataFrame的一行提取 <semantic_info> 文本。
    包含：起止POI的名称、类别、子类别。
    GSRM输出的POI序列地理语义信息。
    """
    pickup_name = row.get('pickup_name', row.get('pickup_predicted_name', ''))
    pickup_cat = row.get('pickup_category', row.get('pickup_predicted_category', ''))
    pickup_vcat = row.get('pickup_venueCategory', '')

    dropoff_name = row.get('dropoff_name', row.get('dropoff_predicted_name', ''))
    dropoff_cat = row.get('dropoff_category', row.get('dropoff_predicted_category', ''))
    dropoff_vcat = row.get('dropoff_venueCategory', '')

    parts = []
    if pickup_name or pickup_cat or pickup_vcat:
        parts.append(
            f"Origin: {pickup_name} ({pickup_cat}, {pickup_vcat})"
            .replace(" (,", " (").replace(", )", ")")
        )
    if dropoff_name or dropoff_cat or dropoff_vcat:
        parts.append(
            f"Destination: {dropoff_name} ({dropoff_cat}, {dropoff_vcat})"
            .replace(" (,", " (").replace(", )", ")")
        )
    return "; ".join(parts) if parts else "No semantic information available."
