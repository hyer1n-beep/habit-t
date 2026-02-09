# app.py
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, date

import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Helpers (HTTP)
# -----------------------------
def _http_get_json(url: str, timeout: int = 10):
    """Return parsed JSON dict or None."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


# -----------------------------
# API: Weather (OpenWeatherMap)
# -----------------------------
@st.cache_data(ttl=600)  # 10 minutes
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap 현재 날씨(섭씨, 한국어) 가져오기.
    실패 시 None 반환.
    """
    if not api_key:
        return None

    # 한국 주요 도시는 대체로 q=City,KR 로 잘 잡힙니다.
    q = f"{city},KR"
    params = {
        "q": q,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }
    url = "https://api.openweathermap.org/data/2.5/weather?" + urllib.parse.urlencode(params)

    data = _http_get_json(url, timeout=10)
    if not data or str(data.get("cod")) not in ("200", "200.0"):
        return None

    try:
        main = data.get("main", {}) or {}
        weather0 = (data.get("weather", []) or [{}])[0] or {}
        clouds = data.get("clouds", {}) or {}
        wind = data.get("wind", {}) or {}

        # 강수확률(pop)은 current weather 응답에 보통 없어서,
        # rain/snow 존재 여부로 "가능"만 표현하거나 None 처리.
        rain = data.get("rain", {}) or {}
        snow = data.get("snow", {}) or {}
        precip_mm = None
        if "1h" in rain:
            precip_mm = rain.get("1h")
        elif "3h" in rain:
            precip_mm = rain.get("3h")
        elif "1h" in snow:
            precip_mm = snow.get("1h")
        elif "3h" in snow:
            precip_mm = snow.get("3h")

        return {
            "city": city,
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "desc": weather0.get("description"),
            "humidity": main.get("humidity"),
            "clouds": clouds.get("all"),
            "wind": wind.get("speed"),
            "precip_mm": precip_mm,  # mm (최근 1h/3h 강수량) 또는 None
        }
    except Exception:
        return None


# -----------------------------
# API: Dog (Dog CEO)
# -----------------------------
@st.cache_data(ttl=600)  # 10 minutes (or refresh by button with cache clear)
def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL + 품종 문자열 가져오기.
    실패 시 None 반환.
    """
    url = "https://dog.ceo/api/breeds/image/random"
    data = _http_get_json(url, timeout=10)
    if not data or data.get("status") != "success":
        return None

    img_url = data.get("message")
    if not img_url:
        return None

    # 품종 파싱: .../breeds/{breed}/... 형태
    breed = "Unknown"
    try:
        parts = img_url.split("/breeds/")
        if len(parts) >= 2:
            after = parts[1]
            breed_part = after.split("/")[0]  # e.g. "hound-afghan" or "retriever-golden"
            # 하이픈은 sub-breed일 수 있음 -> 보기 좋게 변환
            breed = " ".join([w.capitalize() for w in breed_part.split("-")])
    except Exception:
        breed = "Unknown"

    return {"url": img_url, "breed": breed}


# -----------------------------
# AI Report (OpenAI)
# -----------------------------
def _style_system_prompt(style: str) -> str:
    if style == "스파르타 코치":
        return (
            "너는 매우 엄격하고 직설적인 습관 코치다. 변명은 받아주지 않는다. "
            "짧고 강하게 핵심만 말하되, 행동으로 옮기기 쉬운 지시를 준다."
        )
    if style == "따뜻한 멘토":
        return (
            "너는 따뜻하고 지지적인 멘토다. 사용자의 감정을 공감하고, "
            "작은 성취도 크게 칭찬하며, 부담 없는 다음 행동을 제안한다."
        )
    # 게임 마스터
    return (
        "너는 RPG 게임 마스터다. 사용자의 하루를 퀘스트/레벨업/아이템/보상처럼 연출한다. "
        "재미있지만 실제로 실행 가능한 미션을 준다."
    )


def _rule_based_weather_tip(weather: dict | None) -> str:
    if not weather:
        return "날씨 정보를 못 가져왔어. 대신 오늘은 컨디션에 맞춰 '가장 쉬운 습관 1개'만 확실히!"
    desc = (weather.get("desc") or "").lower()
    temp = weather.get("temp")
    precip_mm = weather.get("precip_mm")

    if precip_mm is not None and precip_mm > 0:
        return "비/눈 기운이 있어. 실내 스트레칭·홈트 추천! 짧게라도 몸을 깨우자."
    if "비" in desc or "소나기" in desc or "눈" in desc:
        return "강수 징후가 있어. 실내 루틴이 유리해! 10분 스트레칭부터 가자."
    if temp is not None and temp >= 28:
        return "더운 편이야. 물 보충 + 가벼운 강도의 실내 운동이 좋아."
    if temp is not None and temp <= 2:
        return "추운 편이야. 무리한 야외운동보다 워밍업 철저히, 짧게라도 움직이자."
    return "무난한 날씨야. 가능하면 가볍게 산책/야외 활동으로 에너지 충전!"


def generate_report(
    openai_api_key: str,
    coach_style: str,
    nickname: str,
    goal: str,
    mood: int,
    habits_checked: dict,
    weather: dict | None,
    dog: dict | None,
) -> str | None:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달.
    실패 시 None 반환.
    """
    if not openai_api_key or OpenAI is None:
        return None

    system = _style_system_prompt(coach_style)

    # 사용자 입력 요약 (토큰 절약)
    checked = [k for k, v in habits_checked.items() if v]
    unchecked = [k for k, v in habits_checked.items() if not v]

    weather_summary = "날씨 정보 없음"
    if weather:
        w_desc = weather.get("desc")
        t = weather.get("temp")
        fl = weather.get("feels_like")
        rain = weather.get("precip_mm")
        weather_summary = f"{weather.get('city')} / {w_desc}, {t}°C(체감 {fl}°C)"
        if rain is not None:
            weather_summary += f", 강수량 {rain}mm"

    dog_summary = "강아지 정보 없음"
    if dog:
        dog_summary = f"{dog.get('breed')} (랜덤 강아지)"

    user_prompt = f"""
[사용자]
- 닉네임: {nickname or "사용자"}
- 목표: {goal or "미입력"}
- 오늘 기분(1~10): {mood}

[오늘 습관 체크]
- 완료: {", ".join(checked) if checked else "없음"}
- 미완료: {", ".join(unchecked) if unchecked else "없음"}

[날씨]
- {weather_summary}

[동기부여]
- {dog_summary}

너의 출력은 반드시 아래 형식을 정확히 지켜라(한국어):

컨디션 등급: (S/A/B/C/D 중 하나)

습관 분석:
- (1) 오늘 잘한 점 2~3개
- (2) 오늘 아쉬운 점 1~2개
- (3) 내일 개선 포인트 1개(아주 구체적으로)

날씨 코멘트: (날씨가 없으면 대체 코멘트)

내일 미션:
- 미션 1:
- 미션 2:
- 보너스 미션(선택):

오늘의 한마디: (한 줄)

주의: 과한 장황함 금지. 실행 가능한 말만. 
"""

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


# -----------------------------
# Session State init
# -----------------------------
if "history" not in st.session_state:
    # 데모용 6일 샘플 + 오늘은 나중에 append/replace
    # (최근 6일: D-6 ~ D-1)
    base = date.today()
    sample_rates = [40, 60, 80, 50, 70, 90]  # 예시
    sample_moods = [4, 6, 7, 5, 6, 8]
    st.session_state.history = []
    for i in range(6, 0, -1):
        d = base - timedelta(days=i)
        st.session_state.history.append(
            {
                "date": d.isoformat(),
                "rate": sample_rates[6 - i],
                "mood": sample_moods[6 - i],
                "done": None,  # 과거는 단순 데모
            }
        )

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "weather_cache_bust" not in st.session_state:
    st.session_state.weather_cache_bust = 0

if "dog_cache_bust" not in st.session_state:
    st.session_state.dog_cache_bust = 0


# -----------------------------
# Sidebar: keys & settings
# -----------------------------
st.sidebar.title("AI Habit Tracker")

openai_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...", key="openai_key")
owm_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password", placeholder="owm-...", key="owm_key")

st.sidebar.divider()

nickname = st.sidebar.text_input("닉네임", value=st.session_state.get("nickname", ""), key="nickname")
goal = st.sidebar.text_input("목표", value=st.session_state.get("goal", ""), key="goal")

CITY_OPTIONS = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Sejong",
    "Jeju",
]
city = st.sidebar.selectbox("도시 선택", CITY_OPTIONS, index=0, key="city")

st.sidebar.caption("🔐 API 키는 **세션에만 입력**하고, 저장/커밋하지 마세요.")

# -----------------------------
# Main: UI
# -----------------------------
st.title("📊 AI 습관 트래커")

# Controls row
left, right = st.columns([2, 1], vertical_alignment="top")
with right:
    coach_style = st.radio(
        "코치 스타일",
        ["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
        horizontal=False,
        key="coach_style",
    )
    mood = st.slider("오늘 기분", min_value=1, max_value=10, value=6, key="mood")

with left:
    st.subheader("✅ 오늘 습관 체크인")

    habits = [
        ("🌅", "기상 미션"),
        ("💧", "물 마시기"),
        ("📚", "공부/독서"),
        ("🏃", "운동하기"),
        ("😴", "수면"),
    ]

    c1, c2 = st.columns(2)
    habit_state = {}
    for idx, (emo, name) in enumerate(habits):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            habit_state[name] = st.checkbox(f"{emo} {name}", value=False, key=f"habit_{name}")

# Achievement calculation
done_count = sum(1 for v in habit_state.values() if v)
total_count = len(habit_state)
rate = int(round((done_count / total_count) * 100))

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%")
m2.metric("달성 습관", f"{done_count}/{total_count}")
m3.metric("기분", f"{mood}/10")

# History update (keep 7 days: last 6 samples + today)
def upsert_today_history(rate_value: int, mood_value: int, done_value: int):
    today_iso = date.today().isoformat()
    # replace if exists
    replaced = False
    for row in st.session_state.history:
        if row["date"] == today_iso:
            row["rate"] = rate_value
            row["mood"] = mood_value
            row["done"] = done_value
            replaced = True
            break
    if not replaced:
        st.session_state.history.append({"date": today_iso, "rate": rate_value, "mood": mood_value, "done": done_value})

    # keep only last 7 days by date
    st.session_state.history.sort(key=lambda x: x["date"])
    if len(st.session_state.history) > 7:
        st.session_state.history = st.session_state.history[-7:]


# Chart (7-day bar chart)
st.divider()
st.subheader("📈 최근 7일 달성률")

# Ensure today's placeholder exists for chart (even before generating report)
upsert_today_history(rate, mood, done_count)

def _history_df():
    rows = st.session_state.history[-7:]
    labels = []
    rates = []
    for r in rows:
        d = datetime.fromisoformat(r["date"]).date()
        labels.append(d.strftime("%m-%d"))
        rates.append(r["rate"])
    if pd is None:
        # fallback: list-of-dict for st.bar_chart can work sometimes, but pandas is usually present.
        return {"date": labels, "rate": rates}
    return pd.DataFrame({"date": labels, "달성률(%)": rates}).set_index("date")

st.bar_chart(_history_df())

# Weather & Dog cards
st.divider()
st.subheader("🌦️ 오늘의 컨디션 재료")

wc, dc = st.columns(2)

with wc:
    st.markdown("#### 🌦️ 날씨 카드")
    # cache bust by changing dummy key into function input (simple trick)
    _ = st.session_state.weather_cache_bust
    weather = get_weather(city=city, api_key=owm_key)

    if weather is None:
        st.warning("날씨 정보를 불러오지 못했습니다. (키/네트워크/도시 설정을 확인)")
        weather_tip = _rule_based_weather_tip(None)
        st.caption(weather_tip)
    else:
        st.write(f"**{weather['city']}**")
        st.write(f"- 현재: **{weather['temp']}°C** (체감 **{weather['feels_like']}°C**)")
        st.write(f"- 상태: **{weather['desc']}**")
        if weather.get("precip_mm") is not None:
            st.write(f"- 강수량: **{weather['precip_mm']} mm**")
        st.write(f"- 습도: {weather.get('humidity')}% / 구름: {weather.get('clouds')}% / 바람: {weather.get('wind')} m/s")
        weather_tip = _rule_based_weather_tip(weather)
        st.info(weather_tip)

    if st.button("🔄 날씨 새로고침"):
        st.cache_data.clear()
        st.session_state.weather_cache_bust += 1
        st.rerun()

with dc:
    st.markdown("#### 🐶 오늘의 강아지")
    _ = st.session_state.dog_cache_bust
    dog = get_dog_image()

    if dog is None:
        st.warning("강아지 이미지를 불러오지 못했습니다.")
        st.caption("대신: 오늘도 체크 하나만 해도 이긴 거야 🐾")
    else:
        st.image(dog["url"], use_container_width=True)
        st.write(f"품종: **{dog['breed']}**")
        st.caption("칭찬: 오늘도 한 칸만 채워도 ‘연속 달성’에 가까워진다!")

    if st.button("🎲 다른 강아지 보기"):
        st.cache_data.clear()
        st.session_state.dog_cache_bust += 1
        st.rerun()

# Report generation
st.divider()
st.subheader("🧠 AI 코치 리포트")

col_a, col_b = st.columns([1, 2], vertical_alignment="top")
with col_a:
    st.markdown("**리포트 생성 조건**")
    if not openai_key:
        st.error("OpenAI API Key가 필요합니다. (사이드바에 입력)")
    if OpenAI is None:
        st.error("openai 라이브러리를 불러올 수 없습니다. requirements/설치 상태를 확인하세요.")
    st.caption("버튼을 누르면: 습관+기분+날씨+강아지 정보를 합쳐 리포트를 생성합니다.")

generate = st.button("📌 컨디션 리포트 생성", type="primary")

if generate:
    # Ensure history saved (today)
    upsert_today_history(rate, mood, done_count)

    report = generate_report(
        openai_api_key=openai_key,
        coach_style=coach_style,
        nickname=nickname,
        goal=goal,
        mood=mood,
        habits_checked=habit_state,
        weather=weather,
        dog=dog,
    )

    if report is None:
        st.error("AI 코칭 생성 실패: (키 누락/네트워크/요청 오류 가능) — 잠시 후 다시 시도해 주세요.")
        st.session_state.last_report = None
    else:
        st.session_state.last_report = report

# Display report + share text
if st.session_state.last_report:
    st.markdown("### ✅ 리포트 결과")
    st.markdown(st.session_state.last_report)

    # Share text
    checked_names = [name for name, v in habit_state.items() if v]
    unchecked_names = [name for name, v in habit_state.items() if not v]

    weather_line = "날씨: 불러오기 실패"
    if weather:
        weather_line = f"날씨: {weather.get('city')} / {weather.get('desc')}, {weather.get('temp')}°C(체감 {weather.get('feels_like')}°C)"

    dog_line = "강아지: 불러오기 실패"
    if dog:
        dog_line = f"강아지: {dog.get('breed')} 🐶"

    share_text = f"""[AI 습관 트래커 - 오늘 기록]
- 닉네임: {nickname or "사용자"}
- 목표: {goal or "미입력"}
- 달성률: {rate}% ({done_count}/{total_count})
- 기분: {mood}/10
- 완료: {", ".join(checked_names) if checked_names else "없음"}
- 미완료: {", ".join(unchecked_names) if unchecked_names else "없음"}
- {weather_line}
- {dog_line}

[AI 리포트]
{st.session_state.last_report}
"""
    st.markdown("### 📤 공유용 텍스트")
    st.code(share_text, language="text")

# API 안내
st.divider()
with st.expander("ℹ️ API 안내 / 키 관리 / 참고"):
    st.markdown(
        """
**사용 API**
- OpenWeatherMap Current Weather: `https://api.openweathermap.org/data/2.5/weather`
  - 파라미터: `q=Seoul,KR`, `units=metric`, `lang=kr`, `appid=YOUR_KEY`
- Dog CEO Random Image: `https://dog.ceo/api/breeds/image/random`
- OpenAI Chat Completions (Python SDK)

**키 관리(중요)**
- API Key는 **사이드바 입력(세션)** 또는 `st.secrets`로만 관리하세요.
- GitHub에 커밋 금지: `.streamlit/secrets.toml` 사용 시 `.gitignore`에 포함 권장.

**실패 처리**
- 날씨/강아지 API 실패 시 `None` 반환 → 앱은 계속 동작(대체 문구 표시)
- OpenAI 실패 시 에러 메시지 표시 후 리포트 출력 생략

**팁**
- 날씨/강아지 새로고침이 안 먹는다면: 버튼으로 캐시를 비우도록 구현되어 있습니다.
"""
    )
