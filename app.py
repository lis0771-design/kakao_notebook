import platform
import random
import re
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

st.set_page_config(page_title="카카오톡 대화 분석기", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFBEA 0%, #FFFFFF 100%);
        border: 1px solid #F3E28A;
        border-radius: 16px;
        padding: 18px 16px 14px 16px;
        box-shadow: 0 8px 22px rgba(254, 229, 0, 0.14);
    }
    div[data-testid="stMetric"] label {
        color: #7A6A00 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #191919 !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #B8860B !important;
    }
    .stTabs {
        --primary-color: #7C3AED;
    }
    .stTabs [data-testid="stTab"] p,
    .stTabs button p,
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
    }
    .stTabs [data-testid="stTab"],
    .stTabs button,
    .stTabs [data-baseweb="tab"] {
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }
    .stTabs [data-testid="stTab"]:hover,
    .stTabs [data-testid="stTab"]:hover p,
    .stTabs button:hover,
    .stTabs button:hover p,
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover p {
        color: #7C3AED !important;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"],
    .stTabs [data-testid="stTab"][aria-selected="true"] p,
    .stTabs button[aria-selected="true"] p {
        color: #7C3AED !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-testid="stTabHighlight"],
    .stTabs [data-baseweb="tab"]:hover {
        background-color: transparent;
        border-bottom-color: #7C3AED !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #7C3AED !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = ["Date", "User", "Message"]
KAKAO_SCALE = ["#FFF6B7", "#FEE500", "#FFCD00", "#F5A623", "#E67E22", "#3C1E1E"]


def load_csv(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    last_error = None

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


WEEKDAYS_KO = ["월", "화", "수", "목", "금", "토", "일"]
SYSTEM_MESSAGES = ("사진", "이모티콘", "동영상", "보이스톡")
SYSTEM_MESSAGE_RE = re.compile(
    r"^(사진|이모티콘|동영상|보이스톡)(\s+\d+\s*(장|개))?$"
)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[가-힣A-Za-z]{2,}")
NUMERIC_MESSAGE_RE = re.compile(r"^[0-9\s.:\-~/]+$")
KAKAO_WORDCLOUD_COLORS = ["#3C1E1E", "#7A5C00", "#E67E22", "#F5A623", "#FFCD00"]
NEW_CONVERSATION_HOURS = 6
READ_IGNORE_HOURS = 1


def style_chart(fig, title: str, reverse_y: bool = False, showlegend: bool = False):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18, color="#191919")),
        template="plotly_white",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif", size=13, color="#333333"),
        margin=dict(l=10, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(250, 250, 248, 0.85)",
        hoverlabel=dict(bgcolor="white", font_size=13, font_color="#191919"),
        coloraxis_colorbar=dict(title=""),
        showlegend=showlegend,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    if reverse_y:
        fig.update_yaxes(showgrid=False, autorange="reversed")
    else:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig


def make_bar_chart(stats: pd.DataFrame, value_col: str, title: str, hover_label: str, ascending: bool = False):
    chart_df = stats.sort_values(value_col, ascending=ascending)
    fig = px.bar(
        chart_df,
        x=value_col,
        y="User",
        orientation="h",
        color=value_col,
        color_continuous_scale=KAKAO_SCALE,
        text=value_col,
    )
    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=f"<b>%{{y}}</b><br>{hover_label}: %{{x}}<extra></extra>",
    )
    fig.update_layout(xaxis_title=hover_label, yaxis_title="")
    fig.update_layout(height=max(420, 36 * len(chart_df) + 80))
    return style_chart(fig, title, reverse_y=True)


def make_time_bar_chart(
    chart_df: pd.DataFrame,
    x_col: str,
    title: str,
    hover_label: str,
    rotate_ticks: bool = False,
):
    peak_value = chart_df["메시지_개수"].max()
    chart_df = chart_df.copy()
    chart_df["구분"] = chart_df["메시지_개수"].eq(peak_value).map({True: "가장 활발", False: "그 외"})
    chart_df["라벨"] = chart_df["메시지_개수"].map(lambda n: "" if n == 0 else f"{int(n)}")
    fig = px.bar(
        chart_df,
        x=x_col,
        y="메시지_개수",
        color="구분",
        color_discrete_map={"가장 활발": "#E67E22", "그 외": "#FEE500"},
        text="라벨",
        category_orders={x_col: chart_df[x_col].tolist()},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=f"<b>%{{x}}</b><br>{hover_label}: %{{y}}개<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="메시지 개수",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    if rotate_ticks:
        fig.update_xaxes(tickangle=-45)
    return style_chart(fig, title, showlegend=True)


def make_monthly_line_chart(monthly_df: pd.DataFrame, title: str):
    fig = px.line(
        monthly_df,
        x="월",
        y="메시지_개수",
        markers=True,
        text="메시지_개수",
    )
    fig.update_traces(
        line=dict(color="#E67E22", width=3),
        marker=dict(size=10, color="#FEE500", line=dict(width=2, color="#E67E22")),
        fill="tozeroy",
        fillcolor="rgba(254, 229, 0, 0.28)",
        textposition="top center",
        hovertemplate="<b>%{x|%Y년 %m월}</b><br>메시지 개수: %{y}개<extra></extra>",
    )
    fig.update_layout(xaxis_title="", yaxis_title="메시지 개수", height=420)
    fig.update_xaxes(dtick="M1", tickformat="%Y년 %m월")
    return style_chart(fig, title)


def make_word_bar_chart(word_df: pd.DataFrame, title: str, hover_label: str):
    chart_df = word_df.sort_values("횟수", ascending=True)
    fig = px.bar(
        chart_df,
        x="횟수",
        y="단어",
        orientation="h",
        color="횟수",
        color_continuous_scale=KAKAO_SCALE,
        text="횟수",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=f"<b>%{{y}}</b><br>{hover_label}: %{{x}}회<extra></extra>",
    )
    fig.update_layout(xaxis_title=hover_label, yaxis_title="", height=max(420, 28 * len(chart_df) + 80))
    return style_chart(fig, title)


def get_korean_font_path() -> str | None:
    if platform.system() == "Windows":
        candidates = [
            Path(r"C:\Windows\Fonts\malgun.ttf"),
            Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        ]
    elif platform.system() == "Darwin":
        candidates = [
            Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
            Path("/Library/Fonts/AppleGothic.ttf"),
            Path("/System/Library/Fonts/AppleGothic.ttf"),
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]

    for font_path in candidates:
        if font_path.exists():
            return str(font_path)
    return None


def is_system_message(message: str) -> bool:
    text = message.strip()
    return text in SYSTEM_MESSAGES or bool(SYSTEM_MESSAGE_RE.fullmatch(text))


def is_numeric_message(message: str) -> bool:
    text = URL_RE.sub("", message).strip()
    return bool(text) and bool(NUMERIC_MESSAGE_RE.fullmatch(text))


def tokenize_message(message: str) -> list[str]:
    text = URL_RE.sub(" ", message)
    tokens = []
    for token in TOKEN_RE.findall(text):
        if len(token) < 2:
            continue
        if token.isascii():
            tokens.append(token.lower())
        else:
            tokens.append(token)
    return tokens


def messages_for_word_analysis(messages: pd.Series) -> pd.Series:
    cleaned = messages.astype(str)
    mask = ~(
        cleaned.map(is_system_message)
        | cleaned.map(is_numeric_message)
        | cleaned.map(lambda text: not URL_RE.sub("", text).strip())
    )
    return cleaned[mask]


def count_words(messages: pd.Series) -> Counter:
    counter: Counter = Counter()
    for message in messages_for_word_analysis(messages):
        counter.update(tokenize_message(message))
    return counter


def top_words_df(counter: Counter, n: int) -> pd.DataFrame:
    return pd.DataFrame(counter.most_common(n), columns=["단어", "횟수"])


def kakao_color_func(*_args, **_kwargs) -> str:
    return random.choice(KAKAO_WORDCLOUD_COLORS)


def make_wordcloud_image(counter: Counter, font_path: str):
    wordcloud = WordCloud(
        font_path=font_path,
        width=1400,
        height=700,
        background_color="white",
        max_words=120,
        prefer_horizontal=0.85,
        min_font_size=12,
        color_func=kakao_color_func,
        random_state=42,
    ).generate_from_frequencies(counter)
    return wordcloud.to_image()


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
    if minutes:
        return f"{minutes}분 {secs}초" if secs else f"{minutes}분"
    return f"{secs}초"


def build_reply_events(messages: pd.DataFrame) -> pd.DataFrame:
    ordered = messages.sort_values("datetime").reset_index(drop=True)
    records = []
    prev_user = None
    prev_time = None
    prev_message = None

    for row in ordered.itertuples(index=False):
        user = row.User
        ts = row.datetime
        message = row.Message

        if prev_user is not None:
            gap_seconds = (ts - prev_time).total_seconds()
            is_same_user = user == prev_user
            is_new_conversation = gap_seconds >= NEW_CONVERSATION_HOURS * 3600
            if not is_same_user and not is_new_conversation:
                records.append(
                    {
                        "replier": user,
                        "prev_user": prev_user,
                        "seconds": gap_seconds,
                        "is_ignore": gap_seconds >= READ_IGNORE_HOURS * 3600,
                        "reply_at": ts,
                        "reply_message": message,
                        "prev_message": prev_message,
                    }
                )

        prev_user = user
        prev_time = ts
        prev_message = message

    return pd.DataFrame.from_records(records)


def make_duration_bar_chart(stats: pd.DataFrame, value_col: str, title: str, hover_label: str):
    chart_df = stats.sort_values(value_col, ascending=True).copy()
    chart_df["label"] = chart_df[value_col].map(format_duration)
    fig = px.bar(
        chart_df,
        x=value_col,
        y="User",
        orientation="h",
        color=value_col,
        color_continuous_scale=KAKAO_SCALE,
        text="label",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate=f"<b>%{{y}}</b><br>{hover_label}: %{{x:.0f}}초<extra></extra>",
    )
    fig.update_layout(xaxis_title=hover_label, yaxis_title="")
    fig.update_layout(height=max(420, 36 * len(chart_df) + 80))
    return style_chart(fig, title, reverse_y=True)


PARTICIPANT_CARD_THEMES = [
    {"bg": "#FFFBEA", "border": "#FEE500", "bar": "#E6B800", "muted": "#8A7A00"},
    {"bg": "#FFF3E0", "border": "#FFCC80", "bar": "#EF6C00", "muted": "#A15C00"},
    {"bg": "#F3E5F5", "border": "#CE93D8", "bar": "#8E24AA", "muted": "#6A1B9A"},
    {"bg": "#E3F2FD", "border": "#90CAF9", "bar": "#1565C0", "muted": "#0D47A1"},
    {"bg": "#E8F5E9", "border": "#A5D6A7", "bar": "#2E7D32", "muted": "#1B5E20"},
    {"bg": "#FCE4EC", "border": "#F48FB1", "bar": "#C2185B", "muted": "#880E4F"},
]


def render_participant_card(name: str, count: int, total: int, avg_len: float, total_chars: int, theme: dict) -> str:
    share = (count / total * 100) if total else 0
    return f"""
    <div style="
        background: {theme['bg']};
        border: 1px solid {theme['border']};
        border-left: 7px solid {theme['bar']};
        border-radius: 16px;
        padding: 16px 18px 14px 18px;
        min-height: 168px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.06);
    ">
        <div style="font-size:12px; color:{theme['muted']}; font-weight:700; margin-bottom:6px;">참여자</div>
        <div style="font-size:16px; font-weight:800; color:#191919; line-height:1.35; min-height:44px;">
            {escape(name)}
        </div>
        <div style="font-size:28px; font-weight:800; color:{theme['bar']}; margin-top:8px;">
            {count:,}<span style="font-size:14px; color:#666666; font-weight:600;"> 개</span>
        </div>
        <div style="margin-top:8px; font-size:12px; color:#555555; line-height:1.5;">
            점유율 {share:.1f}% · 평균 {avg_len:.1f}자<br>총 {total_chars:,}자
        </div>
    </div>
    """


with st.sidebar:
    st.title("💬 카카오톡 대화 분석기")
    st.caption("카카오톡 CSV 대화를 업로드하면 참여자, 시간대, 단어, 답장 속도를 한눈에 볼 수 있습니다.")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

    df = None
    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
        except Exception as error:
            st.error(f"CSV를 불러오지 못했습니다: {error}")
            df = None
        else:
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                st.error(
                    "필수 칼럼이 없습니다: "
                    + ", ".join(missing_columns)
                    + f"\n현재 칼럼: {', '.join(df.columns.astype(str))}"
                )
                df = None

    if df is not None:
        df = df.copy()
        df["User"] = df["User"].fillna("알 수 없음").astype(str)
        df["Message"] = df["Message"].fillna("").astype(str)
        df["char_len"] = df["Message"].str.len()
        df["datetime"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = int(df["datetime"].isna().sum())
        time_df = df.dropna(subset=["datetime"]).copy()
        total_messages = len(df)
        participants = sorted(df["User"].unique())

        st.divider()
        st.subheader("📌 기본 정보")
        st.metric("메시지 수", f"{total_messages:,}개")
        st.markdown("**참여자**")
        st.write(", ".join(participants) if participants else "없음")

        if time_df.empty:
            st.markdown("**기간**")
            st.write("날짜를 읽을 수 없습니다.")
        else:
            start_date = time_df["datetime"].min()
            end_date = time_df["datetime"].max()
            st.markdown("**기간**")
            st.write(f"{start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}")
            if invalid_dates:
                st.caption(f"날짜 없음 {invalid_dates:,}개 제외")

    st.divider()
    st.subheader("📖 사용 방법")
    st.markdown(
        """
1. `Date`, `User`, `Message` 칼럼이 있는 CSV를 업로드하세요.
2. 위쪽에 메시지 수, 참여자, 기간이 표시됩니다.
3. 오른쪽 탭에서 통계, 시간, 단어, 답장 속도를 확인하세요.
4. 단어 분석에서는 사진/이모티콘, URL, 숫자만 있는 메시지는 제외됩니다.
        """
    )


if df is None:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하면 분석 탭이 나타납니다.")
    st.stop()

stats = (
    df.groupby("User", as_index=False)
    .agg(
        메시지_개수=("Message", "count"),
        총_글자_수=("char_len", "sum"),
        평균_메시지_길이=("char_len", "mean"),
    )
    .sort_values("메시지_개수", ascending=False)
)
stats["평균_메시지_길이"] = stats["평균_메시지_길이"].round(1)

longest_idx = df["char_len"].idxmax()
longest = df.loc[longest_idx]

tab_basic, tab_time, tab_words, tab_reply = st.tabs(
    ["📊 기본 통계", "⏰ 시간 분석", "💬 단어 분석", "⚡ 답장 속도"]
)

with tab_basic:
    if time_df.empty:
        period_label = "-"
        period_delta = "날짜 정보 없음"
        daily_avg = float(total_messages)
        daily_delta = "기간을 계산할 수 없음"
        busiest_label = "-"
        busiest_delta = "날짜 정보 없음"
    else:
        start_date = time_df["datetime"].min()
        end_date = time_df["datetime"].max()
        period_days = max(1, int((end_date.normalize() - start_date.normalize()).days) + 1)
        period_label = f"{period_days}일"
        period_delta = f"{start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d}"
        daily_avg = total_messages / period_days
        daily_delta = f"{period_days}일 기준"
        daily_counts = time_df.groupby(time_df["datetime"].dt.date).size()
        busiest_day = daily_counts.idxmax()
        busiest_label = f"{int(daily_counts.max()):,}개"
        busiest_delta = f"{busiest_day:%Y-%m-%d}"

    st.markdown("### 📌 핵심 지표")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("💬 전체 메시지 수", f"{total_messages:,}개", f"참여자 {len(participants)}명", border=True)
    kpi2.metric("📅 대화 기간", period_label, period_delta, border=True)
    kpi3.metric("📈 일평균 메시지 수", f"{daily_avg:.1f}개", daily_delta, border=True)
    kpi4.metric("🔥 가장 활발한 날", busiest_label, busiest_delta, border=True)

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    kpi5.metric("✍️ 평균 메시지 길이", f"{df['char_len'].mean():.1f}자", f"총 {int(df['char_len'].sum()):,}자", border=True)
    kpi6.metric("👥 대화 참여자 수", f"{len(participants):,}명", stats.iloc[0]["User"] + " 1위", border=True)
    kpi7.metric("🏆 가장 많이 말한 사람", stats.iloc[0]["User"], f"{int(stats.iloc[0]['메시지_개수']):,}개", border=True)
    kpi8.metric("📝 가장 긴 메시지", f"{int(longest['char_len']):,}자", longest["User"], border=True)

    st.markdown("### 👥 참여자별 통계")
    card_cols = st.columns(3)
    for index, row in enumerate(stats.to_dict("records")):
        theme = PARTICIPANT_CARD_THEMES[index % len(PARTICIPANT_CARD_THEMES)]
        with card_cols[index % 3]:
            st.markdown(
                render_participant_card(
                    str(row["User"]),
                    int(row["메시지_개수"]),
                    total_messages,
                    float(row["평균_메시지_길이"]),
                    int(row["총_글자_수"]),
                    theme,
                ),
                unsafe_allow_html=True,
            )

    st.markdown("### 📈 참여자 비교")
    st.plotly_chart(
        make_bar_chart(stats, "메시지_개수", "누가 가장 많이 말했나", "메시지 개수"),
        use_container_width=True,
    )

    length_col1, length_col2 = st.columns(2)
    with length_col1:
        st.plotly_chart(
            make_bar_chart(stats, "총_글자_수", "누가 글을 가장 많이 썼나", "총 글자 수"),
            use_container_width=True,
        )
    with length_col2:
        st.plotly_chart(
            make_bar_chart(stats, "평균_메시지_길이", "누가 메시지를 길게 쓰나", "평균 길이(자)"),
            use_container_width=True,
        )

    st.markdown("### 💌 가장 긴 메시지")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #FFFBE6 0%, #FFF3B0 100%);
            border: 1px solid #FEE500;
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: 0 8px 24px rgba(254, 229, 0, 0.18);
        ">
            <div style="color:#7A6A00; font-size:13px; margin-bottom:8px;">
                {escape(str(longest["User"]))} · {int(longest["char_len"]):,}자
                {f"· {escape(str(longest['Date']))}" if "Date" in longest and pd.notna(longest["Date"]) else ""}
            </div>
            <div style="white-space: pre-wrap; line-height: 1.6; color:#191919; font-size:16px;">
                {escape(longest["Message"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 데이터 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

with tab_time:
    if time_df.empty:
        st.warning("Date 칼럼을 datetime으로 파싱하지 못해 시간대 분석을 건너뜁니다.")
    else:
        if invalid_dates:
            st.caption(f"날짜를 읽지 못한 메시지 {invalid_dates:,}개는 시간 분석에서 제외했습니다.")

        hour_counts = (
            time_df.groupby(time_df["datetime"].dt.hour)
            .size()
            .reindex(range(24), fill_value=0)
            .rename_axis("시간")
            .reset_index(name="메시지_개수")
        )
        hour_counts["시간대"] = hour_counts["시간"].map(lambda hour: f"{hour:02d}시")

        weekday_counts = (
            time_df.groupby(time_df["datetime"].dt.dayofweek)
            .size()
            .reindex(range(7), fill_value=0)
        )
        weekday_df = pd.DataFrame(
            {"요일": WEEKDAYS_KO, "메시지_개수": weekday_counts.to_numpy()}
        )

        month_period = time_df["datetime"].dt.to_period("M")
        month_range = pd.period_range(month_period.min(), month_period.max(), freq="M")
        monthly_counts = (
            time_df.groupby(month_period)
            .size()
            .reindex(month_range, fill_value=0)
            .rename_axis("월")
            .reset_index(name="메시지_개수")
        )
        monthly_counts["월"] = monthly_counts["월"].dt.to_timestamp()

        peak_hour_row = hour_counts.loc[hour_counts["메시지_개수"].idxmax()]
        peak_weekday_row = weekday_df.loc[weekday_df["메시지_개수"].idxmax()]
        peak_hour = int(peak_hour_row["시간"])
        peak_hour_label = f"{peak_hour:02d}시대"

        peak1, peak2 = st.columns(2)
        peak1.metric(
            "🔥 가장 활발한 시간대",
            peak_hour_label,
            f"{int(peak_hour_row['메시지_개수']):,}개",
        )
        peak2.metric(
            "📅 가장 활발한 요일",
            peak_weekday_row["요일"] + "요일",
            f"{int(peak_weekday_row['메시지_개수']):,}개",
        )

        st.subheader("🕐 시간대별 메시지 분포")
        st.plotly_chart(
            make_time_bar_chart(
                hour_counts,
                "시간대",
                "0시부터 23시까지 대화량",
                "메시지 개수",
                rotate_ticks=True,
            ),
            use_container_width=True,
        )

        st.subheader("📆 요일별 메시지 분포")
        st.plotly_chart(
            make_time_bar_chart(weekday_df, "요일", "월요일부터 일요일까지 대화량", "메시지 개수"),
            use_container_width=True,
        )

        st.subheader("📉 월별 메시지 추이")
        st.plotly_chart(
            make_monthly_line_chart(monthly_counts, "월별 대화량 변화"),
            use_container_width=True,
        )

with tab_words:
    word_counter = count_words(df["Message"])
    top20 = top_words_df(word_counter, 20)
    font_path = get_korean_font_path()

    if top20.empty:
        st.info("필터 후 분석할 단어가 없습니다.")
    else:
        st.subheader("🏅 전체 메시지 단어 TOP 20")
        st.plotly_chart(
            make_word_bar_chart(top20, "가장 많이 쓴 단어", "등장 횟수"),
            use_container_width=True,
        )

        st.subheader("☁️ 워드클라우드")
        if font_path:
            st.caption(f"한글 폰트: {font_path}")
            st.image(make_wordcloud_image(word_counter, font_path), use_container_width=True)
        else:
            st.warning(
                "한글 폰트를 찾지 못해 워드클라우드를 만들지 못했습니다. "
                "Windows는 malgun.ttf, Mac은 AppleGothic이 필요합니다."
            )

        st.subheader("👤 참여자별 단어 TOP 10")
        user_options = stats["User"].tolist()
        selected_user = st.selectbox("참여자 선택", user_options)
        user_counter = count_words(df.loc[df["User"] == selected_user, "Message"])
        user_top10 = top_words_df(user_counter, 10)
        if user_top10.empty:
            st.info(f"{selected_user}님의 분석 가능한 단어가 없습니다.")
        else:
            st.plotly_chart(
                make_word_bar_chart(user_top10, f"{selected_user}님이 많이 쓴 단어", "등장 횟수"),
                use_container_width=True,
            )

    laugh_df = pd.DataFrame(
        {
            "User": df["User"],
            "k_count": df["Message"].str.contains("ㅋ", na=False).astype(int),
            "h_count": df["Message"].str.contains("ㅎ", na=False).astype(int),
        }
    )
    laugh_df = laugh_df.groupby("User", as_index=False).sum(numeric_only=True)
    laugh_df = laugh_df.sort_values(["k_count", "h_count"], ascending=False)

    k_chart = laugh_df.rename(columns={"k_count": "메시지_개수"})[["User", "메시지_개수"]]
    h_chart = laugh_df.rename(columns={"h_count": "메시지_개수"})[["User", "메시지_개수"]]

    laugh_col1, laugh_col2 = st.columns(2)
    with laugh_col1:
        st.subheader("ㅋㅋ ㅋ이 포함된 메시지")
        st.caption("ㅋㅋ, ㅋㅋㅋ처럼 ㅋ이 들어간 메시지 횟수")
        st.plotly_chart(
            make_bar_chart(k_chart, "메시지_개수", "누가 ㅋ을 많이 쓰나", "메시지 개수"),
            use_container_width=True,
        )
    with laugh_col2:
        st.subheader("ㅎㅎ ㅎ이 포함된 메시지")
        st.caption("ㅎㅎ, ㅎㅎㅎ처럼 ㅎ이 들어간 메시지 횟수")
        st.plotly_chart(
            make_bar_chart(h_chart, "메시지_개수", "누가 ㅎ을 많이 쓰나", "메시지 개수"),
            use_container_width=True,
        )

with tab_reply:
    st.caption(
        "같은 사람의 연속 메시지는 제외하고, 6시간 이상 공백은 새로운 대화로 보고 답장 시간에서 빼었습니다. "
        "카카오톡 내보내기는 분 단위라 같은 분의 답장은 0초로 표시됩니다."
    )

    if time_df.empty:
        st.warning("Date 칼럼을 datetime으로 파싱하지 못해 답장 속도 분석을 건너뜁니다.")
    else:
        reply_events = build_reply_events(time_df)
        if reply_events.empty:
            st.info("답장으로 볼 수 있는 메시지가 없습니다.")
        else:
            reply_stats = (
                reply_events.groupby("replier", as_index=False)
                .agg(
                    avg_seconds=("seconds", "mean"),
                    reply_count=("seconds", "count"),
                    ignore_count=("is_ignore", "sum"),
                )
                .sort_values("avg_seconds", ascending=True)
            )
            reply_stats["ignore_count"] = reply_stats["ignore_count"].astype(int)

            overall_avg = reply_events["seconds"].mean()
            fastest = reply_stats.iloc[0]
            ignore_total = int(reply_events["is_ignore"].sum())

            reply1, reply2, reply3 = st.columns(3)
            reply1.metric("⏱️ 평균 답장 시간", format_duration(overall_avg), f"{len(reply_events):,}회 답장")
            reply2.metric("🚀 가장 빠른 답장러", fastest["replier"], format_duration(fastest["avg_seconds"]))
            reply3.metric("👀 읽고 답장하지 않은 횟수", f"{ignore_total:,}회")

            st.subheader("⚡ 참여자별 평균 답장 속도")
            speed_chart = reply_stats.rename(columns={"replier": "User"})
            st.plotly_chart(
                make_duration_bar_chart(speed_chart, "avg_seconds", "답장이 빠를수록 왼쪽", "평균 답장 시간(초)"),
                use_container_width=True,
            )

            st.subheader("🏁 가장 빠른 답장 TOP 5")
            fastest_replies = reply_events.sort_values(["seconds", "reply_at"], ascending=True).head(5).copy()
            fastest_replies["답장 시간"] = fastest_replies["seconds"].map(format_duration)
            fastest_replies["상대방 메시지"] = (
                fastest_replies["prev_message"].astype(str).str.replace("\n", " ").str.slice(0, 80)
            )
            fastest_replies["답장 내용"] = (
                fastest_replies["reply_message"].astype(str).str.replace("\n", " ").str.slice(0, 80)
            )
            st.dataframe(
                fastest_replies.rename(
                    columns={"replier": "답장한 사람", "prev_user": "상대방", "reply_at": "답장 시각"}
                )[["답장한 사람", "상대방", "답장 시간", "답장 시각", "상대방 메시지", "답장 내용"]],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("🙈 읽고 답장하지 않은 메시지")
            ignore_chart = reply_stats.rename(
                columns={"replier": "User", "ignore_count": "메시지_개수"}
            ).sort_values("메시지_개수", ascending=False)
            st.caption("답장까지 1시간 이상 6시간 미만 걸린 횟수입니다.")
            st.plotly_chart(
                make_bar_chart(ignore_chart, "메시지_개수", "읽씹이 많은 사람", "읽씹 횟수"),
                use_container_width=True,
            )

            st.subheader("📋 참여자별 답장 상세")
            detail = reply_stats.copy()
            detail["avg_label"] = detail["avg_seconds"].map(format_duration)
            st.dataframe(
                detail.rename(
                    columns={
                        "replier": "참여자",
                        "avg_label": "평균 답장 시간",
                        "reply_count": "답장 횟수",
                        "ignore_count": "읽씹 횟수",
                    }
                )[["참여자", "평균 답장 시간", "답장 횟수", "읽씹 횟수"]],
                use_container_width=True,
                hide_index=True,
            )
