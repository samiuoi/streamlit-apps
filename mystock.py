import streamlit as st
import pandas as pd
import datetime
import os
import pickle
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vnstock import Vnstock, register_user

# --- 1. CẤU HÌNH HỆ THỐNG & API KEY ---
st.set_page_config(page_title="Bộ Lọc MA, BB & StochRSI", layout="wide", page_icon="📉")

st.markdown(
    """
    <style>
    div[data-testid="stDialog"] > div {
        width: 95vw !important;
        max-width: 95vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

API_KEY = "vnstock_afe1347bb9ea2d856a7a9390024b33d1"
os.environ['VNSTOCK_API_KEY'] = API_KEY
register_user(API_KEY)

LIST_FILE = 'stock.list'
CACHE_FILE = 'stock_cache_ma_v1.pkl'


# --- 2. HÀM TÍNH TOÁN CÁC CHỈ BÁO ---
def calculate_indicators(df, pivot_timeframe='Tháng'):
    # 1. Tính MA9 và MA26
    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['MA26'] = df['Close'].rolling(window=26).mean()

    df['SMA_Vol_20'] = df['Volume'].rolling(window=20).mean()

    # 2. Tính Bollinger Bands (20, 2)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_BB'] = df['SMA_20'] - (df['STD_20'] * 2)

    # 3. Tính Stoch RSI (TradingView Standard: 14, 14, 3, 3)
    length_rsi = 14
    length_stoch = 14
    smooth_k = 3
    smooth_d = 3

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rma_gain = gain.ewm(alpha=1 / length_rsi, min_periods=length_rsi, adjust=False).mean()
    rma_loss = loss.ewm(alpha=1 / length_rsi, min_periods=length_rsi, adjust=False).mean()
    rs = rma_gain / rma_loss
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(window=length_stoch).min()
    max_rsi = rsi.rolling(window=length_stoch).max()
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    df['StochRSI_%K'] = stoch_rsi.rolling(window=smooth_k).mean()
    df['StochRSI_%D'] = df['StochRSI_%K'].rolling(window=smooth_d).mean()

    # 4. Tính Pivot Points linh hoạt theo Timeframe
    if pivot_timeframe == 'Tháng':
        period_str = 'ME'
    elif pivot_timeframe == 'Tuần':
        period_str = 'W-MON'
    else:
        period_str = 'D'

    df['TempPeriod'] = df.index.to_period(period_str.replace('ME', 'M').replace('W-MON', 'W'))

    period_data = df.groupby('TempPeriod').agg({
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })

    period_data['Prev_High'] = period_data['High'].shift(1)
    period_data['Prev_Low'] = period_data['Low'].shift(1)
    period_data['Prev_Close'] = period_data['Close'].shift(1)

    P = (period_data['Prev_High'] + period_data['Prev_Low'] + period_data['Prev_Close']) / 3
    period_data['Pivot'] = P
    period_data['R1'] = (2 * P) - period_data['Prev_Low']
    period_data['S1'] = (2 * P) - period_data['Prev_High']
    period_data['R2'] = P + (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['S2'] = P - (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['R3'] = P + 2 * (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['S3'] = P - 2 * (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['R4'] = P + 3 * (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['S4'] = P - 3 * (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['R5'] = P + 4 * (period_data['Prev_High'] - period_data['Prev_Low'])
    period_data['S5'] = P - 4 * (period_data['Prev_High'] - period_data['Prev_Low'])

    pivot_cols = ['Pivot', 'R1', 'R2', 'R3', 'R4', 'R5', 'S1', 'S2', 'S3', 'S4', 'S5']

    df = df.join(period_data[pivot_cols], on='TempPeriod')
    df.drop(columns=['TempPeriod'], inplace=True)

    # 5. Tính Ichimoku Cloud (Mây Ichi)
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

    # 6. Tính toán Trạng thái nền (Base / Breakout / Breakdown)
    window_base = 15  # Xét đỉnh/đáy của 15 phiên trước
    df['Base_High'] = df['High'].rolling(window=window_base).max().shift(1)
    df['Base_Low'] = df['Low'].rolling(window=window_base).min().shift(1)
    df['Base_Width'] = (df['Base_High'] - df['Base_Low']) / df['Base_Low']

    return df


@st.cache_data
def load_tickers_from_file():
    if not os.path.exists(LIST_FILE):
        with open(LIST_FILE, 'w') as f:
            f.write("SSI\nVND\nMBB\nFPT\nHPG\nBVH")
    with open(LIST_FILE, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers


tickers = load_tickers_from_file()

if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                st.session_state.market_data = pickle.load(f)
        except Exception:
            pass

if 'pivot_tf' not in st.session_state:
    st.session_state.pivot_tf = 'Tháng'

# --- 3. SIDEBAR ĐIỀU KHIỂN & TẢI DỮ LIỆU ---
with st.sidebar:
    st.header("⚙️ Điều khiển & Cấu hình")
    st.write(f"Đang theo dõi **{len(tickers)}** mã.")
    st.markdown("---")

    selected_pivot = st.selectbox(
        "📊 Khung thời gian Pivot:",
        options=['Ngày', 'Tuần', 'Tháng'],
        index=['Ngày', 'Tuần', 'Tháng'].index(st.session_state.pivot_tf)
    )
    if selected_pivot != st.session_state.pivot_tf:
        st.session_state.pivot_tf = selected_pivot

    st.markdown("---")

    delay_time = 1.0

    if st.button("🔄 Cập nhật Dữ liệu Mới", width='stretch', type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        estimated_time = round((len(tickers) * delay_time) / 60, 1)

        st.info(f"⏳ Đang tải dữ liệu... (Dự kiến {estimated_time} phút)")

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=180)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        new_data = {}
        for i, ticker in enumerate(tickers):
            status_text.text(f"Đang tải {ticker}... ({i + 1}/{len(tickers)})")
            try:
                stock = Vnstock().stock(symbol=ticker, source='KBS')
                df = stock.quote.history(start=start_str, end=end_str)

                if df is not None and not df.empty:
                    df.columns = [col.capitalize() for col in df.columns]
                    if 'Time' in df.columns:
                        df['Time'] = pd.to_datetime(df['Time'])
                        df.set_index('Time', inplace=True)
                    df = df.sort_index()
                    new_data[ticker] = {'raw_data': df}
            except Exception:
                pass

            progress_bar.progress((i + 1) / len(tickers))
            if i < len(tickers) - 1:
                time.sleep(delay_time)

        status_text.text("✅ Hoàn tất!")
        st.session_state.market_data = new_data
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(new_data, f)
        st.rerun()

# --- 4. XỬ LÝ DỮ LIỆU ---
current_pivot_tf = st.session_state.pivot_tf
summary_list = []

for ticker, info in st.session_state.market_data.items():
    if isinstance(info, dict) and 'raw_data' in info:
        raw_df = info['raw_data'].copy()
    else:
        continue

    valid_df = raw_df.dropna(subset=['Close']).copy()
    if len(valid_df) > 30:
        df = calculate_indicators(valid_df, pivot_timeframe=current_pivot_tf)

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        close_price = last_row['Close']
        low_price = last_row['Low']
        vol = last_row['Volume']
        sma_vol_20 = last_row.get('SMA_Vol_20', np.nan)
        lower_bb = last_row['Lower_BB']
        stoch_k = last_row.get('StochRSI_%K', np.nan)
        stoch_d = last_row.get('StochRSI_%D', np.nan)

        ma9 = last_row.get('MA9', np.nan)
        ma26 = last_row.get('MA26', np.nan)
        prev_ma9 = prev_row.get('MA9', np.nan)

        base_high = last_row.get('Base_High', np.nan)
        base_low = last_row.get('Base_Low', np.nan)
        base_width = last_row.get('Base_Width', np.nan)

        pivot_levels = {
            'P': last_row['Pivot'],
            'R1': last_row['R1'], 'R2': last_row['R2'], 'R3': last_row['R3'], 'R4': last_row['R4'],
            'R5': last_row['R5'],
            'S1': last_row['S1'], 'S2': last_row['S2'], 'S3': last_row['S3'], 'S4': last_row['S4'], 'S5': last_row['S5']
        }

        closest_pivot_str = "Chưa có"
        if pd.notna(pivot_levels['P']):
            closest_key = min(pivot_levels, key=lambda k: abs(pivot_levels[k] - close_price))
            closest_val = pivot_levels[closest_key]
            closest_pivot_str = f"{closest_key} ({closest_val:,.2f})"

        diff_bb = (low_price - lower_bb) if pd.notna(lower_bb) else 0
        diff_ma = (ma9 - ma26) if pd.notna(ma9) and pd.notna(ma26) else 0

        trend_ma9 = "➖ Không rõ"
        if pd.notna(ma9) and pd.notna(prev_ma9):
            if ma9 > prev_ma9:
                trend_ma9 = "🟢 Vòng lên"
            elif ma9 < prev_ma9:
                trend_ma9 = "🔴 Vòng xuống"
            else:
                trend_ma9 = "➖ Đi ngang"

        # Tính toán V20
        v20_val = 0.0
        if pd.notna(vol) and pd.notna(sma_vol_20) and sma_vol_20 > 0:
            v20_val = vol / sma_vol_20

        # ----------------------------------------------------
        # TÍNH TOÁN MA CROSS VÀ ĐẾM SỐ NGÀY LIÊN TIẾP
        # ----------------------------------------------------
        ma_cross_str = "➖ Không rõ"
        ma9_vals = df['MA9'].values
        ma26_vals = df['MA26'].values

        if len(ma9_vals) > 0 and pd.notna(ma9_vals[-1]) and pd.notna(ma26_vals[-1]):
            # Trạng thái hiện tại
            if ma9_vals[-1] > ma26_vals[-1]:
                curr_ma_state = 1  # Nằm trên
            elif ma9_vals[-1] < ma26_vals[-1]:
                curr_ma_state = -1  # Nằm dưới
            else:
                curr_ma_state = 0  # Bằng nhau

            # Đếm số ngày liên tiếp
            ma_days_count = 0
            if curr_ma_state != 0:
                for i in range(len(ma9_vals) - 1, -1, -1):
                    if pd.isna(ma9_vals[i]) or pd.isna(ma26_vals[i]):
                        break

                    if curr_ma_state == 1 and ma9_vals[i] > ma26_vals[i]:
                        ma_days_count += 1
                    elif curr_ma_state == -1 and ma9_vals[i] < ma26_vals[i]:
                        ma_days_count += 1
                    else:
                        break

            if curr_ma_state == 1:
                ma_cross_str = f"🟢 Nằm trên ({ma_days_count})"
            elif curr_ma_state == -1:
                ma_cross_str = f"🔴 Nằm dưới ({ma_days_count})"
            else:
                ma_cross_str = "➖ Cắt nhau"

        # ----------------------------------------------------
        # TÍNH TOÁN MÂY ICHI VÀ SỐ NGÀY DUY TRÌ TRẠNG THÁI
        # ----------------------------------------------------
        may_ichi_str = "➖ Không rõ"
        closes = df['Close'].values
        span_as = df['Senkou_Span_A'].values
        span_bs = df['Senkou_Span_B'].values

        if len(closes) > 0 and pd.notna(span_as[-1]) and pd.notna(span_bs[-1]):
            # Trạng thái phiên hiện tại
            curr_top = max(span_as[-1], span_bs[-1])
            curr_bottom = min(span_as[-1], span_bs[-1])

            if closes[-1] > curr_top:
                curr_state = 1  # Trên
            elif closes[-1] < curr_bottom:
                curr_state = -1  # Dưới
            else:
                curr_state = 0  # Trong

            # Đếm lùi số ngày
            days_count = 0
            for i in range(len(closes) - 1, -1, -1):
                if pd.isna(span_as[i]) or pd.isna(span_bs[i]):
                    break

                top = max(span_as[i], span_bs[i])
                bottom = min(span_as[i], span_bs[i])

                if closes[i] > top:
                    state = 1
                elif closes[i] < bottom:
                    state = -1
                else:
                    state = 0

                if state == curr_state:
                    days_count += 1
                else:
                    break

            if curr_state == 1:
                may_ichi_str = f"🟢 Trên mây ({days_count})"
            elif curr_state == -1:
                may_ichi_str = f"🔴 Dưới mây ({days_count})"
            else:
                may_ichi_str = f"🟡 Trong mây ({days_count})"

        # Xác định trạng thái nền/trend
        if pd.notna(ma9) and pd.notna(ma26):
            if ma9 > ma26:
                trang_thai_nen = "📈 Đang leo dốc"
            else:
                trang_thai_nen = "📉 Rơi tự do"
        else:
            trang_thai_nen = "➖ Không rõ"

        if pd.notna(base_high) and pd.notna(base_low) and pd.notna(base_width):
            if base_width <= 0.08 and base_low <= close_price <= base_high:
                trang_thai_nen = "🟡 Đi nền"
            elif base_width <= 0.15 and close_price > base_high:
                if v20_val > 1.2:
                    trang_thai_nen = "🟢 Vừa Break (Vol to)"
                else:
                    trang_thai_nen = "🟢 Vừa Break"
            elif base_width <= 0.15 and close_price < base_low:
                trang_thai_nen = "🔴 Vừa Thủng"

        summary_list.append({
            "Mã CK": ticker,
            "Giá Cuối": round(close_price, 2),
            "Trạng Thái Nền": trang_thai_nen,
            "Gần Pivot": closest_pivot_str,
            "MA9 - MA26": round(diff_ma, 2),
            "MA Cross": ma_cross_str,
            "Xu hướng MA9": trend_ma9,
            "Mây Ichi": may_ichi_str,
            "Low - Lower BB": round(diff_bb, 2),
            "StochRSI %K": round(stoch_k, 2) if pd.notna(stoch_k) else 0.00,
            "StochRSI %D": round(stoch_d, 2) if pd.notna(stoch_d) else 0.00,
            "V20": round(v20_val, 2),
            "Khối lượng": f"{int(vol):,}"
        })

summary_df = pd.DataFrame(summary_list)


# --- 5. POPUP BIỂU ĐỒ ---
@st.dialog("Đồ thị Phân tích", width="large")
def show_chart_popup(ticker, pivot_tf):
    info = st.session_state.market_data.get(ticker)
    if info and 'raw_data' in info:
        df_plot = info['raw_data'].iloc[-150:].copy()
        df_plot = calculate_indicators(df_plot, pivot_timeframe=pivot_tf)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.25, 0.75])

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot['Upper_BB'], line=dict(color='rgba(173,216,230,0.5)', width=1),
                       name='Upper BB'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot['Lower_BB'], line=dict(color='rgba(173,216,230,0.5)', width=1),
                       fill='tonexty', fillcolor='rgba(173,216,230,0.1)', name='Lower BB'), row=1, col=1)

        # MAs
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA9'], line=dict(color='#FFA500', width=1.5), name='MA9'),
                      row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot['MA26'], line=dict(color='#8A2BE2', width=1.5), name='MA26'), row=1,
            col=1)

        # Candlesticks
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'],
                                     close=df_plot['Close'], name='Giá'), row=1, col=1)

        colors = ['#26a69a' if r['Close'] > r['Open'] else '#ef5350' for i, r in df_plot.iterrows()]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=colors, name='Vol'), row=2, col=1)
        fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width='stretch')


# --- 6. HIỂN THỊ ---
st.title(f"📐 Lọc MA, BB & StochRSI")

if summary_df.empty:
    st.info("Bấm nút Cập nhật ở Sidebar để bắt đầu.")
else:
    search = st.text_input("🔍 Tìm mã:", "").upper()
    filtered = summary_df[summary_df['Mã CK'].str.contains(search)] if search else summary_df

    st.markdown(
        f"💡 *Mẹo: Pivot đang tính theo khung **{current_pivot_tf}**. Cột **Low - Lower BB** nếu âm chứng tỏ nến đã xuyên thủng dải dưới BB.*")


    def color_stoch(val):
        try:
            val = float(val)
            if val >= 80:
                return 'color: #FF4500; font-weight: bold'
            elif val <= 20:
                return 'color: #32CD32; font-weight: bold'
            return 'color: #808080'
        except:
            return 'color: #808080'


    def color_trend(val):
        val_str = str(val)
        if 'Break' in val_str:
            return 'color: #00FF00; font-weight: bold'
        elif 'leo dốc' in val_str:
            return 'color: #32CD32; font-style: italic'
        elif 'Đi nền' in val_str:
            return 'color: #FFD700; font-weight: bold'
        elif 'Thủng' in val_str:
            return 'color: #FF0000; font-weight: bold'
        elif 'Rơi tự do' in val_str:
            return 'color: #FF4500; font-style: italic'
        return 'color: #808080'


    # Áp dụng định dạng cho bảng
    event = st.dataframe(
        filtered.style.map(
            lambda x: 'color: #32CD32; font-weight: bold' if x > 0 else 'color: #FF4500; font-weight: bold',
            subset=['MA9 - MA26'])
        .map(lambda x: 'color: #32CD32; font-weight: bold' if 'Nằm trên' in str(x) else (
            'color: #FF4500; font-weight: bold' if 'Nằm dưới' in str(x) else 'color: #808080'),
             subset=['MA Cross'])
        .map(lambda x: 'color: #32CD32; font-weight: bold' if 'Vòng lên' in str(x) else (
            'color: #FF4500; font-weight: bold' if 'Vòng xuống' in str(x) else 'color: #808080'),
             subset=['Xu hướng MA9'])
        .map(lambda x: 'color: #32CD32; font-weight: bold' if 'Trên' in str(x) else (
            'color: #FFD700; font-weight: bold' if 'Trong' in str(x) else (
                'color: #FF4500; font-weight: bold' if 'Dưới' in str(x) else 'color: #808080')),
             subset=['Mây Ichi'])
        .map(color_trend, subset=['Trạng Thái Nền'])
        .map(lambda x: 'color: #ef5350; font-weight: bold' if 'R' in str(x) else (
            'color: #26a69a; font-weight: bold' if 'S' in str(x) else 'color: #ffa726; font-weight: bold'),
             subset=['Gần Pivot'])
        .map(lambda x: 'color: #FF8C00; font-weight: bold' if x < 0 else 'color: #808080', subset=['Low - Lower BB'])
        .map(color_stoch, subset=['StochRSI %K', 'StochRSI %D'])
        .format({
            'Giá Cuối': '{:.2f}',
            'MA9 - MA26': '{:.2f}',
            'Low - Lower BB': '{:.2f}',
            'StochRSI %K': '{:.2f}',
            'StochRSI %D': '{:.2f}',
            'V20': '{:.2f}'
        }),
        width='stretch', hide_index=True, height=500, on_select="rerun", selection_mode="single-row"
    )

    if event and len(event.selection.rows) > 0:
        show_chart_popup(filtered.iloc[event.selection.rows[0]]['Mã CK'], current_pivot_tf)