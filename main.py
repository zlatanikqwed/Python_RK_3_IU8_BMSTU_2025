import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
import hashlib
from fpdf import FPDF

warnings.filterwarnings('ignore')

# @file main.py
# @brief Main application file for the IT Monitoring and Analytics Dashboard.
# This Streamlit app provides real-time monitoring of server metrics and web application logs.

st.set_page_config(
    page_title="ИТ Мониторинг и Аналитика",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

st.markdown("""
<style>
.metric-card {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.critical {
    border-left: 4px solid #ff4b4b;
    background-color: #ffe6e6;
}
.warning {
    border-left: 4px solid #ffa500;
    background-color: #fff4e6;
}
.healthy {
    border-left: 4px solid #00cc96;
    background-color: #e6f7f2;
}
.main-header {
    text-align: center;
    font-size: 2.2rem;
    color: #1f77b4;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


def hash_dataframe(df: pd.DataFrame) -> str:
    """@brief Generate a unique hash for a DataFrame.
    @param df The DataFrame to hash.
    @return A 12-character hexadecimal string representing the hash.
    @note Returns empty string if df is None or empty.
    """
    if df is None or df.empty:
        return ""
    hash_vals = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.md5(hash_vals).hexdigest()[:12]


class ITMonitoringDashboard:
    """@class ITMonitoringDashboard
    @brief Main class for the IT Monitoring and Analytics Dashboard.
    Manages data loading, visualization, and analysis modules.
    """

    def __init__(self):
        """@brief Initialize the dashboard by syncing with session state.
        @note Metrics and logs are loaded from st.session_state if available.
        """
        self.metrics_df = st.session_state.get('metrics_df')
        self.logs_df = st.session_state.get('logs_df')
        self.metrics_hash = st.session_state.get('metrics_hash')
        self.logs_hash = st.session_state.get('logs_hash')

    def generate_sample_metrics(self, save_to_disk=True):
        """@brief Generate server metrics data.
        @param save_to_disk If True, saves the generated data to 'data/server_metrics.csv'.
        @return A pandas DataFrame containing simulated metrics.
        """
        dates = pd.date_range('2025-01-15 08:00:00', periods=48, freq='5min')
        servers = ['web-server-01', 'db-server-01', 'app-server-01', 'cache-server-01']
        data = []
        for date in dates:
            for server in servers:
                base_cpu = 40 if 'web' in server else 60 if 'db' in server else 30
                cpu = np.clip(np.random.normal(base_cpu, 15), 5, 95)
                base_memory = 70 if 'app' in server else 50 if 'db' in server else 35
                memory = np.clip(np.random.normal(base_memory, 12), 15, 90)
                disk = np.clip(np.random.normal(60, 20), 25, 95)

                if cpu > 85 or memory > 85 or disk > 90:
                    status = 'critical'
                elif cpu > 75 or memory > 75 or disk > 80:
                    status = 'warning'
                else:
                    status = 'healthy'

                data.append({
                    'timestamp': date,
                    'server_name': server,
                    'cpu_percent': round(cpu, 1),
                    'memory_percent': round(memory, 1),
                    'disk_usage_percent': round(disk, 1),
                    'network_in_mbps': round(np.random.uniform(10, 200), 1),
                    'network_out_mbps': round(np.random.uniform(5, 150), 1),
                    'disk_io_read': np.random.randint(50, 1500),
                    'disk_io_write': np.random.randint(30, 1000),
                    'status': status
                })
        df = pd.DataFrame(data)
        if save_to_disk:
            path = DATA_DIR / "server_metrics.csv"
            df.to_csv(path, index=False)
            st.sidebar.success(f"Сохранено: `{path}`")
        return df

    def generate_sample_logs(self, save_to_disk=True):
        """@brief Generate web application log data.
        @param save_to_disk If True, saves the generated data to 'data/web_app_logs.csv'.
        @return A pandas DataFrame containing simulated logs.
        """
        dates = pd.date_range('2025-01-15 08:00:00', periods=200, freq='30s')
        servers = ['web-server-01', 'app-server-01', 'db-server-01']
        endpoints = ['/api/users', '/api/products', '/api/orders', '/api/login',
                     '/api/health', '/api/reports', '/api/search', '/api/payment']
        methods = ['GET', 'POST', 'PUT', 'DELETE']
        data = []

        for date in dates:
            server = np.random.choice(servers)
            endpoint = np.random.choice(endpoints)
            method = np.random.choice(methods)

            status_probs = [0.85, 0.08, 0.04, 0.02, 0.01]
            http_status = np.random.choice([200, 401, 404, 500, 503], p=status_probs)

            if http_status >= 500:
                level = 'ERROR'
            elif http_status >= 400:
                level = 'WARNING'
            else:
                level = np.random.choice(['INFO', 'DEBUG'], p=[0.8, 0.2])

            if http_status >= 500:
                response_time = np.random.uniform(100, 500)
            elif 'db' in server:
                response_time = np.random.uniform(50, 150)
            else:
                response_time = np.random.uniform(10, 80)

            data.append({
                'timestamp': date,
                'level': level,
                'server_name': server,
                'client_ip': f"192.168.1.{np.random.randint(1, 255)}",
                'http_method': method,
                'endpoint': endpoint,
                'http_status': http_status,
                'response_time_ms': round(response_time),
                'user_agent': np.random.choice([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Chrome/120.0.0.0 Safari/537.36",
                    "Firefox/121.0",
                    "PostmanRuntime/7.0.0"
                ]),
                'message': f"{method} request to {endpoint} completed with status {http_status}"
            })
        df = pd.DataFrame(data)
        if save_to_disk:
            path = DATA_DIR / "web_app_logs.csv"
            df.to_csv(path, index=False)
            st.sidebar.success(f"Сохранено: `{path}`")
        return df

    def load_data(self):
        """@brief Load data automatically from disk or generate new data if files don't exist (regeneration).
        @note Data is stored in st.session_state for persistence across interactions.
        """
        st.sidebar.header("Данные")

        metrics_path = DATA_DIR / "server_metrics.csv"
        logs_path = DATA_DIR / "web_app_logs.csv"

        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.session_state['metrics_df'] = df
                st.session_state['metrics_hash'] = hash_dataframe(df)
                st.sidebar.success("Метрики загружены с диска")
            except Exception as e:
                st.sidebar.error(f"Ошибка чтения метрик: {e}")
                metrics_path.unlink(missing_ok=True)
                df = self.generate_sample_metrics(save_to_disk=True)
                st.session_state['metrics_df'] = df
                st.session_state['metrics_hash'] = hash_dataframe(df)
        else:
            st.sidebar.info("Файл метрик не найден. Генерация...")
            df = self.generate_sample_metrics(save_to_disk=True)
            st.session_state['metrics_df'] = df
            st.session_state['metrics_hash'] = hash_dataframe(df)

        if st.sidebar.button("Перегенерировать метрики"):
            df = self.generate_sample_metrics(save_to_disk=True)
            st.session_state['metrics_df'] = df
            st.session_state['metrics_hash'] = hash_dataframe(df)
            st.sidebar.success("Метрики перегенерированы и сохранены")

        if logs_path.exists():
            try:
                df = pd.read_csv(logs_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.session_state['logs_df'] = df
                st.session_state['logs_hash'] = hash_dataframe(df)
                st.sidebar.success("Логи загружены с диска")
            except Exception as e:
                st.sidebar.error(f"Ошибка чтения логов: {e}")
                logs_path.unlink(missing_ok=True)
                df = self.generate_sample_logs(save_to_disk=True)
                st.session_state['logs_df'] = df
                st.session_state['logs_hash'] = hash_dataframe(df)
        else:
            st.sidebar.info("Файл логов не найден. Генерация...")
            df = self.generate_sample_logs(save_to_disk=True)
            st.session_state['logs_df'] = df
            st.session_state['logs_hash'] = hash_dataframe(df)

        if st.sidebar.button("Перегенерировать логи"):
            df = self.generate_sample_logs(save_to_disk=True)
            st.session_state['logs_df'] = df
            st.session_state['logs_hash'] = hash_dataframe(df)
            st.sidebar.success("Логи перегенерированы и сохранены")

        self.metrics_df = st.session_state['metrics_df']
        self.logs_df = st.session_state['logs_df']
        self.metrics_hash = st.session_state['metrics_hash']
        self.logs_hash = st.session_state['logs_hash']

    def generate_pdf_report(self):
        """@brief Generate a PDF report.
        @throws Exception if fonts cannot be loaded.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        current_dir = Path(__file__).parent
        font_dir = current_dir / "fonts"

        try:
            pdf.add_font("DejaVu", "", str(font_dir / "DejaVuSans.ttf"))
            pdf.add_font("DejaVu", "B", str(font_dir / "DejaVuSans-Bold.ttf"))
            pdf.set_font("DejaVu", size=12)
            use_cyrillic = True
        except Exception:
            pdf.set_font("Arial", size=12)
            use_cyrillic = False

        title = "Отчёт по ИТ-мониторингу" if use_cyrillic else "IT Monitoring Report"
        pdf.set_font("DejaVu" if use_cyrillic else "Arial", 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)

        info_title = "Общая информация" if use_cyrillic else "General Info"
        pdf.set_font("DejaVu" if use_cyrillic else "Arial", 'B', 12)
        pdf.cell(0, 10, info_title, ln=True)
        pdf.set_font("DejaVu" if use_cyrillic else "Arial", size=10)
        metrics_count = len(self.metrics_df) if self.metrics_df is not None else 0
        logs_count = len(self.logs_df) if self.logs_df is not None else 0
        pdf.cell(0, 8, f"Записей в метриках: {metrics_count}" if use_cyrillic else f"Metric records: {metrics_count}", ln=True)
        pdf.cell(0, 8, f"Записей в логах: {logs_count}" if use_cyrillic else f"Log records: {logs_count}", ln=True)
        pdf.ln(5)

        incidents = []
        if self.metrics_df is not None and self.logs_df is not None:
            high_cpu = self.metrics_df[self.metrics_df['cpu_percent'] > 85]
            errors_5xx = self.logs_df[self.logs_df['http_status'] >= 500]
            for _, cpu_row in high_cpu.iterrows():
                t = cpu_row['timestamp']
                related = errors_5xx[
                    (errors_5xx['timestamp'] >= t - timedelta(minutes=2)) &
                    (errors_5xx['timestamp'] <= t + timedelta(minutes=2))
                ]
                if not related.empty:
                    incidents.append({
                        'timestamp': t.strftime('%Y-%m-%d %H:%M'),
                        'server': cpu_row['server_name'],
                        'cpu': cpu_row['cpu_percent'],
                        'errors': len(related)
                    })

        inc_title = "Обнаруженные инциденты" if use_cyrillic else "Detected Incidents"
        pdf.set_font("DejaVu" if use_cyrillic else "Arial", 'B', 12)
        pdf.cell(0, 10, inc_title, ln=True)
        pdf.set_font("DejaVu" if use_cyrillic else "Arial", size=10)
        if incidents:
            for inc in incidents[:5]:
                text = f"- {inc['timestamp']}: {inc['server']} | CPU: {inc['cpu']}% | Ошибок: {inc['errors']}"
                if not use_cyrillic:
                    text = f"- {inc['timestamp']}: {inc['server']} | CPU: {inc['cpu']}% | Errors: {inc['errors']}"
                pdf.cell(0, 8, text, ln=True)
        else:
            msg = "Инциденты не обнаружены." if use_cyrillic else "No incidents detected."
            pdf.cell(0, 8, msg, ln=True)

        if self.logs_df is not None:
            error_endpoints = self.logs_df[self.logs_df['http_status'] >= 500]['endpoint'].value_counts().head(3)
            top_err_title = "Топ эндпоинтов с ошибками 5xx" if use_cyrillic else "Top Endpoints with 5xx Errors"
            pdf.ln(5)
            pdf.set_font("DejaVu" if use_cyrillic else "Arial", 'B', 12)
            pdf.cell(0, 10, top_err_title, ln=True)
            pdf.set_font("DejaVu" if use_cyrillic else "Arial", size=10)
            if not error_endpoints.empty:
                for ep, cnt in error_endpoints.items():
                    pdf.cell(0, 8, f"- {ep}: {cnt} ошибок" if use_cyrillic else f"- {ep}: {cnt} errors", ln=True)
            else:
                msg = "Ошибок 5xx не найдено." if use_cyrillic else "No 5xx errors found."
                pdf.cell(0, 8, msg, ln=True)

        return bytes(pdf.output(dest='S'))

    def show_metrics_dashboard(self):
        """@brief Display the System Metrics module.
        Shows KPI cards, time-series charts, and heatmap for server metrics.
        """
        st.header("Мониторинг системных метрик")
        if self.metrics_df is None:
            st.warning("Загрузите данные метрик или сгенерируйте пример.")
            return

        if self.metrics_hash:
            st.caption(f"ID данных: `{self.metrics_hash}`")

        col1, col2, col3 = st.columns(3)
        with col1:
            servers = st.multiselect(
                "Серверы:", self.metrics_df['server_name'].unique(),
                default=self.metrics_df['server_name'].unique()
            )
        with col2:
            min_date = self.metrics_df['timestamp'].min().date()
            max_date = self.metrics_df['timestamp'].max().date()
            date_range = st.date_input("Диапазон дат:", value=[min_date, max_date])
        with col3:
            cpu_thresh = st.slider("Порог CPU (%)", 0, 100, 80)
            mem_thresh = st.slider("Порог Memory (%)", 0, 100, 75)

        filtered = self.metrics_df[
            (self.metrics_df['server_name'].isin(servers)) &
            (self.metrics_df['timestamp'].dt.date >= date_range[0]) &
            (self.metrics_df['timestamp'].dt.date <= date_range[1])
        ]
        if filtered.empty:
            st.error("Нет данных для отображения.")
            return

        st.subheader("Текущие метрики")
        latest = filtered.sort_values('timestamp').groupby('server_name').last()
        cols = st.columns(len(latest))
        for i, (server, row) in enumerate(latest.iterrows()):
            cpu_status = 'critical' if row['cpu_percent'] > cpu_thresh else 'warning' if row['cpu_percent'] > cpu_thresh - 10 else 'healthy'
            mem_status = 'critical' if row['memory_percent'] > mem_thresh else 'warning' if row['memory_percent'] > mem_thresh - 10 else 'healthy'
            with cols[i]:
                st.markdown(f'<div class="metric-card {cpu_status}">', unsafe_allow_html=True)
                st.metric(f"{server} - CPU", f"{row['cpu_percent']}%")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-card {mem_status}">', unsafe_allow_html=True)
                st.metric(f"{server} - RAM", f"{row['memory_percent']}%")
                st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Динамика метрик")
        tabs = st.tabs(["CPU", "Memory", "Disk", "Network"])
        with tabs[0]:
            fig = px.line(filtered, x='timestamp', y='cpu_percent', color='server_name')
            fig.add_hline(y=cpu_thresh, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            fig = px.line(filtered, x='timestamp', y='memory_percent', color='server_name')
            fig.add_hline(y=mem_thresh, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            fig = px.line(filtered, x='timestamp', y='disk_usage_percent', color='server_name')
            st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Входящий трафик", "Исходящий трафик"))
            for server in servers:
                sdata = filtered[filtered['server_name'] == server]
                fig.add_trace(go.Scatter(x=sdata['timestamp'], y=sdata['network_in_mbps'], name=f"{server} IN"), row=1, col=1)
                fig.add_trace(go.Scatter(x=sdata['timestamp'], y=sdata['network_out_mbps'], name=f"{server} OUT"), row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Heatmap загрузки CPU")
        heatmap_df = filtered.pivot_table(values='cpu_percent', index=filtered['timestamp'].dt.strftime('%H:%M'), columns='server_name', aggfunc='mean')
        fig = px.imshow(heatmap_df.T, aspect='auto', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

    def show_logs_analyzer(self):
        """@brief Display the Logs Analysis module.
        Shows statistics, top endpoints, and allows filtering and exporting logs.
        """
        st.header("Анализ логов веб-приложений")
        if self.logs_df is None:
            st.warning("Загрузите данные логов или сгенерируйте пример.")
            return

        if self.logs_hash:
            st.caption(f"ID данных: `{self.logs_hash}`")

        col1, col2, col3 = st.columns(3)
        with col1:
            servers = st.multiselect("Серверы:", self.logs_df['server_name'].unique(), default=self.logs_df['server_name'].unique())
        with col2:
            levels = st.multiselect("Уровни:", self.logs_df['level'].unique(), default=self.logs_df['level'].unique())
        with col3:
            query = st.text_input("Поиск по сообщениям:")

        filtered = self.logs_df[
            (self.logs_df['server_name'].isin(servers)) &
            (self.logs_df['level'].isin(levels))
        ]
        if query:
            filtered = filtered[filtered['message'].str.contains(query, case=False, na=False)]
        if filtered.empty:
            st.error("Нет логов по выбранным фильтрам.")
            return

        st.subheader("Статистика HTTP-статусов")
        status_counts = filtered['http_status'].value_counts().reset_index()
        status_counts.columns = ['http_status', 'count']
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            fig = px.pie(status_counts, values='count', names='http_status')
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            total = len(filtered)
            success = len(filtered[filtered['http_status'] < 400])
            client_err = len(filtered[(filtered['http_status'] >= 400) & (filtered['http_status'] < 500)])
            server_err = len(filtered[filtered['http_status'] >= 500])
            st.metric("Успешно", success)
            st.metric("Ошибки клиента (4xx)", client_err)
            st.metric("Ошибки сервера (5xx)", server_err)
        with col_c:
            avg_time = filtered['response_time_ms'].mean()
            err_rate = (server_err / total * 100) if total > 0 else 0
            st.metric("Всего запросов", total)
            st.metric("Среднее время", f"{avg_time:.1f} мс")
            st.metric("Rate ошибок", f"{err_rate:.1f}%")

        st.subheader("Анализ эндпоинтов")
        tabs = st.tabs(["Топ эндпоинтов", "Ошибки", "Время ответа"])
        with tabs[0]:
            top = filtered.groupby('endpoint').size().nlargest(10)
            fig = px.bar(x=top.index, y=top.values, labels={'x': 'Эндпоинт', 'y': 'Запросы'})
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            errors = filtered[filtered['http_status'] >= 400]
            if not errors.empty:
                err_group = errors.groupby(['endpoint', 'http_status']).size().reset_index(name='count')
                fig = px.sunburst(err_group, path=['http_status', 'endpoint'], values='count')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ошибок не найдено.")
        with tabs[2]:
            slow = filtered.groupby('endpoint')['response_time_ms'].mean().nlargest(10)
            fig = px.bar(x=slow.index, y=slow.values, labels={'x': 'Эндпоинт', 'y': 'Среднее время ответа (мс)'})
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Просмотр логов"):
            st.dataframe(filtered.sort_values('timestamp', ascending=False), use_container_width=True)
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Экспорт в CSV", csv, "logs_export.csv", "text/csv")

    def show_integration_analytics(self):
        """@brief Display the Integration Analytics module.
        """
        st.header("Интеграционная аналитика")
        if self.metrics_df is None or self.logs_df is None:
            st.warning("Загрузите оба набора данных для корреляционного анализа.")
            return

        metrics_agg = self.metrics_df.set_index('timestamp').groupby('server_name').resample('5min').mean(numeric_only=True).reset_index()
        logs_agg = self.logs_df.set_index('timestamp').resample('5min').agg({
            'http_status': 'count',
            'response_time_ms': 'mean'
        }).rename(columns={'http_status': 'request_count', 'response_time_ms': 'avg_response_time'}).reset_index()

        server = st.selectbox("Выберите сервер:", self.metrics_df['server_name'].unique())
        m = metrics_agg[metrics_agg['server_name'] == server]
        merged = pd.merge(m, logs_agg, on='timestamp', how='inner')

        if merged.empty:
            st.info("Недостаточно совпадающих данных для анализа.")
            return

        st.subheader("Корреляция метрик и нагрузки")
        corr_cpu_req = merged['cpu_percent'].corr(merged['request_count'])
        corr_mem_resp = merged['memory_percent'].corr(merged['avg_response_time'])

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(merged, x='request_count', y='cpu_percent', trendline='ols',
                             title=f"CPU vs Запросы (ρ = {corr_cpu_req:.2f})")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(merged, x='memory_percent', y='avg_response_time', trendline='ols',
                             title=f"RAM vs Время ответа (ρ = {corr_mem_resp:.2f})")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Обнаружение инцидентов")
        high_cpu = self.metrics_df[self.metrics_df['cpu_percent'] > 85]
        errors_5xx = self.logs_df[self.logs_df['http_status'] >= 500]
        incidents = []
        for _, cpu_row in high_cpu.iterrows():
            t = cpu_row['timestamp']
            related = errors_5xx[
                (errors_5xx['timestamp'] >= t - timedelta(minutes=2)) &
                (errors_5xx['timestamp'] <= t + timedelta(minutes=2))
            ]
            if not related.empty:
                incidents.append({
                    'timestamp': t,
                    'server': cpu_row['server_name'],
                    'cpu_usage': cpu_row['cpu_percent'],
                    'error_count': len(related),
                    'endpoints': ', '.join(related['endpoint'].unique())
                })

        if incidents:
            df_inc = pd.DataFrame(incidents)
            st.dataframe(df_inc, use_container_width=True)
        else:
            st.success("Инцидентов не обнаружено.")

        st.subheader("Рекомендации")
        slow_endpoints = self.logs_df.groupby('endpoint')['response_time_ms'].mean().nlargest(3)
        error_endpoints = self.logs_df[self.logs_df['http_status'] >= 500]['endpoint'].value_counts().head(3)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Самые медленные эндпоинты:")
            for ep, t in slow_endpoints.items():
                st.write(f"• `{ep}`: {t:.1f} мс")
        with col2:
            st.write("Эндпоинты с ошибками (5xx):")
            for ep, cnt in error_endpoints.items():
                st.write(f"• `{ep}`: {cnt} ошибок")

        st.subheader("Экспорт отчёта")
        if st.button("Сгенерировать PDF-отчёт"):
            pdf_bytes = self.generate_pdf_report()
            st.download_button(
                label="Скачать PDF",
                data=pdf_bytes,
                file_name="it_monitoring_report.pdf",
                mime="application/pdf"
            )


def main():
    """@brief Main entry point of the application.
    Initializes session state, creates the dashboard instance, and runs the UI.
    """
    if 'metrics_df' not in st.session_state:
        st.session_state['metrics_df'] = None
    if 'logs_df' not in st.session_state:
        st.session_state['logs_df'] = None
    if 'metrics_hash' not in st.session_state:
        st.session_state['metrics_hash'] = None
    if 'logs_hash' not in st.session_state:
        st.session_state['logs_hash'] = None

    st.markdown('<h1 class="main-header">Комплексный мониторинг ИТ-инфраструктуры</h1>', unsafe_allow_html=True)
    app = ITMonitoringDashboard()
    app.load_data()

    page = st.sidebar.radio("Модули:", ["Системные метрики", "Анализ логов", "Интеграционная аналитика", "О проекте"])

    if page == "Системные метрики":
        app.show_metrics_dashboard()
    elif page == "Анализ логов":
        app.show_logs_analyzer()
    elif page == "Интеграционная аналитика":
        app.show_integration_analytics()
    else:
        st.subheader("О проекте")
        st.markdown("""
        **Комплексный дашборд мониторинга ИТ-инфраструктуры и анализа логов**

        **Функционал:**
        - Мониторинг ресурсов серверов (CPU, RAM, Disk, Network)
        - Анализ логов веб-приложений (HTTP-статусы, ошибки, производительность)
        - Корреляционный анализ и обнаружение инцидентов
        - Генерация рекомендаций и PDF-отчётов
        - Сохранение данных между перезагрузками (в рамках сессии)

        **Технологии:**  
        Python, Streamlit, Pandas, Plotly, FPDF2

        **Файлы для загрузки:**  
        - `server_metrics.csv` - метрики серверов  
        - `web_app_logs.csv` - логи веб-приложений
        """)


if __name__ == "__main__":
    main()