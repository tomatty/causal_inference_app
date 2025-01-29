# 必要なライブラリーのインストール
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from tqdm import tqdm
import os
import io
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module')))
import hashlib
from function import Function


# ページの設定
st.set_page_config(
    page_title = "効果検証分析ツール",
    page_icon = ":computer:"
)


st.sidebar.markdown('# 効果検証分析ツール')
# セッションステートのリフレッシュボタン
def reset_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
if st.sidebar.button('リフレッシュ'):
    reset_session_state()
    st.experimental_rerun()


st.write(f"### データセットの準備")
# データセットのURLリスト
DATASETS = {
    "CH2 Log Data": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch2_logdata.csv",
    "Lenta Dataset": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/lenta_dataset.csv",
    "Cluster Trial": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch3_cluster_trial.csv",
    "Stratified Trial": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch3_stratified_trial.csv",
    "AA Test": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch3_aatest_trial.csv",
    "Noncompliance AB Test": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch3_noncompliance_abtest.csv",
    "Organ Donations (Short)": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch4_organ_donations_short.csv",
    "Organ Donations (Full)": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch4_organ_donations_full.csv",
    "Coupon Data": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch5_coupon.csv",
    "Coupon Data v2": "https://raw.githubusercontent.com/HirotakeIto/intro_to_impact_evaluation_with_python/main/data/ch5_coupon_v2.csv",
}

# ユーザーがデータの取得方法を選択
option = st.radio("データの取得方法を選択してください", ("サンプルデータを選択", "ファイルをアップロード"))

# 空データを用意
data = None

# サンプルデータを選択した場合
if option == "サンプルデータを選択":
    # ユーザーが選択するセレクトボックス
    selected_dataset = st.selectbox("サンプルデータを選択してください", list(DATASETS.keys()))
    
    # 選択されたURL
    selected_url = DATASETS[selected_dataset]
    
    # データの読み込みと表示
    @st.cache_data
    def load_data(url):
        return pd.read_csv(url)
    
    data = load_data(selected_url)

# ファイルアップロードを選択した場合
elif option == "ファイルをアップロード":
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

# データセットが空でない場合
if data is not None:
    if st.checkbox("データフレームを表示"):
        st.dataframe(data)
    if st.checkbox("データフレームの統計情報を表示"):
        st.write(data.describe())
    if st.checkbox("相関行列を表示"):
        corr_matrix = data.corr()
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig_corr)


st.sidebar.markdown('---')
st.sidebar.markdown('### A/Aテストのリプレイ')


# AAテストのリプレイ
if 'aatest_clicked' not in st.session_state:
    st.session_state.aatest_clicked = False

if st.sidebar.button('A/Aテストのリプレイ'):
    st.session_state.aatest_clicked = not st.session_state.aatest_clicked

if st.session_state.aatest_clicked:
    st.markdown('---')
    st.write(f"### A/Aテストのリプレイ")

    # ユーザーがデータの取得方法を選択
    option = st.radio("回帰分析の手法を選択してください", ("クラスター頑健標準誤差を使用しない", "クラスター頑健標準誤差を使用する"))

    # クラスター頑健標準誤差を使用しない場合
    if option == "クラスター頑健標準誤差を使用しない":

        # カラム選択（hash_col, outcome）
        hash_col = st.selectbox("ハッシュ化に使用するカラムを選択してください", data.columns)
        outcome = st.selectbox("結果変数として使用するカラムを選択してください", data.columns)
        treatment_col = st.text_input("新たに生成するカラム名を指定してください", "is_treatment_in_aa")
        # シミュレーションの設定
        num_replays = st.number_input("リプレイ回数", min_value=100, max_value=1000, value=300, step=50)

        # 乱数生成器の作成
        def assign_treatment_randomly(hash_col, salt):
            return int(hashlib.sha256(f"{hash_col}_{salt}".encode()).hexdigest(), 16) % 2

        if st.button("A/Aテストを実行"):
                replays = []

                # リプレイの実行
                for i in range(num_replays):
                    salt = f"salt{i}"
                    data[treatment_col] = data[hash_col].apply(assign_treatment_randomly, salt=salt)
                    result = smf.ols(f"{outcome} ~ {treatment_col}", data=data).fit()
                    p_value = result.pvalues[treatment_col]
                    replays.append(p_value)

                # ヒストグラムの可視化
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(replays, bins=20, edgecolor="black", alpha=0.7)
                ax.set_xlabel("p-value")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of p-values")
                st.pyplot(fig)

                # コルモゴロフ–スミルノフ検定
                kstest_p = stats.kstest(replays, "uniform", args=(0, 1)).pvalue
                if kstest_p < 0.05:
                    st.error(f"A/Aテスト不合格（分布に差がある） p-value: {kstest_p:.5f}")
                else:
                    st.success(f"A/Aテスト合格（分布に差があるとは言えない） p-value: {kstest_p:.5f}")

    # クラスター頑健標準誤差を使用する場合
    if option == "クラスター頑健標準誤差を使用する":
        # カラム選択（hash_col, outcome）
        hash_col = st.selectbox("ハッシュ化に使用するカラムを選択してください(クラスター)", data.columns)
        outcome = st.selectbox("結果変数として使用するカラムを選択してください", data.columns)
        treatment_col = st.text_input("新たに生成するカラム名を指定してください", "is_treatment_in_aa")
        # シミュレーションの設定
        num_replays = st.number_input("リプレイ回数", min_value=100, max_value=1000, value=300, step=50)

        # ランダム生成器の作成
        def assign_treatment_randomly(hash_col, salt):
            return int(hashlib.sha256(f"{hash_col}_{salt}".encode()).hexdigest(), 16) % 2

        if st.button("A/Aテストを実行"):
                replays = []

                # リプレイの実行
                for i in range(num_replays):
                    salt = f"salt{i}"
                    data[treatment_col] = data[hash_col].apply(assign_treatment_randomly, salt=salt)
                    result = smf.ols(f"{outcome} ~ {treatment_col}", data=data).fit()
                    result_corrected = result.get_robustcov_results("cluster", groups=data[hash_col])
                    p_value = result_corrected.pvalues[result_corrected.model.exog_names.index(treatment_col)]
                    replays.append(p_value)

                # ヒストグラムの可視化
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(replays, bins=20, edgecolor="black", alpha=0.7)
                ax.set_xlabel("p-value")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of p-values")
                st.pyplot(fig)

                # コルモゴロフ–スミルノフ検定
                kstest_p = stats.kstest(replays, "uniform", args=(0, 1)).pvalue
                if kstest_p < 0.05:
                    st.error(f"A/Aテスト不合格（分布に差がある） p-value: {kstest_p:.5f}")
                else:
                    st.success(f"A/Aテスト合格（分布に差があるとは言えない） p-value: {kstest_p:.5f}")


st.sidebar.markdown('---')
st.sidebar.markdown('### A/Bテスト')


# 通常のA/Bテスト
if 'normal_abtest_clicked' not in st.session_state:
    st.session_state.normal_abtest_clicked = False

if st.sidebar.button('通常のA/Bテスト'):
    st.session_state.normal_abtest_clicked = not st.session_state.normal_abtest_clicked

if st.session_state.normal_abtest_clicked:
    st.markdown('---')
    st.write(f"### 通常のA/Bテスト")
    option = st.radio("ATEの計算方法を選択してください", ("集計比較", "回帰分析"))

    # 集計比較の場合
    if option == "集計比較":

    # カラム選択
        col1, col2 = st.columns(2)
        with col1:
            treatment_col = st.selectbox("トリートメント群を示すカラムを選択", data.columns)
        with col2:
            response_col = st.selectbox("結果変数を選択", data.columns)
        
        # トリートメント群ごとに集計
        df_result = data.groupby(treatment_col)[response_col].mean() * 100
        
        # 結果表示
        st.write("データフレーム")
        st.write(df_result)
        
        # ATEの計算とweltchのt検定
        df_treatment = data[data[treatment_col] == 1][response_col]
        df_control = data[data[treatment_col] == 0][response_col]
        ate = df_result[1] - df_result[0]
        t_stat, p_value = stats.ttest_ind(df_treatment, df_control)
        st.write("ATEの計算とweltchのt検定の結果")
        st.write(f"#### ATE: {ate:.5f}, t_statistic: {t_stat:.5f}, p_value: {p_value:.5f}")

    # 回帰分析の場合
    if option == "回帰分析":

        st.write("共変量バランステスト")
        is_treatment_col = st.selectbox("トリートメント群のカラムを選択してください", options=data.columns, index=0)
        selected_columns = st.multiselect(
        "集計するカラムを選択してください",
        options=[col for col in data.columns if col != is_treatment_col],
        )
        
        if selected_columns:
            # トリートメント群ごとに集計
            df_balance_test = data.groupby(is_treatment_col)[selected_columns].mean()
            st.dataframe(df_balance_test)
        
        # 回帰分析の実行
        formula = st.text_input("回帰式を入力してください", key='formula001')
        if formula:
            try:
                result = smf.ols(formula, data=data).fit()
                st.write("回帰分析の結果")
                st.text(result.summary())
            except Exception as e:
                st.error(f"エラー: {e}")


# クラスターA/Bテスト
if 'cluster_abtest_clicked' not in st.session_state:
    st.session_state.cluster_abtest_clicked = False

if st.sidebar.button('クラスターA/Bテスト'):
    st.session_state.cluster_abtest_clicked = not st.session_state.cluster_abtest_clicked

if st.session_state.cluster_abtest_clicked:
    st.markdown('---')
    st.write(f"### クラスターA/Bテスト")

    # カラム選択
    cluster_col = st.selectbox("クラスターを選択してください", data.columns)

    # クラスターの種類を表示
    unique_cluster = data[cluster_col].unique()
    st.write("クラスターの種類")
    st.write(unique_cluster)

    # 回帰分析の実行
    formula = st.text_input("回帰式を入力してください", key='formula002')
    if formula:
        try:
            result = smf.ols(formula, data=data).fit()
            result_corrected = result.get_robustcov_results("cluster", groups=data[cluster_col])
            result_corrected.summary()
            st.write("回帰分析の結果")
            st.text(result.summary())
        except Exception as e:
            st.error(f"エラー: {e}")


# 層化A/Bテスト
if 'stratified_abtest_clicked' not in st.session_state:
    st.session_state.stratified_abtest_clicked = False

if st.sidebar.button('層化A/Bテスト'):
    st.session_state.stratified_abtest_clicked = not st.session_state.stratified_abtest_clicked

if st.session_state.stratified_abtest_clicked:
    st.markdown('---')
    st.write(f"### 層化A/Bテスト")

    # 回帰分析の実行
    formula = st.text_input("回帰式を入力してください", key='formula003')
    if formula:
        try:
            result = smf.ols(formula, data=data).fit()
            st.write("回帰分析の結果")
            st.text(result.summary())
        except Exception as e:
            st.error(f"エラー: {e}")


st.sidebar.markdown('---')
st.sidebar.markdown('### RDD')
st.sidebar.write('作成中')


st.sidebar.markdown('---')
st.sidebar.markdown('### DID')
st.sidebar.write('作成中')