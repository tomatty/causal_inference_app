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
from statsmodels.stats.power import TTestIndPower
from rdrobust import rdbwselect, rdplot, rdrobust
import rddensity
from causallib.estimation import IPW, PropensityMatching,StratifiedStandardization
from causallib.evaluation import evaluate
from sklearn.linear_model import LogisticRegression
from causalimpact import CausalImpact
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

#リフレッシュボタンの実装
def reset_session_state():
    st.session_state.clear()  # 全キーを安全に削除

if st.sidebar.button('リフレッシュ'):
    reset_session_state()
    st.rerun() # st.experimental_rerun()は使えない


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
    "Time Series Sample Data": "https://raw.githubusercontent.com/tomatty/dataset/refs/heads/main/timeseries_data.csv"
}

# ユーザーがデータの取得方法を選択
# `key` を明示的に初期化
#if "data_selection_method" not in st.session_state:
#    st.session_state["data_selection_method"] = "サンプルデータを選択"  # デフォルト値を設定
option = st.radio("データの取得方法を選択してください", ("サンプルデータを選択", "ファイルをアップロード"), key="data_selection_method")

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
st.sidebar.markdown('### A/Bテスト')


# A/Bテスト設計
if 'abtest_plan_clicked' not in st.session_state:
    st.session_state.abtest_plan_clicked = False

if st.sidebar.button('A/Bテスト設計'):
    st.session_state.abtest_plan_clicked = not st.session_state.abtest_plan_clicked

if st.session_state.abtest_plan_clicked:
    st.markdown('---')
    st.write(f"### A/Bテスト設計")

    # A/Bテスト設計書用のデータフレームを作成
    df = pd.DataFrame(
        [
        {"設計項目": "施策内容", "設計内容": None},
        {"設計項目": "ゴールメトリクス", "設計内容": None},
        {"設計項目": "ガードレールメトリクス", "設計内容": None},
        {"設計項目": "割当単位", "設計内容": None},
        {"設計項目": "割当比率", "設計内容": None},
        {"設計項目": "サンプルサイズ", "設計内容": None},
        {"設計項目": "SUTVAを満たしているか", "設計内容": None},
        {"設計項目": "その他", "設計内容": None},
    ]
    )
    edited_df = st.data_editor(df, num_rows="dynamic", width=1000)

    if st.checkbox("サンプルサイズを計算"):

        # パラメータを指定
        alpha = st.number_input("有意水準 α", min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%.3f")
        power = st.number_input("検出力 (1-β)", min_value=0.5, max_value=1.0, value=0.8, step=0.05, format="%.2f")
        effect_size = st.number_input("効果量 (Cohen's d)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, format="%.1f")
        allocation_ratio = st.number_input("割当比率（トリートメント群 : コントロール群）", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")

        # サンプルサイズの計算
        analysis = TTestIndPower()
        control_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided', ratio=allocation_ratio)
        control_size = int(np.ceil(control_size))  # コントロール群のサイズ（切り上げ）
        treatment_size = int(np.ceil(control_size * allocation_ratio))  # トリートメント群のサイズ

        # 計算結果の表示
        col1, col2, col3 = st.columns(3)
        col1.metric("トリートメント群", f"{treatment_size}", border=True)
        col2.metric("コントロール群", f"{control_size}", border=True)
        col3.metric("合計サンプルサイズ", f"{treatment_size + control_size}", border=True)
        st.info("注：statsmodels.stats.powerのTTestIndPowerを使用して計算しています")


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
                # プログレスバーの初期化
                progress_bar = st.progress(0)
                replays = []

                # リプレイの実行
                for i in range(num_replays):
                    salt = f"salt{i}"
                    data[treatment_col] = data[hash_col].apply(assign_treatment_randomly, salt=salt)
                    result = smf.ols(f"{outcome} ~ {treatment_col}", data=data).fit()
                    p_value = result.pvalues[treatment_col]
                    replays.append(p_value)

                    # プログレスバーを更新
                    progress_bar.progress((i + 1) / num_replays)

                # 処理完了後、バーを100%にしてメッセージを表示
                progress_bar.progress(1.0)
                st.success("リプレイ完了")

                # ヒストグラムの可視化
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(replays, bins=20, edgecolor="black", alpha=0.7)
                ax.set_xlabel("p-value")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of p-values")
                st.pyplot(fig)

                # コルモゴロフ–スミルノフ検定
                st.write("コルモゴロフ–スミルノフ検定の結果：")
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
                # プログレスバーの初期化
                progress_bar = st.progress(0)
                replays = []

                # リプレイの実行
                for i in range(num_replays):
                    salt = f"salt{i}"
                    data[treatment_col] = data[hash_col].apply(assign_treatment_randomly, salt=salt)
                    result = smf.ols(f"{outcome} ~ {treatment_col}", data=data).fit()
                    result_corrected = result.get_robustcov_results("cluster", groups=data[hash_col])
                    p_value = result_corrected.pvalues[result_corrected.model.exog_names.index(treatment_col)]
                    replays.append(p_value)

                    # プログレスバーを更新
                    progress_bar.progress((i + 1) / num_replays)

                # 処理完了後、バーを100%にしてメッセージを表示
                progress_bar.progress(1.0)
                st.success("リプレイ完了")

                # ヒストグラムの可視化
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(replays, bins=20, edgecolor="black", alpha=0.7)
                ax.set_xlabel("p-value")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of p-values")
                st.pyplot(fig)

                # コルモゴロフ–スミルノフ検定
                st.write("コルモゴロフ–スミルノフ検定の結果：")
                kstest_p = stats.kstest(replays, "uniform", args=(0, 1)).pvalue
                if kstest_p < 0.05:
                    st.error(f"A/Aテスト不合格（分布に差がある） p-value: {kstest_p:.5f}")
                else:
                    st.success(f"A/Aテスト合格（分布に差があるとは言えない） p-value: {kstest_p:.5f}")


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
        col1, col2, col3 = st.columns(3)
        col1.metric("ATE", f"{ate:.5f}", border=True)
        col2.metric("T_statistic", f"{t_stat:.5f}", border=True)
        col3.metric("P_value", f"{p_value:.5f}", border=True)
        #st.write(f"#### ATE: {ate:.5f}, t_statistic: {t_stat:.5f}, p_value: {p_value:.5f}")

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
st.sidebar.markdown('### 観察データ分析')


# 傾向スコア
if 'propensity_clicked' not in st.session_state:
    st.session_state.propensity_clicked = False

if st.sidebar.button('傾向スコア'):
    st.session_state.propensity_clicked = not st.session_state.propensity_clicked

if st.session_state.propensity_clicked:
    st.markdown('---')
    st.write(f"### 傾向スコア")

    option = st.radio("傾向スコアの手法を選択してください", ("マッチング", "IPW"))

    # マッチング
    if option == "マッチング":

        # 欠損値の確認
        if data.isnull().sum().sum() > 0:
            st.warning("データに欠損値が含まれています。欠損値は中央値で補完されます。")

        # ユーザーが選択するためのUIを作成
        columns = data.columns.tolist()
        selected_x = st.multiselect("特徴量を選択", columns)
        selected_a = st.selectbox("トリートメント変数を選択", columns)
        selected_y = st.selectbox("結果変数を選択", columns)

        #if selected_x and selected_a and selected_y:
        if st.button("推定"):
            # データを分割
            X = data[selected_x]
            a = data[selected_a]
            y = data[selected_y]

            # 欠損値処理（X のみ）
            X.fillna(X.median(), inplace=True)

            # 傾向スコアモデルを定義
            learner = LogisticRegression(solver="liblinear", class_weight="balanced")

            # 傾向スコアマッチング
            pm = PropensityMatching(learner=learner)
            pm.fit(X, a, y)

            # ATEの計算
            outcomes = pm.estimate_population_outcome(X, a)
            effect = pm.estimate_effect(outcomes[1], outcomes[0])

            # 結果の表示
            st.write("#### 推定結果")
            col1, col2, col3 = st.columns(3)
            col1.metric("未処置群の平均アウトカム", f"{outcomes[0]:.3f}", border=True)
            col2.metric("処置群の平均アウトカム", f"{outcomes[1]:.3f}", border=True)
            col3.metric("平均処置効果 (ATE)", f"{effect['diff']:.3f}", border=True)

    # IPW
    if option == "IPW":

        # 欠損値の確認
        if data.isnull().sum().sum() > 0:
            st.warning("データに欠損値が含まれています。欠損値は中央値で補完されます。")
        
        # ユーザーが選択するためのUIを作成
        columns = data.columns.tolist()
        selected_x = st.multiselect("特徴量を選択", columns)
        selected_a = st.selectbox("トリートメント変数を選択", columns)
        selected_y = st.selectbox("結果変数を選択", columns)
        asmd_thresh = st.number_input("ASMDのカットオフ値", value=0.1, step=0.01)

        #if selected_x and selected_a and selected_y:
        if st.button("推定"):
            # データを分割
            X = data[selected_x]
            a = data[selected_a]
            y = data[selected_y]

            # 欠損値処理（X のみ）
            X.fillna(X.median(), inplace=True)

            # 傾向スコアモデルを定義
            learner = LogisticRegression(solver="liblinear", class_weight="balanced")

            #傾向スコアを算出し、IPWを実施
            ipw = IPW(learner = learner)
            ipw.fit(X, a)

            # ATEの計算
            outcomes = ipw.estimate_population_outcome(X, a, y)
            effect = ipw.estimate_effect(outcomes[1], outcomes[0])

            # 結果の表示
            st.write("#### 推定結果")
            col1, col2, col3 = st.columns(3)
            col1.metric("未処置群の平均アウトカム", f"{outcomes[0]:.3f}", border=True)
            col2.metric("処置群の平均アウトカム", f"{outcomes[1]:.3f}", border=True)
            col3.metric("平均処置効果 (ATE)", f"{effect['diff']:.3f}", border=True)

            # 共変量のバランスを表示
            st.write("#### 共変量のバランス")
            results = evaluate(ipw, X, a, y)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            results.plot_covariate_balance(kind="love", ax=ax, thresh=asmd_thresh)
            st.pyplot(fig)

            # 重みの分布を表示
            st.write("#### 分布")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            results.plot_weight_distribution(ax=ax)
            st.pyplot(fig)


# DID
if 'did_clicked' not in st.session_state:
    st.session_state.did_clicked = False

if st.sidebar.button('DID'):
    st.session_state.did_clicked = not st.session_state.did_clicked

if st.session_state.did_clicked:
    st.markdown('---')
    st.write(f"### DID")

    # グループ集計を表示
    if st.checkbox("グループ集計を表示"):
        # ユーザーに選択させる
        selected_groupby = st.multiselect("グループを選択してください", data.columns)
        selected_agg = st.selectbox("集計キーを選択してください", data.columns)

        # グループ化と集計
        if selected_groupby:
            grouped_data = data.groupby(selected_groupby)[selected_agg].mean().reset_index()
            st.write(grouped_data)
        else:
            st.write("カラムを選択してください")
    
    # プレトレンドテスト
    if st.checkbox("プレトレンドテスト"):
            fig, ax = plt.subplots()
            sns.lineplot(data=grouped_data, x=selected_groupby[0], y=selected_agg, 
                         hue=selected_groupby[1] if len(selected_groupby) > 1 else None, ax=ax)
            st.pyplot(fig)
    
    # 回帰分析の実行
    option = st.radio("クラスターの有無を選択してください", ("クラスターなし", "クラスターあり"))

    if option == "クラスターなし":
        formula = st.text_input("回帰式を入力してください", key='formula004')
        if formula:
            try:
                result = smf.ols(formula, data=data).fit()
                st.write("回帰分析の結果")
                st.text(result.summary())
            except Exception as e:
                st.error(f"エラー: {e}")
    
    if option == "クラスターあり":
        # クラスターを選択
        cluster_col = st.selectbox("クラスターを選択してください", data.columns)
        formula = st.text_input("回帰式を入力してください", key='formula004')
        if formula:
            try:
                result = smf.ols(formula, data=data).fit()
                st.write("回帰分析の結果")
                # クラスター頑健標準誤差を計算
                result_corrected = result.get_robustcov_results("cluster", groups=data[cluster_col])
                st.text(result_corrected.summary())
            except Exception as e:
                st.error(f"エラー: {e}")


# Causal Impact
if 'causal_impact_clicked' not in st.session_state:
    st.session_state.causal_impact_clicked = False

if st.sidebar.button('Causal Impact'):
    st.session_state.causal_impact_clicked = not st.session_state.causal_impact_clicked

if st.session_state.causal_impact_clicked:
    st.markdown('---')
    st.write(f"### Causal Impact")
    #st.info("作成中のため使用できません。リリースまでしばらくお待ちください。", icon=None)
    #st.error("ModuleNotFoundError: No module named 'tf_keras'", icon=None)

    # 期間情報を入力させる
    col1, col2 = st.columns(2)
    with col1:
        pre_start = st.number_input('施策前の開始日', min_value=1, value=1)
    with col2:
        pre_end = st.number_input('施策前の終了日', min_value=pre_start, value=90)

    col1, col2 = st.columns(2)
    with col1:
        post_start = st.number_input('施策後の開始日', min_value=pre_end+1, value=91)
    with col2:
        post_end = st.number_input('施策後の終了日', min_value=post_start, value=119)

    # CausalImpactの学習・推定
    if st.button('Causal Impact 実行'):
        ci = CausalImpact(data, [pre_start, pre_end], [post_start, post_end])

        # 結果の可視化
        st.write('Causal Impact 結果:')
        st.pyplot(ci.plot())

        #推定結果の要点を出力
        st.write(ci.summary())


# Sharp RDD
if 'sharp_rdd_clicked' not in st.session_state:
    st.session_state.sharp_rdd_clicked = False

if st.sidebar.button('Sharp RDD'):
    st.session_state.sharp_rdd_clicked = not st.session_state.sharp_rdd_clicked

if st.session_state.sharp_rdd_clicked:
    st.markdown('---')
    st.write(f"### Sharp RDD ※ノンパラメトリックな局所多項式回帰")

    if data is not None:
        
        # ユーザー入力
        y_col = st.selectbox("従属変数 (y)", data.columns, index=0)
        x_col = st.selectbox("独立変数 (x)", data.columns, index=1)
        c_value = st.number_input("カットオフ値 (c)", value=10000, step=100)
        binselect = st.selectbox("ビニング手法", ["es", "qsmv", "qser"], index=0)
        ci_value = st.slider("信頼区間 (%)", min_value=50, max_value=99, value=95)

        # 推定
        result_rdd = rdrobust(y=data[y_col], x=data[x_col], c=c_value, all=True)
        st.write("#### RD推定結果:")
        st.write(result_rdd.__dict__)
        
        # プロット
        st.write("#### RDプロット:")
        fig = plt.figure()
        rdplot(y=data[y_col], x=data[x_col], binselect=binselect, c=c_value, ci=ci_value,
            title="Causal Effects", y_label=y_col, x_label=x_col)
        st.pyplot(plt.gcf()) #gcf()を使用しないと表示されない

        if st.checkbox("McCrary検定（連続性の検定）"):
            # ヒストグラムの表示
            st.write("#### ヒストグラム:")
            fig, ax = plt.subplots()  # 明示的に Figure, Axes を取得
            ax.hist(data[x_col], bins=30, edgecolor="black")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

            # McCraryの検定
            st.write("#### McCraryの検定結果（連続性の検定）:")
            result_mccrary = rddensity.rddensity(
                X=data[x_col], c=c_value
            )
            st.write(result_mccrary.__dict__)

        # 共変量のバランステスト
        if st.checkbox("共変量のバランステスト(カットオフ前後での同質性を確認)"):
            # バランステスト
            selected_covs = st.multiselect("バランステストの共変量を選択してください", data.columns)
            covs = data[selected_covs]
            balance = pd.DataFrame(columns=["RD Effect", "Robust p-val"], index=selected_covs)

            for z in covs.columns:
                est = rdrobust(y=covs[z], x=data[x_col], c=c_value)
                balance.loc[z, "RD Effect"] = est.Estimate["tau.us"].values[0]
                balance.loc[z, "Robust p-val"] = est.pv.iloc[2].values[0]

            st.write("#### バランステストの結果:")
            st.dataframe(balance)


# Fuzzy RDD
if 'fuzzy_rdd_clicked' not in st.session_state:
    st.session_state.fuzzy_rdd_clicked = False

if st.sidebar.button('Fuzzy RDD'):
    st.session_state.fuzzy_rdd_clicked = not st.session_state.fuzzy_rdd_clicked

if st.session_state.fuzzy_rdd_clicked:
    st.markdown('---')
    st.write(f"### fuzzy RDD ※rdrobustを使用")

    if data is not None:
        
        # ユーザー入力
        y_col = st.selectbox("従属変数 (y)", data.columns, index=0)
        x_col = st.selectbox("独立変数 (x)", data.columns, index=1)
        fuzzy_col = st.selectbox("トリートメント変数", data.columns, index=2)
        c_value = st.number_input("カットオフ値 (c)", value=10000, step=100)

        # グラフの表示
        st.write("#### グラフ:")
        # ラベルごとにデータを分ける
        df_label0 = data[data[fuzzy_col] == 0]
        df_label1 = data[data[fuzzy_col] == 1]
        
        # 散布図の作成
        fig, ax = plt.subplots()
        ax.scatter(df_label0[x_col], df_label0[y_col], c="gray", label="0", marker="s")
        ax.scatter(df_label1[x_col], df_label1[y_col], c="gray", label="1", marker="x")
        
        # 閾値線の描画（閾値は適宜調整）
        threshold = c_value
        ax.axvline(x=threshold, color="black", linestyle="--")
        
        # 軸ラベルと凡例の追加
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        
        # Streamlit で表示
        st.pyplot(fig)

        # 推定
        st.write("#### RD推定結果:")
        result_fuzzy_rdd = rdrobust(y=data[y_col], x=data[x_col], fuzzy=data[fuzzy_col], c=c_value, all=True)
        #st.write(result_fuzzy_rdd)
        st.write(result_fuzzy_rdd.__dict__)
