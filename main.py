import logging

import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

import data

ALIVE = 1
DEAD = 0
FEMALE = 0
MALE = 1

# Survived,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked
val_names = [
    "乗客のクラス(Pclass): 1が高級",
    "性別(Gender): 0が女性, 1が男性",
    "年齢(Age)",
    "乗船していた兄弟、配偶者の人数(SibSp)",
    "乗船していた両親、子供の人数(Parch)",
    "運賃(Fare)",
    "乗船した港(Embarked): 0がCherbourg, 1がQueenstown, 2がSouthampton",
]

st.set_page_config(
    page_title="Titanic Analysis App",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s,%(message)s")


DATA_SOURCE = "./data/titanic.csv"


@st.cache_data
def load_full_data():
    data = pd.read_csv(DATA_SOURCE)
    return data


@st.cache_data
def load_num_data():
    data = pd.read_csv(DATA_SOURCE)
    rows = ["Survived"]
    data = data.drop(rows, axis=1)
    return data


@st.cache_data
def load_ML_data(feature1, feature2, train_num=600):
    df = load_full_data()
    X = df[[feature1, feature2]]
    y = df.Survived  # yはSurvivedの列の値

    train_num = 600
    train_X = X[:train_num]
    test_X = X[train_num:]
    train_y = y[:train_num]
    test_y = y[train_num:]
    return (train_X, test_X, train_y, test_y)


def main():

    # ログをとるときのみコメントを外す
    # If username is already initialized, don't do anything
    # if 'username' not in st.session_state or st.session_state.username == 'default':
    #     st.session_state.username = 'default'
    #     input_name()
    #     st.stop()
    # if 'username' not in st.session_state:
    #     st.session_state.username = 'test'

    # 個別のログをとるときはinputを受け取るので以下は不要
    st.session_state.username = "test"

    if "page" not in st.session_state:
        st.session_state.page = "input_name"  # usernameつける時こっち

    # --- page選択ラジオボタン
    st.sidebar.markdown("## ページを選択")
    page = st.sidebar.radio("", ("データ表示", "グラフを表示"))
    if page == "データ表示":
        st.session_state.page = "deal_data"
        logging.info(",%s,ページ選択,%s", st.session_state.username, page)
    elif page == "グラフを表示":
        st.session_state.page = "vis"

    # --- page振り分け
    if st.session_state.page == "input_name":
        input_name()
    elif st.session_state.page == "deal_data":
        deal_data()
    elif st.session_state.page == "vis":
        vis()


# ---------------- usernameの登録 ----------------------------------
def input_name():
    # Input username
    with st.form("my_form"):
        inputname = st.text_input("学籍番号", "ここに学籍番号を入力")
        submitted = st.form_submit_button("Go!!")
        if submitted:  # Submit buttonn 押された時に
            if inputname == "ユーザ名" or input_name == "":  # nameが不適当なら
                submitted = False  # Submit 取り消し

        if submitted:
            st.session_state.username = inputname
            st.session_state.page = "deal_data"
            st.write("名前: ", inputname)
            st.text("↑ 自分の学籍番号であることを確認したら、もう一度 Go! をクリック")


# ---------------- 訓練データの加工 ----------------------------------
def deal_data():
    st.title("データの表示")
    full_df = load_full_data()

    # highlight の ON/OFF
    high_light = st.checkbox("最大値をハイライトする")
    if high_light:
        # dataframeを表示
        st.dataframe(full_df.style.highlight_max(axis=0))
    else:
        st.dataframe(full_df)


# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("タイタニック データ")

    feature_data = load_num_data()
    full_data = load_full_data()

    st.sidebar.markdown("## 様々なグラフを試してみよう")

    # sidebar でグラフを選択
    graph = st.sidebar.radio("グラフの種類", ("ヒストグラム", "散布図", "全ての散布図"))

    # ヒストグラム
    if graph == "ヒストグラム":
        logging.info(",%s,データ可視化,%s", st.session_state.username, graph)
        st.markdown("## データの分布を見てみよう")
        st.markdown("赤は生存者の分布、青は死亡者の分布を示す。")

        with st.form("ヒストグラム"):
            hist_val = st.selectbox("変数を選択", val_names)
            hist_val = data.get_val(hist_val)
            logging.info(",%s,ヒストグラム,%s", st.session_state.username, hist_val)

            # Submitボタン
            plot_button = st.form_submit_button("グラフ表示")

            if plot_button:
                if hist_val == "Fare":
                    g = px.histogram(
                        full_data,
                        x=hist_val,
                        nbins=20,
                        color="Survived",
                        barmode="group",
                    )
                    st.plotly_chart(g)
                elif hist_val == "Age":
                    g = px.histogram(
                        full_data,
                        x=hist_val,
                        nbins=10,
                        color="Survived",
                        barmode="group",
                    )
                    st.plotly_chart(g)
                else:
                    g = sns.catplot(
                        x=hist_val, y="Survived", data=full_data, kind="bar", ci=None
                    )
                    g = px.histogram(
                        full_data, x=hist_val, color="Survived", barmode="group"
                    )
                    st.plotly_chart(g)

        # コードの表示
        code = st.sidebar.checkbox("コードを表示")
        if code:
            code_txt = (
                "px.histogram(full_data, x="
                + hist_val
                + ", nbins=20, color='Survived', barmode='group')"
            )
            st.sidebar.markdown("---")
            st.sidebar.write(code_txt)
            st.sidebar.markdown("---")

    # 散布図
    elif graph == "散布図":
        logging.info(",%s,データ可視化,%s", st.session_state.username, graph)
        st.markdown("## 散布図 で 分布 を調べる")
        with st.form("散布図"):
            left, right = st.columns(2)

            with left:  # 変数選択
                x_label = st.selectbox("横軸を選択", val_names)
                x_label = data.get_val(x_label)

            with right:
                y_label = st.selectbox("縦軸を選択", val_names)
                y_label = data.get_val(y_label)

            logging.info(
                ",%s,散布図,%s", st.session_state.username, x_label + "_" + y_label
            )

            # Submitボタン
            plot_button = st.form_submit_button("グラフ表示")
            tmp = (
                full_data.copy()
            )  # ただ=で複製すると参照渡し → full_dataも値が書き変わる
            tmp.Survived.replace(DEAD, "dead", inplace=True)
            tmp.Survived.replace(ALIVE, "alive", inplace=True)

            if plot_button:
                # 散布図表示
                fig = px.scatter(data_frame=tmp, x=x_label, y=y_label, color="Survived")
                st.plotly_chart(fig, use_container_width=True)

        # コードの表示
        code = st.sidebar.checkbox("コードを表示")
        if code:
            code_txt = (
                "g = sns.catplot(x='"
                + x_label
                + "', y='"
                + y_label
                + "', data=full_data, kind = 'swarm')"
            )
            st.sidebar.markdown("---")
            st.sidebar.write(code_txt)
            st.sidebar.markdown("---")

    # 散布図行列
    if graph == "全ての散布図":
        logging.info(",%s,データ可視化,%s", st.session_state.username, graph)

        st.markdown("## 全ての変数 を 散布図 に表示する")
        st.markdown("このグラフの見方は、ページ一番下の「グラフの見方」ボタン参照")

        st.image(data.my_pairplot())

        # 散布図行列の見方を表示
        reference_btn = st.button("グラフの見方")
        if reference_btn:
            st.image(data.how_to_check())

        # コードの表示
        code = st.sidebar.checkbox("コードを表示")
        if code:
            code_txt = "g = sns.pairplot(full_data, hue='Survived', palette='seismic')"
            st.sidebar.markdown("---")
            st.sidebar.write(code_txt)
            st.sidebar.markdown("---")


main()
