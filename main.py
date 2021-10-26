import streamlit as st
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import data
import time

LIVE = 1
DEAD = 0

st.set_page_config(
    page_title="Titanic Analysis App",
    layout="wide",
    initial_sidebar_state="collapsed",
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s,%(message)s")


DATA_SOURCE = './data/titanic.csv'

@st.cache
def load_full_data():
    data = pd.read_csv(DATA_SOURCE)
    return data

@st.cache 
def load_num_data():
    data = pd.read_csv(DATA_SOURCE)
    rows = ['Survived']
    data = data.drop(rows, axis=1)
    return data

@st.cache
def load_ML_data(feature1, feature2, train_num = 600):
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
    # If username is already initialized, don't do anything
    if 'username' not in st.session_state or st.session_state.username == 'default':
        st.session_state.username = 'default'
        input_name()
        st.stop()
    if 'username' not in st.session_state:
        st.session_state.username = 'test'
            
    if 'page' not in st.session_state:
        st.session_state.page = 'input_name' # usernameつける時こっち


    # --- page選択ラジオボタン
    st.sidebar.markdown('## ページを選択')
    page = st.sidebar.radio('', ('データ表示', 'グラフを表示'))
    if page == 'データ表示':
        st.session_state.page = 'deal_data'
        logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    elif page == 'グラフを表示':
        st.session_state.page = 'vis'
        # logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    # elif page == 'テストデータ':
    #     st.session_state.page = 'test'
    #     logging.info(',%s,ページ選択,%s', st.session_state.username, page)
    # elif page == '決定木':
    #     st.session_state.page = 'decision_tree'
    #     logging.info(',%s,ページ選択,%s', st.session_state.username, page)

    # --- page振り分け
    if st.session_state.page == 'input_name':
        input_name()
    elif st.session_state.page == 'deal_data':
        deal_data()
    elif st.session_state.page == 'vis':
        vis()
    elif st.session_state.page == 'test':
        test()  
    elif st.session_state.page == 'decision_tree':
        decision_tree()        

# ---------------- usernameの登録 ----------------------------------
def input_name():
    # Input username
    with st.form("my_form"):
        inputname = st.text_input('username', 'ユーザ名')
        submitted = st.form_submit_button("Go!!")
        if submitted: # Submit buttonn 押された時に
            if inputname == 'ユーザ名' or input_name == '': # nameが不適当なら
                submitted = False  # Submit 取り消し

        if submitted:
            st.session_state.username = inputname
            st.session_state.page = 'deal_data'
            st.write("名前: ", inputname)

# ---------------- 訓練データの加工 ----------------------------------
def deal_data():
    st.title("データの表示")
    full_df = load_full_data()

    # highlight の ON/OFF
    high_light = st.checkbox('最大値をハイライトする')
    if high_light:
        # dataframeを表示
        st.dataframe(full_df.style.highlight_max(axis=0))
    else:
        st.dataframe(full_df)


# ---------------- テストデータ　プロット ----------------------------------
def test():
    st.title('テストデータ')
    test_idx = st.number_input("データ番号を入力(0~200)", min_value=0, max_value=200)

    # テストデータを取得
    full_data = load_full_data()
    test_num = 200
    # test_numまでがテストデータ = 分割後もindexが揃う
    train = full_data[test_num:]
    # test_num以降が訓練データ
    test = full_data[:test_num]
    train.drop('Survived', axis=1) 
    
    # テストデータを取得
    test_df = full_data[test_idx: test_idx+1].drop('Survived', axis=1)
    # 選択したデータの表示
    st.dataframe(test_df)

    # 学習
    # ここでは決定木を用いる
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    train_X = train.drop('Survived', axis=1)
    train_y = train.Survived
    clf = clf.fit(train_X, train_y)
    # コンピューターの予測結果  # 1が生存、0が死亡
    pred = clf.predict(test_df)

    pred_btn = st.checkbox('予測結果をみる')
    if pred_btn:
        st.write('\n機械学習による予測結果は...')
        if pred[0] == 1:
            st.success('生存！！')
        else:
            st.success('亡くなってしまうかも...')
        
        # その後、正解を見る
        ans = st.checkbox('正解をみる')
        if ans:
            st.write('\n実際は...')
            if test['Survived'][test_idx] == 1:
                st.success('生存！！')
            else:
                st.success('亡くなってしまった...')

            test[test_idx: test_idx+1]


# ---------------- 決定木 : dtreeviz ----------------------------------
def decision_tree():
    st.title("生存できるか予測しよう")
    
    st.write('予測に使う変数を2つ選ぼう')
    left, right = st.beta_columns(2)
    features = ['Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']
    with left:
        feature1 = st.selectbox('予測に使う変数1',features)
    with right:
        feature2 = st.selectbox('予測に使う変数2',features)

    logging.info(',%s,決定木変数,%s', st.session_state.username, feature1+'_'+feature2)
    # 学習スタート
    started = st.button('学習スタート')
    if not started: 
        st.stop()
    
    # データの取得
    train_X, test_X, train_y, test_y = load_ML_data(feature1, feature2, train_num = 600)

    # 木の深さを3に制限
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    # 学習
    clf = clf.fit(train_X, train_y)

    # test_Xデータを全部予測する
    pred = clf.predict(test_X)
    # 正解率を計算する
    acc = accuracy_score(pred, test_y)

    st.success('学習終了！！')
    st.write(f'accuracy: {acc:.5f}')

    #　決定木の表示までにタイムラグがほしい
    # 待たせられる
    with st.spinner('Wait for it...'):
        time.sleep(3.5)

    # 決定木の可視化
    tree = data.my_dtree(feature1, feature2)
    st.image(tree, caption=feature1+'_'+feature2)


# ---------------- 可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("タイタニック データ")

    feature_data = load_num_data()
    full_data = load_full_data()
    label = feature_data.columns

    st.sidebar.markdown('## 様々なグラフを試してみよう')

    # sidebar でグラフを選択
    graph = st.sidebar.radio(
        'グラフの種類',
        ('棒グラフ', '棒グラフ(男女別)', '分布', '散布図', '全ての散布図')
    )

    # 棒グラフ
    if graph == "棒グラフ":
        logging.info(',%s,データ可視化,%s', st.session_state.username, graph)
        st.markdown('## 生存率 × 他の変数')
        with st.form("棒グラフ"):
            # 変数選択
            hist_val = st.selectbox('変数を選択',label)
            logging.info(',%s,棒グラフ,%s', st.session_state.username, hist_val)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                g = sns.catplot(x=hist_val, y='Survived', data=full_data, kind='bar', ci=None)
                g = g.set_ylabels("survival probability")
                st.pyplot(g)
        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "g = sns.catplot(x='" + hist_val + "', y='Survived', kind='bar', data=full_data, ci=None)"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')

    # 棒グラフ: Hue あり
    elif graph == "棒グラフ(男女別)":
        logging.info(',%s,データ可視化,%s', st.session_state.username, graph)
        # Genderを抜いたラベル
        label = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        st.markdown('## 生存率 × 他の変数')
        st.write('性別ごとの分類あり')
        with st.form("棒グラフ(男女別)"):
            # 変数選択
            hist_val = st.selectbox('変数を選択',label)
            logging.info(',%s,棒グラフ(男女別),%s', st.session_state.username, hist_val)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                g = sns.catplot(x=hist_val, y='Survived', data=full_data, hue='Gender', kind='bar', ci=None)
                st.pyplot(g)
        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "g = sns.catplot(x='" + hist_val + "', y='Survived', hue='Gender', data=full_data, kind='bar',  ci=None)"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')

    # 分布
    elif graph == "分布":
        logging.info(',%s,データ可視化,%s', st.session_state.username, graph)
        st.markdown('## 生存率 × 他の変数')

        with st.form("分布"):
            # 変数選択
            hist_val = st.selectbox('変数を選択',label)
            logging.info(',%s,分布,%s', st.session_state.username, hist_val)

            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            
            if plot_button:
                g = sns.displot(data=full_data, x=hist_val, hue="Survived",fill = True, kind="kde")
                g.set_axis_labels(hist_val, "survival probability")
                st.pyplot(g)

        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "g = sns.displot(data=full_data, x='" + hist_val + "', hue='Survived',fill = True, kind='kde')"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')
    
    # 散布図
    elif graph == '散布図':
        logging.info(',%s,データ可視化,%s', st.session_state.username, graph)
        st.markdown('## 散布図 で 分布 を調べる')
        with st.form("散布図"):
            left, right = st.beta_columns(2)

            with left: # 変数選択 
                x_label = st.selectbox('横軸を選択',label)

            with right:
                y_label = st.selectbox('縦軸を選択',label)
            
            logging.info(',%s,散布図,%s', st.session_state.username, x_label+'_'+ y_label)
        
            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # 散布図表示
                g = sns.catplot(x=x_label, y=y_label, data=full_data, kind = 'swarm')
                st.pyplot(g)

        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "g = sns.catplot(x='" +  x_label + "', y='" + y_label + "', data=full_data, kind = 'swarm')"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')
    
    # 散布図行列
    if graph == '全ての散布図':
        logging.info(',%s,データ可視化,%s', st.session_state.username, graph)

        st.markdown('## 全ての変数 を 散布図 に表示する')
        st.markdown('このグラフの見方は、ページの一番下にある「グラフの見方」ボタン参照')

        with st.form("散布図行列"):
            # Submitボタン
            plot_button = st.form_submit_button('グラフ表示')
            if plot_button:
                # sns.pairplot(full_data, hue='Survived').savefig('./all_pairplot.png')
                # st.pyplot(g)
                # 処理時間が長すぎるので画像を表示
                st.image(data.my_pairplot())
            
        # 散布図行列の見方を表示
        reference_btn = st.button('グラフの見方')
        if reference_btn:
            st.markdown('ここにスライドの画像を表示')

        # コードの表示
        code = st.sidebar.checkbox('コードを表示')
        if code:
            code_txt = "g = sns.pairplot(full_data, hue='Survived')"
            st.sidebar.markdown('---')
            st.sidebar.write(code_txt)
            st.sidebar.markdown('---')

main()