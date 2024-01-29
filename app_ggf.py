def data_1(corpo):
    # ライブラリのインポート
    from time import sleep
    import time
    import requests
    from selenium.webdriver.common.keys import Keys
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from bs4 import BeautifulSoup
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import pandas as pd

    from datetime import datetime
    import seaborn as sns

    # Google Finance を開く
    # ChromeDriverの実行可能ファイルへの絶対パスを指定
    try:
        chrome_driver_path = "C:\\Users\\81808\\OneDrive\\デスクトップ\\My pands\\chromedriver-win32\\chromedriver.exe"
    except:
        pass
    # ChromeDriverのサービスオブジェクトを作成
    chrome_service = webdriver.chrome.service.Service(chrome_driver_path)

    # Google Financeのトップページを表示
    options = Options()
    options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.google.com/finance/?hl=ja")
    time.sleep(1)

    # 検索窓に"～～"を入力して銘柄のコードを取得する
    sele__1 = "#yDmH0d > c-wiz.zQTmif.SSPGKf.ccEnac > div > div.KdK6Xc > div.e1AOyf > div > div > div > div.d1dlne > input.Ax4B8.ZAGvjd"
    search_code_1 = driver.find_element(By.CSS_SELECTOR, sele__1)
    search_code_1.send_keys(corpo)
    search_code_1.send_keys(Keys.RETURN)
    time.sleep(1)
    # 検索完了

    # 銘柄コードを取得

    cur_url = driver.current_url

    # BeautifulSoupでHTMLを解析する

    url = cur_url
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")
    text_code_market = soup.find(class_="PdOqHc").get_text()
    # 株の銘柄を取得
    print(text_code_market[3:7])
    com_code = text_code_market[3:7]

    # 市場名を取得
    print(text_code_market[10:])
    market_name = text_code_market[10:]

    # 以下、application_gspから引用
    # import
    import gspread
    import googlefinance
    import gspread
    import json
    import pandas as pd

    # ServiceAccountCredentials：Googleの各サービスへアクセスできるservice変数を生成します。
    from oauth2client.service_account import ServiceAccountCredentials

    # 2つのAPIを記述しないとリフレッシュトークンを3600秒毎に発行し続けなければならない
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # 認証情報設定
    # ダウンロードしたjsonファイル名をクレデンシャル変数に設定（秘密鍵、Pythonファイルから読み込みしやすい位置に置く）
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "my-project-scr-403611-993e42646ecb.json", scope
    )

    # OAuth2の資格情報を使用してGoogle APIにログインします。
    gc = gspread.authorize(credentials)

    # 共有設定したスプレッドシートキーを変数[SPREADSHEET_KEY]に格納する。
    SPREADSHEET_KEY = "1ZPMtkZPOQIBgPAL3E9hfNu-i2S3KZSWoiATBI7-Orh0"

    # 共有設定したスプレッドシートのシート1を開く
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1

    # 入力する企業のコードを設定
    code_1 = market_name + ":" + com_code
    "NASDAQ:MSFT"

    # 自動入力する関数を設定

    formula = '=GOOGLEFINANCE("' + code_1 + '","price","2018/10/01",TODAY(),"DAILY")'
    # formula = '=GOOGLEFINANCE("NASDAQ:MSFT","price","2020/11/30","2020/12/30","DAILY")'
    # セルに関数を設定
    cell_range = "A1"  # 関数を入力したいセルの範囲を指定

    # value_input_option パラメータを使用して関数を評価モードに設定
    value_input_option = "USER_ENTERED"

    worksheet.update(cell_range, [[formula]], value_input_option=value_input_option)

    # スプレッドシートの値をデータフレーム型で取得
    data = worksheet.get_all_values()

    # データをDataFrameに変換
    df_data = pd.DataFrame(data[1:], columns=data[0])

    df_date_data = df_data.iloc[1:, 0]
    df_close_data = df_data.iloc[1:, 1]

    # df_date_dataの/を-に置換する

    df_date_data_replace = df_date_data.str.replace("/", "-")
    df_date_data_replace = df_date_data_replace["Date"].str[0:2]
    # データフレームを結合
    df_2dim = pd.concat([df_date_data_replace, df_close_data], axis=1)
    from sklearn.preprocessing import MinMaxScaler

    # 日付列を日付型に変換
    df_2dim["Date"] = pd.to_datetime(df_2dim["Date"])
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_2dim[["Close"]]),
        index=df_2dim.index,
        columns=["Close"],
    )
    # 正規化したCloseとdatetime型のDateを結合してdf_2dim_newにする
    df_2dim_new = pd.concat([df_2dim["Date"], df_scaled], axis=1)
    return df_2dim_new

    """    df_date_data = df_date_data.str.strip("16:00:00")
    df_date_data = df_date_data.str.strip("13:00:00")
    # ("13:00:00")以外があったらどうしよう... """


# 以下検証用
corpo_1 = "アマゾン"
data_before = data_1(corpo_1)
print(data_before)


def data_2(data_after):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # 学習データ、試験データに分離
    data_after.index = pd.to_datetime(data_after.index)
    df_week_scale_train = data_after["2018":"2022"]
    df_week_scale_test = data_after["Date"] > datetime(2023, 1, 1)

    look_back = 30  # どれくらい過去まで見るか

    def create_input_data(data, look_back):
        raw_data = data.T.values.tolist()
        data_size = len(data) - look_back

        X = [[] for i in range(len(raw_data))]
        y = [[] for i in range(len(raw_data))]

        for i in range(data_size):
            for j in range(len(raw_data)):
                X[j].append(raw_data[j][i : i + look_back])
                y[j].append([raw_data[j][i + look_back]])

        X_tmp = X[-1]
        y_tmp = y[-1]

        for i in range(len(raw_data) - 1):
            X_tmp = np.insert(
                X_tmp,
                np.arange(0, (look_back - 1) * (i + 1) + 1, i + 1),
                X[-i - 2],
                """ axis=1 """,
            )
            y_tmp = np.insert(
                y_tmp, np.arange(0, (i + 1), i + 1), y[-i - 2], """ axis=1 """
            )

        X = np.array(X_tmp).reshape(data_size, look_back, len(raw_data))
        y = np.array(y_tmp).reshape(data_size, 1, len(raw_data))

        return y, X

    y_train, X_train = create_input_data(data=df_week_scale_train, look_back=look_back)

    y_test, X_test = create_input_data(data=df_week_scale_test, look_back=look_back)

    # モデルの定義
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    x = model.add(
        LSTM(
            10, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(LSTM(8))
    model.add(Dense(1))  # 出力層はデータ数に合わせる

    model.compile(loss="mean_squared_error", optimizer="adam")

    # 実行
    history = model.fit(X_train, y_train, epochs=100, batch_size=1)

    # 未来予測
    future_steps = 50
    X_future = [y_test[-look_back:].values.tolist()]
    y_future_list = []

    for step in range(future_steps):
        y_future = model.predict(X_future)
        y_future_list.append(y_future[0])
        X_future = [X_future[0][1:] + y_future.tolist()]
    return df_week_scale_train, df_week_scale_test


data = data_2(data_before)
print(data)
