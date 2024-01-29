def function(corpo_func):
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
    import re
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import datetime

    # ChromeDriverの実行可能ファイルへの絶対パスを指定
    try:
        chrome_driver_path = "C:\\Users\\81808\\OneDrive\\デスクトップ\\My pands\\chromedriver-win32\\chromedriver.exe"
    except:
        pass
    # ChromeDriverのサービスオブジェクトを作成
    chrome_service = webdriver.chrome.service.Service(chrome_driver_path)

    # ブルームバーグのトップを表示
    options = Options()
    options.add_argument("--ignore-certificate-errors")
    driver = webdriver.Chrome(options=options)
    driver.get("https://www.bloomberg.co.jp/")

    time.sleep(1)

    # 検索ボックスへ入力
    sele_2 = "#nav-bar-search-button"
    search_mrk = driver.find_element(By.CSS_SELECTOR, sele_2)
    driver.execute_script("arguments[0].click();", search_mrk)
    sele_2_2 = "#gsc-i-id1"
    search_input = driver.find_element(By.CSS_SELECTOR, sele_2_2)

    search_input.send_keys(corpo_func)
    sele_2_3 = "#___gcse_0 > form > table > tbody > tr > td.gsc-search-button > button"
    search__click = driver.find_element(By.CSS_SELECTOR, sele_2_3)
    driver.execute_script("arguments[0].click();", search__click)
    time.sleep(1)

    # 要素をクリック
    sele_3 = "#___gcse_1 > div > div > div.gsc-results-wrapper-overlay.gsc-results-wrapper-visible > div.gsc-above-wrapper-area > table > tbody > tr > td.gsc-orderby-container > div > div.gsc-option-menu-container.gsc-inline-block > div.gsc-selected-option-container.gsc-inline-block"
    search_click = driver.find_element(By.CSS_SELECTOR, sele_3)

    driver.execute_script("arguments[0].click();", search_click)

    # 検索後、日付順に並べ替え
    sele_4 = "#___gcse_1 > div > div > div.gsc-results-wrapper-overlay.gsc-results-wrapper-visible > div.gsc-above-wrapper-area > table > tbody > tr > td.gsc-orderby-container > div > div.gsc-option-menu-container.gsc-inline-block > div.gsc-selected-option-container.gsc-inline-block"
    search_order_op = driver.find_element(By.CSS_SELECTOR, sele_4)
    driver.execute_script("arguments[0].click();", search_order_op)
    time.sleep(1)

    # Dateをクリック
    sele_5 = "#___gcse_1 > div > div > div.gsc-results-wrapper-overlay.gsc-results-wrapper-visible > div.gsc-above-wrapper-area > table > tbody > tr > td.gsc-orderby-container > div > div.gsc-option-menu-container.gsc-inline-block > div.gsc-option-menu > div:nth-child(2) > div"
    search_order_date = driver.find_element(By.CSS_SELECTOR, sele_5)
    driver.execute_script("arguments[0].click();", search_order_date)
    time.sleep(3)
    # 並べ替え完了

    # 今日の日付の取得
    tdy = datetime.date.today()
    tdy_1 = tdy - datetime.timedelta(days=1)
    tdy_2 = tdy - datetime.timedelta(days=2)

    tdy = str(tdy)
    tdy_1 = str(tdy_1)
    tdy_2 = str(tdy_2)

    # ブルームバーグのウェブページのURL
    url = "https://www.bloomberg.co.jp/"

    # ページの取得
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # for hoge in soup.find_all(href=re.compile("article")):
    # print(hoge.find_all(href=re.compile("article")))

    link = soup.find_all(href=re.compile("articles"))
    # 今日の日付の取得

    link_box = []
    for i in link:
        i = i.get("href")
        link_box.append(i)
    link_box_2 = []
    for u in link_box:
        url_link = "https://www.bloomberg.co.jp/" + u
        link_box_2.append(url_link)
        # print(url_link)

    list_1 = []
    for ur in link_box_2[:20]:
        url_text = ur
        if url_text[43:53] == tdy:
            res_text = requests.get(url_text)
            soup_text = BeautifulSoup(res_text.text, "html.parser")
            columns = soup_text.find_all(class_="body-copy")
            column_boxs = list(columns)
        else:
            column_boxs = []

        for cc in column_boxs:
            column_text = cc.get_text()
            list_1.append(column_text[:300])

    return list_1

    """     list_1 = []
    for ur in link_box_2:
        url_text = ur
        if url_text[43:53] == tdy:
            res_text = requests.get(url_text)
            soup_text = BeautifulSoup(res_text.text, "html.parser")
            columns = soup_text.find_all(class_="body-copy")
            column_boxs = list(columns)
        else:
            column_boxs = []

    for column in column_boxs:
        column = column.get_text()
        column_texts = list_1.append(column)

    return column_texts """


'corpo = "アリババ"'
"result = function(corpo)"
"print(result)"
