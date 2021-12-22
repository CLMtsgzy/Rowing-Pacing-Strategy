from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time

import traceback


def date2url(url_format, date_list):
    year = str(date_list[0])
    month = str(date_list[1])
    if len(month) == 1:
        month = '0' + month
    day = str(date_list[2])
    if len(day) == 1:
        day = '0' + day
    return url_format.format(year, month, day)

def date2str(date_list):
    return '{}-{}-{}'.format(*date_list)

if __name__ == '__main__':
    url_index = 'http://pc.weathercn.com/weather/long/103010100/?partner=2000001010_hfaw&areatype=smartid&p_source=search&p_type=landing'

    # chromedriver的路径
    chromedriver = '/Users/chuzhumin/Documents/webdriver/chromedriver_80'
    # chromedriver = '/work/czm/driver/chromedriver'
    os.environ["webdriver.chrome.driver"] = chromedriver
    # 设置chrome开启的模式，headless就是无界面模式
    # 一定要使用这个模式，不然截不了全页面，只能截到你电脑的高度
    chrome_options = Options()
    chrome_options.add_argument('headless')
    driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)

    try:

        driver.get(url_index)

        time.sleep(2)

        a1 = driver.switch_to.alert
        time.sleep(1)
        print(a1.text)
        a1.accept()
        time.sleep(1)


        element_dates = driver.find_elements_by_class_name("cal-body-cont")[0].find_elements_by_tag_name("a")

        url_list = []

        for element in element_dates:
            url_list.append(element.get_attribute("href"))

        print(url_list)


        for url in url_list:
            date = url.split("date=")[1].split("&")[0]

            driver.get(url)
            time.sleep(2)
            day_weather = driver.find_elements_by_class_name("weather-card-day")[0].find_elements_by_class_name("weather-infos")[0]

            day_text = day_weather.text
            items = day_text.split('\n')

            high_temp = items[2].split("℃")[0]
            low_temp = items[4].split("℃")[0]
            rainy_prob = items[6].split("%")[0]
            wind_low = items[9]+items[10]
            wind_high = items[11]+items[12]

            print('{}\tday\t{}\t{}\t{}\t{}\t{}'.format(date, high_temp, low_temp, rainy_prob, wind_low, wind_high))

            night_weather = driver.find_elements_by_class_name("weather-card-night")[0].find_elements_by_class_name("weather-infos")[0]

            night_text = night_weather.text
            items = night_text.split('\n')

            high_temp = items[2].split("℃")[0]
            low_temp = items[4].split("℃")[0]
            rainy_prob = items[6].split("%")[0]
            wind_low = items[9] + items[10]
            wind_high = items[11] + items[12]

            print('{}\tnight\t{}\t{}\t{}\t{}\t{}'.format(date, high_temp, low_temp, rainy_prob, wind_low, wind_high))

            # print(day_weather.text)
            #
            # print(date, url)

        print(len(element_dates))
    except:
        traceback.print_exc()

