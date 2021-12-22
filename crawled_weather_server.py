from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time

import traceback

from datetime import datetime

def isRunNian(year):
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True

days_nums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

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


def nextday(date):
    year, month, day = date[0], date[1], date[2]
    if day < days_nums[month-1]:
        return [year, month, day+1]
    else:
        if month == 12:
            return [year+1, 1, 1]
        elif month == 2:
            if not isRunNian(year):
                return [year, 3, 1]
            else:
                if day == 28:
                    return [year, 2, 29]
                else:
                    return [year, 3, 1]
        else:
            return [year, month+1, 1]


if __name__ == '__main__':
    url_format = 'https://www.weathertab.com/zh-hans/long-range-weather/{}/{}/{}/japan/tokyo-to/tokyo/?ref=g'

    now = datetime.now()

    dates_list = []

    print(now)
    start_date_str = str(now)[:10]
    print(start_date_str)

    year, month, day = start_date_str.split('-')
    year, month, day = int(year), int(month), int(day)

    start_date = [year, month, day]

    cnt_total = 500
    dates_list.append(start_date)
    for i in range(cnt_total):
        temp = [year, month, day]
        year, month, day = nextday(temp)
        dates_list.append([year, month, day])

    print(dates_list)




    # dates_list = [[2020, 7, 23], [2020, 7, 24], [2020, 7, 25], [2020, 7, 26], [2020, 7, 27], [2020, 7, 28], [2020, 7, 29], [2020, 7, 30], [2020, 7, 31],
    #               [2020, 8, 1], [2020, 8, 2], [2020, 8, 3], [2020, 8, 4], [2020, 8, 5], [2020, 8, 6], [2020, 8, 7],
    #               [2020, 8, 8],
    #               [2021, 7, 23], [2021, 7, 24], [2021, 7, 25], [2021, 7, 26], [2021, 7, 27], [2021, 7, 28],
    #               [2021, 7, 29], [2021, 7, 30], [2021, 7, 31],
    #               [2021, 8, 1], [2021, 8, 2], [2021, 8, 3], [2021, 8, 4], [2021, 8, 5], [2021, 8, 6], [2021, 8, 7],
    #               [2021, 8, 8]]
    #
    # print(date2url('{}-{}-{}', dates_list[0]))
    #
    # chromedriver的路径
    chromedriver = '/home/ubuntu/ufozgg/rowing/driver/chromedriver'
    # chromedriver = '/work/czm/driver/chromedriver'
    os.environ["webdriver.chrome.driver"] = chromedriver
    # 设置chrome开启的模式，headless就是无界面模式
    # 一定要使用这个模式，不然截不了全页面，只能截到你电脑的高度
    chrome_options = Options()
    chrome_options.add_argument('headless')
    driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)

    main_window = driver.current_window_handle

    weathers_list = []

    for i in range(len(dates_list)):
        date = dates_list[i]
        url = date2url(url_format, date)

        print(url)
        try:

            # 控制浏览器写入并转到链接
            driver.get(url)
            time.sleep(3)

            element = driver.find_elements_by_class_name("col-md-8")[0]

            slices = element.find_elements_by_tag_name("tr")

            rainy_prob = slices[1].find_elements_by_tag_name("div")[0].text
            rainy_prob = int(rainy_prob.split('%')[0])
            print('降水概率 = {}%'.format(rainy_prob))

            high_temp = slices[2].find_elements_by_css_selector("[class='C label label-danger']")[0].text
            low_temp = slices[2].find_elements_by_css_selector("[class='C label label-primary']")[0].text

            high_temp = high_temp.split("高温预报")[1].split("°C")[0]
            low_temp = low_temp.split("低温预报")[1].split("°C")[0]

            high_temp_lc, high_temp_hc = high_temp.split('至')
            high_temp_lc, high_temp_hc = int(high_temp_lc), int(high_temp_hc)

            low_temp_lc, low_temp_hc = low_temp.split('至')
            low_temp_lc, low_temp_hc = int(low_temp_lc), int(low_temp_hc)



            print('高温预报：{}~{}'.format(high_temp_lc, high_temp_hc))
            print('低温预报：{}~{}'.format(low_temp_lc, low_temp_hc))

            weathers_list.append([date2url('{}-{}-{}', dates_list[i]), rainy_prob, high_temp_lc, high_temp_hc, low_temp_lc, low_temp_hc])


        except:
            traceback.print_exc()


    time.sleep(5)

    print('date\trainy_prob\thigh_temp_lc\thigh_temp_hc\tlow_temp_lc\tlow_temp_hc')
    for info in weathers_list:
        print('\t'.join([str(item) for item in info]))


    start_date_str = start_date_str.replace('-', '')
    path_current = './crawled/weathertab_{}'.format(start_date_str)

    with open(path_current, 'w') as f:
        f.write('date\trainy_prob\thigh_temp_lc\thigh_temp_hc\tlow_temp_lc\tlow_temp_hc\n')
        for info in weathers_list:
            f.write('\t'.join([str(item) for item in info])+'\n')