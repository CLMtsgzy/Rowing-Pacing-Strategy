import os
import pickle as pkl

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

numbers_str = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
for i in range(10, 32):
    numbers_str.append(str(i))

print(numbers_str)

day_permonth = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def nextday_noyear(date, isRunNian=False): # date not contains year (01-31)
    month, day = date.split('-')
    month, day = int(month), int(day)
    if day < day_permonth[month]:
        day += 1
        return '{}-{}'.format(numbers_str[month], numbers_str[day]), 0
    elif month == 2 and isRunNian and day == 28:
        return '02-29', 0
    elif month == 12:
        return '01-01', 1
    else:
        return '{}-01'.format(numbers_str[month+1]), 0

def predday_noyear(date, isRunNian=False):
    month, day = date.split('-')
    month, day = int(month), int(day)
    if day > 1:
        day -= 1
        return '{}-{}'.format(numbers_str[month], numbers_str[day]), 0
    elif month == 3 and isRunNian:
        return '02-29', 0
    elif month == 1:
        return '12-31', 1
    else:
        return '{}-{}'.format(numbers_str[month-1], day_permonth[month-1]), 0


def RunNian(year):
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True


def nextday(date):
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)
    date_noyear = '{}-{}'.format(numbers_str[month], numbers_str[day])
    date_next, flag = nextday_noyear(date_noyear, RunNian(year))
    if flag == 0:
        return '{}-{}'.format(year, date_next)
    else:
        return '{}-{}'.format(year+1, date_next)


def predday(date):
    year, month, day = date.split('-')
    year, month, day = int(year), int(month), int(day)
    date_noyear = '{}-{}'.format(numbers_str[month], numbers_str[day])
    date_pred, flag = predday_noyear(date_noyear, RunNian(year))
    if flag == 0:
        return '{}-{}'.format(year, date_pred)
    else:
        return '{}-{}'.format(year - 1, date_pred)

print(sum(day_permonth))

if __name__ == "__main__":
    pm_value = 15

    start_day = '1979-03-01'
    end_day = '2020-03-14'

    print('1979-04-01' < start_day)

    while start_day < '1980-03-01':
        print(start_day)
        start_day = nextday(start_day)

    print()

    start_day = '1980-03-02'
    while start_day > '1979-03-01':
        print(start_day)
        start_day = predday(start_day)

    start_year = 1979
    end_year = 2020

    training_days = {}

    focus_date = '01-01'
    for i in range(366):
        training_days[focus_date] = []
        for year in range(start_year, end_year+1):
            if focus_date == '02-29' and not RunNian(year):
                center_day = '{}-02-28'.format(year)
            else:
                center_day = '{}-{}'.format(year, focus_date)

            dates_list = [center_day]
            temp = center_day
            for j in range(pm_value):
                temp = predday(temp)
                dates_list.append(temp)

            dates_list.reverse()

            temp = center_day
            for j in range(pm_value):
                temp = nextday(temp)
                dates_list.append(temp)

            if dates_list[0] >= start_day and dates_list[-1] <= end_day:
                training_days[focus_date].append(dates_list)

        focus_date, _ = nextday_noyear(focus_date, isRunNian=True)


    # for focus_date in training_days:
    #     print(focus_date)
    #     print(training_days[focus_date])

    print(training_days['02-29'])

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    save_object(training_days, './outputs/training_days.pkl')