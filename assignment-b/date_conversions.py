from datetime import datetime

import re

long_date       = r"(January|February|March|April|May|June|July|August|September|October|November|December) (0?[1-9]|1[0-9]|2[0-9]|30|31), (19|20)[0-9][0-9]"
short_date      = r"(0?[1-9]|1[0-2])(-|\/)(0?[1-9]|1[0-9]|2[0-9]|30|31)\2(19|20)[0-9][0-9]"
extra_long_date = r"the (1st|2nd|3rd|[4-9]th|1[0-9]th|20th|21st|22nd|23rd|2[4-9]th|30th|31st) of (January|February|March|April|May|June|July|August|September|October|November|December), (19|20)[0-9][0-9]"

def convert_long_date(long_date):  
    return datetime.strptime(long_date, '%B %d, %Y').strftime('%Y-%-m-%-d')

def convert_short_date(short_date):
    short_date = short_date.replace('/', '-')
    return datetime.strptime(short_date, '%m-%d-%Y').strftime('%Y-%-m-%-d')

def convert_extra_long_date(extra_long_date):
    extra_long_date = extra_long_date.replace('the ', '')
    extra_long_date = extra_long_date.replace('th', '')
    extra_long_date = extra_long_date.replace('st', '')
    extra_long_date = extra_long_date.replace('nd', '')
    extra_long_date = extra_long_date.replace('rd', '')
    return datetime.strptime(extra_long_date, '%d of %B, %Y').strftime('%Y-%-m-%-d')

date_conversions = {
    'long_date': (long_date, convert_long_date),
    'short_date': (short_date, convert_short_date),
    'extra_long_date': (extra_long_date, convert_extra_long_date)
}

def convert(date, conversion_dict=date_conversions):
    for key in conversion_dict:
        if re.match(conversion_dict[key][0], date):
            return conversion_dict[key][1](date)
    return None

print(convert('September 7, 2011'))
print(convert('09-07-2011'))
print(convert('the 7th of September, 2011')) 
## returns 2011-9-7 for all tests


