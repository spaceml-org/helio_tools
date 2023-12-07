from datetime import datetime
from dateutil.relativedelta import relativedelta


def check_datetime_format(datetime_str: str) -> datetime:
    try:
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d")
        return datetime_obj
    except ValueError:
        msg = "Datetime string format is incorrect."
        msg += f"\nInput: {datetime_str}"
        print(msg)


def check_datetime_format_solo(datetime_str: str) -> datetime:
    try:
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        return datetime_obj
    except ValueError:
        msg = "Datetime string format is incorrect."
        msg += f"\nInput: {datetime_str}"
        print(msg)


def get_num_months(start_date: datetime, end_date: datetime) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


def get_month_dates(start_date: datetime, num_months: int) -> list:
    return [start_date + i * relativedelta(months=1) for i in range(num_months)]
